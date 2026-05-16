"""
app.py  —  Market Movement Prediction  (Flask API + Frontend)
============================================================
Routes:
  GET  /           → HTML dashboard
  GET  /predict    → JSON predictions for all tickers + models

Run:
  pip install flask yfinance vaderSentiment feedparser torch scikit-learn
  python app.py
"""

import os, warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import feedparser
from flask import Flask, jsonify, render_template_string
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
#  CONFIG  —  must match training notebook exactly
# ══════════════════════════════════════════════════════════════════
TICKERS      = ["AAPL", "TSLA", "MSFT", "AMZN", "GOOGL"]
FEATURE_COLS = [
    "Open","High","Low","Close","Volume",
    "MA_7","MA_21","RSI","MACD","Volatility",
    "avg_sentiment","positive_ratio","negative_ratio",
    "sentiment_momentum","news_count"
]
SEQ_LEN = 10
IN  = len(FEATURE_COLS)   # 15
HID = 128
NL  = 2
DO  = 0.3
NC  = 2

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR=BASE_DIR
analyzer   = SentimentIntensityAnalyzer()

# ══════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS  (identical to notebook)
# ══════════════════════════════════════════════════════════════════
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn  = nn.RNN(IN, HID, NL, batch_first=True, dropout=DO)
        self.bn   = nn.BatchNorm1d(HID)
        self.drop = nn.Dropout(DO)
        self.fc   = nn.Linear(HID, NC)
    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(self.drop(self.bn(o[:, -1, :])))

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(IN, HID, NL, batch_first=True, dropout=DO)
        self.bn   = nn.BatchNorm1d(HID)
        self.drop = nn.Dropout(DO)
        self.fc   = nn.Linear(HID, NC)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(self.drop(self.bn(o[:, -1, :])))

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru  = nn.GRU(IN, HID, NL, batch_first=True, dropout=DO)
        self.bn   = nn.BatchNorm1d(HID)
        self.drop = nn.Dropout(DO)
        self.fc   = nn.Linear(HID, NC)
    def forward(self, x):
        o, _ = self.gru(x)
        return self.fc(self.drop(self.bn(o[:, -1, :])))

# ══════════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════════
def load_model(cls, path):
    model = cls().to(DEVICE)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"  ✅ Loaded: {path}")
    else:
        print(f"  ⚠️  Not found (random weights): {path}")
    model.eval()
    return model

print("Loading models …")
MODELS = {
    "RNN" : load_model(RNNModel,  os.path.join(MODEL_DIR, "model_rnn.pth")),
    "LSTM": load_model(LSTMModel, os.path.join(MODEL_DIR, "model_lstm.pth")),
    "GRU" : load_model(GRUModel,  os.path.join(MODEL_DIR, "model_gru.pth")),
}
print("All models ready ✅\n")

# ══════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (mirrors notebook Cell 6)
# ══════════════════════════════════════════════════════════════════
def compute_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / (l + 1e-8)))

def get_stock_features(ticker, lookback_days=60):
    """Download recent OHLCV and compute all technical indicators."""
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), progress=False)
    if df.empty:
        return None
    df.reset_index(inplace=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df["Pct_Change"] = df["Close"].pct_change() * 100
    df["Volatility"] = df["Pct_Change"].rolling(5).std()
    df["MA_7"]       = df["Close"].rolling(7).mean()
    df["MA_21"]      = df["Close"].rolling(21).mean()
    df["RSI"]        = compute_rsi(df["Close"])
    df["MACD"]       = (df["Close"].ewm(span=12).mean()
                        - df["Close"].ewm(span=26).mean())
    return df.dropna().reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════
#  SENTIMENT  (live RSS)
# ══════════════════════════════════════════════════════════════════
RSS_URLS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"
]

def get_sentiment(ticker):
    """Fetch RSS headlines for ticker and return VADER scores + headlines list."""
    headlines, scores = [], []
    url = RSS_URLS[0].format(t=ticker)
    try:
        feed = feedparser.parse(url)
        for e in feed.entries[:10]:
            title = e.get("title", "")
            score = analyzer.polarity_scores(title)["compound"]
            headlines.append({"title": title, "score": round(score, 3)})
            scores.append(score)
    except Exception:
        pass
    avg   = float(np.mean(scores)) if scores else 0.0
    pos_r = sum(1 for s in scores if s >= 0.05) / max(len(scores), 1)
    neg_r = sum(1 for s in scores if s <= -0.05) / max(len(scores), 1)
    return avg, pos_r, neg_r, len(scores), headlines

# ══════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════
def predict_ticker(ticker):
    df = get_stock_features(ticker)
    if df is None or len(df) < SEQ_LEN:
        return None

    avg_sent, pos_r, neg_r, n_news, headlines = get_sentiment(ticker)
    sent_mom = avg_sent   # simplified: use same value for momentum

    # Attach sentiment columns
    df["avg_sentiment"]      = avg_sent
    df["positive_ratio"]     = pos_r
    df["negative_ratio"]     = neg_r
    df["sentiment_momentum"] = sent_mom
    df["news_count"]         = n_news

    # Keep only feature columns that exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = 0.0

    scaler = MinMaxScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    # Take the last SEQ_LEN rows as input sequence
    seq = df[FEATURE_COLS].values[-SEQ_LEN:]
    x   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    results = {}
    with torch.no_grad():
        for name, model in MODELS.items():
            logits = model(x)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred   = int(np.argmax(probs))
            conf   = float(probs[pred])
            results[name] = {
                "direction"  : "UP 📈" if pred == 1 else "DOWN 📉",
                "direction_raw": "UP" if pred == 1 else "DOWN",
                "confidence" : round(conf * 100, 1),
                "prob_up"    : round(float(probs[1]) * 100, 1),
                "prob_down"  : round(float(probs[0]) * 100, 1),
            }

    latest = df[["Open","High","Low","Close","Volume"]].iloc[-1]
    raw_row = scaler.inverse_transform(df[FEATURE_COLS].values)
    close_price = raw_row[-1][FEATURE_COLS.index("Close")]

    return {
        "ticker"    : ticker,
        "close"     : round(float(close_price), 2),
        "sentiment" : {
            "avg_score"  : round(avg_sent, 3),
            "label"      : ("Positive 😊" if avg_sent >= 0.05 else
                            "Negative 😟" if avg_sent <= -0.05 else "Neutral 😐"),
            "pos_ratio"  : round(pos_r * 100, 1),
            "neg_ratio"  : round(neg_r * 100, 1),
            "news_count" : n_news,
        },
        "headlines" : headlines[:5],
        "models"    : results,
        "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# ══════════════════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════════════════
app = Flask(__name__)

# ── HTML Template ─────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Market Prediction Dashboard</title>
<style>
  :root {
    --bg: #0f1117; --card: #1a1d2e; --card2: #222540;
    --accent: #6c63ff; --up: #00c896; --down: #ff4d6d;
    --neutral: #a0a8c0; --text: #e8eaf0; --sub: #8890a8;
    --border: #2d3150; --shadow: 0 4px 24px rgba(0,0,0,0.4);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; min-height: 100vh; }

  /* ── Header ── */
  .header {
    background: linear-gradient(135deg, #1a1d2e 0%, #222540 100%);
    border-bottom: 1px solid var(--border);
    padding: 24px 40px; display: flex; align-items: center;
    justify-content: space-between;
  }
  .header h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.5px; }
  .header h1 span { color: var(--accent); }
  .badge { background: var(--accent); color: #fff; font-size: 0.72rem;
           padding: 4px 12px; border-radius: 20px; font-weight: 600; }
  #clock { color: var(--sub); font-size: 0.85rem; margin-top: 4px; }
  .controls { display: flex; gap: 12px; align-items: center; }
  .btn-refresh {
    background: var(--accent); color: #fff; border: none; cursor: pointer;
    padding: 10px 22px; border-radius: 8px; font-size: 0.9rem; font-weight: 600;
    transition: opacity 0.2s;
  }
  .btn-refresh:hover { opacity: 0.85; }
  .btn-refresh:disabled { opacity: 0.5; cursor: not-allowed; }

  /* ── Main ── */
  .main { padding: 32px 40px; max-width: 1400px; margin: 0 auto; }
  .section-title { font-size: 1rem; font-weight: 600; color: var(--sub);
                   text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px; }

  /* ── Ticker grid ── */
  .ticker-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
                 gap: 24px; margin-bottom: 40px; }

  /* ── Ticker card ── */
  .ticker-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; padding: 24px; box-shadow: var(--shadow);
    transition: transform 0.2s;
  }
  .ticker-card:hover { transform: translateY(-2px); }
  .tc-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 18px; }
  .tc-ticker { font-size: 1.5rem; font-weight: 800; color: #fff; }
  .tc-price { font-size: 1.1rem; font-weight: 600; color: var(--accent); }
  .tc-time { font-size: 0.72rem; color: var(--sub); margin-top: 2px; }

  /* ── Sentiment pill ── */
  .sent-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
    margin-bottom: 16px;
  }
  .sent-pos { background: rgba(0,200,150,0.15); color: var(--up); border: 1px solid rgba(0,200,150,0.3); }
  .sent-neg { background: rgba(255,77,109,0.15); color: var(--down); border: 1px solid rgba(255,77,109,0.3); }
  .sent-neu { background: rgba(160,168,192,0.15); color: var(--neutral); border: 1px solid rgba(160,168,192,0.3); }

  /* ── Model prediction cards ── */
  .model-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 18px; }
  .model-card {
    background: var(--card2); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 12px; text-align: center;
  }
  .model-name { font-size: 0.7rem; font-weight: 700; color: var(--sub);
                text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
  .model-dir { font-size: 1.05rem; font-weight: 800; margin-bottom: 6px; }
  .dir-up   { color: var(--up); }
  .dir-down { color: var(--down); }
  .conf-bar-wrap { background: #111320; border-radius: 4px; height: 5px; overflow: hidden; margin-bottom: 5px; }
  .conf-bar { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
  .bar-up   { background: var(--up); }
  .bar-down { background: var(--down); }
  .conf-label { font-size: 0.75rem; color: var(--sub); }
  .conf-pct { font-size: 0.9rem; font-weight: 700; color: var(--text); }

  /* ── Headlines ── */
  .headlines-title { font-size: 0.75rem; font-weight: 700; color: var(--sub);
                     text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px; }
  .headline-item { display: flex; justify-content: space-between; align-items: flex-start;
                   gap: 10px; padding: 8px 0; border-bottom: 1px solid var(--border); }
  .headline-item:last-child { border-bottom: none; }
  .hl-text { font-size: 0.8rem; color: var(--text); line-height: 1.4; flex: 1; }
  .hl-score { font-size: 0.75rem; font-weight: 700; white-space: nowrap; padding: 2px 8px;
              border-radius: 10px; }
  .score-pos { background: rgba(0,200,150,0.15); color: var(--up); }
  .score-neg { background: rgba(255,77,109,0.15); color: var(--down); }
  .score-neu { background: rgba(160,168,192,0.1); color: var(--neutral); }

  /* ── Loader ── */
  .loader-wrap { display: flex; flex-direction: column; align-items: center;
                 justify-content: center; padding: 80px; gap: 16px; }
  .spinner { width: 48px; height: 48px; border: 4px solid var(--border);
             border-top-color: var(--accent); border-radius: 50%;
             animation: spin 0.9s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loader-txt { color: var(--sub); font-size: 0.9rem; }

  /* ── Error ── */
  .error-box { background: rgba(255,77,109,0.1); border: 1px solid rgba(255,77,109,0.3);
               border-radius: 12px; padding: 20px; color: var(--down); text-align: center; }

  /* ── Consensus bar at bottom of card ── */
  .consensus { margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border); }
  .cons-label { font-size: 0.72rem; color: var(--sub); margin-bottom: 6px; }
  .cons-badges { display: flex; gap: 8px; flex-wrap: wrap; }
  .cons-badge { padding: 3px 10px; border-radius: 10px; font-size: 0.75rem; font-weight: 700; }
  .cons-up   { background: rgba(0,200,150,0.15); color: var(--up); }
  .cons-down { background: rgba(255,77,109,0.15); color: var(--down); }

  @media (max-width: 700px) {
    .header { padding: 16px 20px; flex-wrap: wrap; gap: 10px; }
    .main   { padding: 20px; }
    .ticker-grid { grid-template-columns: 1fr; }
    .model-row { grid-template-columns: repeat(3, 1fr); }
  }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>📊 <span>Market</span> Prediction Dashboard</h1>
    <div id="clock">Loading…</div>
  </div>
  <div class="controls">
    <span class="badge">RNN · LSTM · GRU</span>
    <button class="btn-refresh" id="btnRefresh" onclick="loadPredictions()">🔄 Refresh</button>
  </div>
</div>

<div class="main">
  <div class="section-title">Next-Day Direction Predictions</div>
  <div class="ticker-grid" id="grid">
    <div class="loader-wrap">
      <div class="spinner"></div>
      <div class="loader-txt">Fetching live data & running models…</div>
    </div>
  </div>
</div>

<script>
// ── Clock ──────────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent =
    'Last updated: ' + now.toLocaleString();
}
setInterval(updateClock, 1000);
updateClock();

// ── Helpers ───────────────────────────────────────────────────────────
function sentClass(label) {
  if (label.includes('Positive')) return 'sent-pos';
  if (label.includes('Negative')) return 'sent-neg';
  return 'sent-neu';
}
function scoreClass(s) {
  if (s >= 0.05)  return 'score-pos';
  if (s <= -0.05) return 'score-neg';
  return 'score-neu';
}
function dirClass(d) { return d === 'UP' ? 'dir-up' : 'dir-down'; }
function barClass(d) { return d === 'UP' ? 'bar-up' : 'bar-down'; }

// ── Render one ticker card ─────────────────────────────────────────────
function renderCard(data) {
  const s = data.sentiment;
  const m = data.models;

  // Model rows
  let modelHTML = '';
  for (const [name, pred] of Object.entries(m)) {
    modelHTML += `
      <div class="model-card">
        <div class="model-name">${name}</div>
        <div class="model-dir ${dirClass(pred.direction_raw)}">${pred.direction}</div>
        <div class="conf-bar-wrap">
          <div class="conf-bar ${barClass(pred.direction_raw)}"
               style="width:${pred.confidence}%"></div>
        </div>
        <div class="conf-pct">${pred.confidence}%</div>
        <div class="conf-label">↑${pred.prob_up}% ↓${pred.prob_down}%</div>
      </div>`;
  }

  // Headlines
  let hlHTML = '';
  if (data.headlines.length) {
    data.headlines.forEach(h => {
      hlHTML += `
        <div class="headline-item">
          <span class="hl-text">${h.title}</span>
          <span class="hl-score ${scoreClass(h.score)}">${h.score > 0 ? '+' : ''}${h.score}</span>
        </div>`;
    });
  } else {
    hlHTML = '<div style="color:var(--sub);font-size:0.8rem">No live headlines found.</div>';
  }

  // Consensus
  const ups   = Object.values(m).filter(p => p.direction_raw === 'UP').length;
  const downs = Object.values(m).filter(p => p.direction_raw === 'DOWN').length;
  let consBadges = '';
  if (ups)   consBadges += `<span class="cons-badge cons-up">📈 UP ×${ups}</span>`;
  if (downs) consBadges += `<span class="cons-badge cons-down">📉 DOWN ×${downs}</span>`;

  return `
    <div class="ticker-card">
      <div class="tc-header">
        <div>
          <div class="tc-ticker">${data.ticker}</div>
          <div class="tc-time">${data.timestamp}</div>
        </div>
        <div class="tc-price">$${data.close.toLocaleString()}</div>
      </div>

      <div class="sent-pill ${sentClass(s.label)}">
        ${s.label} &nbsp;|&nbsp; score ${s.avg_sentiment > 0 ? '+' : ''}${s.avg_sentiment}
        &nbsp;|&nbsp; ${s.news_count} headlines
      </div>

      <div class="model-row">${modelHTML}</div>

      <div class="headlines-title">📰 Latest Headlines</div>
      ${hlHTML}

      <div class="consensus">
        <div class="cons-label">Model Consensus (${Object.keys(m).length} models)</div>
        <div class="cons-badges">${consBadges}</div>
      </div>
    </div>`;
}

// ── Fetch & render ─────────────────────────────────────────────────────
async function loadPredictions() {
  const grid = document.getElementById('grid');
  const btn  = document.getElementById('btnRefresh');
  btn.disabled = true;
  btn.textContent = '⏳ Loading…';

  grid.innerHTML = `
    <div class="loader-wrap" style="grid-column:1/-1">
      <div class="spinner"></div>
      <div class="loader-txt">Fetching live prices, sentiment & running models…</div>
    </div>`;

  try {
    const res  = await fetch('/predict');
    const data = await res.json();

    if (data.error) {
      grid.innerHTML = `<div class="error-box" style="grid-column:1/-1">
        ❌ ${data.error}</div>`;
      return;
    }

    grid.innerHTML = data.predictions
      .map(d => d.error
        ? `<div class="ticker-card"><div class="tc-ticker">${d.ticker}</div>
           <div class="error-box" style="margin-top:12px">⚠️ ${d.error}</div></div>`
        : renderCard(d))
      .join('');

  } catch (err) {
    grid.innerHTML = `<div class="error-box" style="grid-column:1/-1">
      ❌ Network error: ${err.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '🔄 Refresh';
    updateClock();
  }
}

// Load on page start
loadPredictions();
</script>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict")
def predict():
    predictions = []
    for ticker in TICKERS:
        try:
            result = predict_ticker(ticker)
            if result is None:
                predictions.append({"ticker": ticker, "error": "Not enough data"})
            else:
                predictions.append(result)
        except Exception as e:
            predictions.append({"ticker": ticker, "error": str(e)})
    return jsonify({"predictions": predictions, "count": len(predictions)})


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Market Prediction Dashboard")
    print("  http://127.0.0.1:5000")
    print("  Place model_rnn.pth / model_lstm.pth / model_gru.pth")
    print("  in the same folder as app.py")
    print("="*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
