# Real-Time Market Movement Prediction System using Deep Learning

A real-time financial market prediction system that predicts next-day stock market movement using deep learning models including **RNN, LSTM, and GRU**.
The project combines **technical indicators**, **live stock market data**, and **news sentiment analysis** to generate intelligent market predictions through an interactive web dashboard.

---

# 📌 Features

* 📈 Real-time stock market prediction
* 🤖 Deep Learning Models:

  * RNN
  * LSTM
  * GRU
* 📰 Live financial news sentiment analysis
* 📊 Technical indicators:

  * RSI
  * MACD
  * Moving Averages
  * Volatility
* 🌐 Interactive Flask-based frontend dashboard
* ⚡ Real-time prediction API
* 📉 Confidence score visualization
* 📡 Yahoo Finance live market data integration

---

# 🛠️ Technologies Used

* Python
* Flask
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* yFinance
* VADER Sentiment Analysis
* Feedparser
* HTML/CSS/JavaScript

---

# 📂 Project Structure

```bash
├── app.py
├── model_rnn.pth
├── model_lstm.pth
├── model_gru.pth
├── real-time-market-movement-prediction-system.ipynb
├── README.md
```

---

# 🧠 Deep Learning Models

The system uses three sequential deep learning architectures:

| Model | Purpose                            |
| ----- | ---------------------------------- |
| RNN   | Captures sequential stock patterns |
| LSTM  | Handles long-term dependencies     |
| GRU   | Efficient sequence learning        |

Each model predicts whether the stock price will move:

* 📈 UP
* 📉 DOWN

---

# 📊 Input Features

The models are trained using:

## Stock Market Features

* Open
* High
* Low
* Close
* Volume

## Technical Indicators

* RSI
* MACD
* Moving Average (7)
* Moving Average (21)
* Volatility

## Sentiment Features

* Average Sentiment Score
* Positive News Ratio
* Negative News Ratio
* Sentiment Momentum
* News Count

---

# 📰 Sentiment Analysis

The project performs live sentiment analysis on financial news headlines using:

* Yahoo Finance RSS Feeds
* VADER Sentiment Analyzer

News headlines are classified as:

* Positive 😊
* Neutral 😐
* Negative 😟

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/market-prediction-system.git
cd market-prediction-system
```

---

## 2. Install Dependencies

```bash
pip install flask yfinance vaderSentiment feedparser torch scikit-learn pandas numpy
```

---

## 3. Run Application

```bash
python app.py
```

---

# 🌐 Open Dashboard

After running the application:

```bash
http://127.0.0.1:5000
```

---

# 📷 Dashboard Features

The frontend dashboard displays:

* Live stock prices
* AI-based predictions
* Confidence scores
* Sentiment analysis
* Latest news headlines
* Model consensus

---

# 📡 API Endpoint

## Get Predictions

```bash
GET /predict
```

Returns JSON predictions for all stocks.

---

# 📈 Example Prediction Output

```json
{
  "ticker": "AAPL",
  "direction": "UP",
  "confidence": 92.4
}
```

---

# 🎯 Supported Stocks

* Apple (AAPL)
* Tesla (TSLA)
* Microsoft (MSFT)
* Amazon (AMZN)
* Google (GOOGL)

---

# 🔮 Future Improvements

* Add Transformer-based models
* Add candlestick charts
* Deploy on cloud platform
* Add cryptocurrency prediction
* Add portfolio analysis
* Add real-time websocket updates

---

# 👨‍💻 Contributors

* Abdullah Saood
* Saim Ahmad

---

# 📄 License

This project is developed for educational and research purposes.

---

# ⭐ Acknowledgements

* Yahoo Finance
* PyTorch
* Flask
* VADER Sentiment
* Scikit-learn
