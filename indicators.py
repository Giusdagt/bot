"" 
indicatori per scalping e historical
""

import numpy as np
import polars as pl  # ✅ Usiamo polars invece di pandas
import logging
import requests
import talib

# 📌 Configurazione del logging avanzato
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# 📌 URL per l'analisi del sentiment
SENTIMENT_API_URL = "https://your-sentiment-api.com/analyze"


def calculate_indicators(data):
    """Calcola tutti gli indicatori tecnici principali e li aggiunge ai dati di mercato."""

    # 📌 RSI (Relative Strength Index) per trend reversal e scalping
    data = data.with_columns(pl.Series("RSI", talib.RSI(data["close"].to_numpy(), timeperiod=14)))

    # 📌 Bollinger Bands per volatilità
    upper_band, middle_band, lower_band = talib.BBANDS(
        data["close"].to_numpy(), timeperiod=20, nbdevup=2, nbdevdn=2)
    data = data.with_columns([
        pl.Series("BB_Upper", upper_band),
        pl.Series("BB_Middle", middle_band),
        pl.Series("BB_Lower", lower_band)
    ])

    # 📌 MACD per trend detection e momentum
    macd, macdsignal, macdhist = talib.MACD(
        data["close"].to_numpy(), fastperiod=12, slowperiod=26, signalperiod=9)
    data = data.with_columns([
        pl.Series("MACD", macd),
        pl.Series("MACD_Signal", macdsignal),
        pl.Series("MACD_Hist", macdhist)
    ])

    # 📌 EMA e SMA per trend-following
    data = data.with_columns([
        pl.Series("EMA_50", talib.EMA(data["close"].to_numpy(), timeperiod=50)),
        pl.Series("EMA_200", talib.EMA(data["close"].to_numpy(), timeperiod=200)),
        pl.Series("SMA_100", talib.SMA(data["close"].to_numpy(), timeperiod=100))
    ])

    # 📌 ADX per forza del trend
    data = data.with_columns(pl.Series("ADX", talib.ADX(
        data["high"].to_numpy(), data["low"].to_numpy(), data["close"].to_numpy(), timeperiod=14)))

    # 📌 Ichimoku Cloud per trend analysis
    data = data.with_columns([
        (data["high"].rolling(9).max() + data["low"].rolling(9).min()) / 2
    ].alias("Tenkan_Sen"))

    data = data.with_columns([
        (data["high"].rolling(26).max() + data["low"].rolling(26).min()) / 2
    ].alias("Kijun_Sen"))

    data = data.with_columns([
        ((data["Tenkan_Sen"] + data["Kijun_Sen"]) / 2).shift(26).alias("Senkou_Span_A"),
        ((data["high"].rolling(52).max() + data["low"].rolling(52).min()) / 2)
        .shift(26).alias("Senkou_Span_B")
    ])

    # 📌 SuperTrend per segnali di acquisto e vendita
    atr = talib.ATR(data["high"].to_numpy(), data["low"].to_numpy(),
                     data["close"].to_numpy(), timeperiod=14)
    data = data.with_columns([
        pl.Series("SuperTrend_Upper", data["close"] + (2 * atr)),
        pl.Series("SuperTrend_Lower", data["close"] - (2 * atr))
    ])

    # 📌 Sentiment Analysis da news e social media
    data = data.with_columns(pl.Series("Sentiment_Score", fetch_sentiment_data()))

    return data


def fetch_sentiment_data():
    """Recupera i dati di sentiment analysis dalle news e dai social media."""
    try:
        response = requests.get(SENTIMENT_API_URL)
        if response.status_code == 200:
            sentiment_data = response.json()
            return sentiment_data['sentiment_score']
        else:
            logging.error("❌ Errore nel recupero del sentiment.")
            return np.nan
    except Exception as e:
        logging.error(f"❌ Errore API Sentiment Analysis: {e}")
        return np.nan


def get_indicators_list():
    """Restituisce una lista di tutti gli indicatori disponibili."""
    return [
        'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'EMA_50', 'EMA_200', 'SMA_100', 'ADX', 'Tenkan_Sen', 'Kijun_Sen', 'Senkou_Span_A',
        'Senkou_Span_B', 'SuperTrend_Upper', 'SuperTrend_Lower', 'Sentiment_Score'
    ]


class TradingIndicators:
    """Classe per calcolare e gestire gli indicatori di trading."""

    def __init__(self, data):
        self.data = data
        self.indicators = {}

    def calculate_all_indicators(self):
        """Calcola tutti gli indicatori disponibili e li assegna ai dati."""
        self.indicators = calculate_indicators(self.data)

    def fetch_sentiment(self):
        """Ottiene dati di sentiment analysis per il mercato."""
        return fetch_sentiment_data()

    def list_available_indicators(self):
        """Restituisce la lista di tutti gli indicatori disponibili."""
        return get_indicators_list()
