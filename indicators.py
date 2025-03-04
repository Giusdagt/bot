"""
Modulo per il calcolo avanzato degli indicatori di trading.
"""

import logging
import polars as pl
import requests
import talib

# 📌 Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 📌 URL API per Sentiment Analysis
SENTIMENT_API_URL = "https://your-sentiment-api.com/analyze"


def calculate_scalping_indicators(data):
    """Indicatori ottimizzati per Scalping."""
    data = data.with_columns(
        pl.Series("RSI", talib.RSI(data["close"], timeperiod=14)),
        pl.Series("STOCH_K", talib.STOCH(data["high"], data["low"], data["close"])[0]),
        pl.Series("STOCH_D", talib.STOCH(data["high"], data["low"], data["close"])[1]),
    )

    macd, macdsignal, macdhist = talib.MACD(
        data["close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    atr = talib.ATR(data["high"], data["low"], data["close"], timeperiod=14)

    data = data.with_columns(
        pl.Series("MACD", macd),
        pl.Series("MACD_Signal", macdsignal),
        pl.Series("MACD_Hist", macdhist),
        pl.Series("SuperTrend_Upper", data["close"] + (2 * atr)),
        pl.Series("SuperTrend_Lower", data["close"] - (2 * atr)),
    )

    return data


def calculate_historical_indicators(data):
    """Indicatori ottimizzati per analisi storiche."""
    upper_band, middle_band, lower_band = talib.BBANDS(
        data["close"], timeperiod=20, nbdevup=2, nbdevdn=2
    )
    adx = talib.ADX(data["high"], data["low"], data["close"], timeperiod=14)
    obv = talib.OBV(data["close"], data["volume"])

    data = data.with_columns(
        pl.Series("BB_Upper", upper_band),
        pl.Series("BB_Middle", middle_band),
        pl.Series("BB_Lower", lower_band),
        pl.Series("ADX", adx),
        pl.Series("OBV", obv),
    )

    # Ichimoku Cloud
    nine_high = data["high"].rolling(window=9).max()
    nine_low = data["low"].rolling(window=9).min()
    twenty_six_high = data["high"].rolling(window=26).max()
    twenty_six_low = data["low"].rolling(window=26).min()
    fifty_two_high = data["high"].rolling(window=52).max()
    fifty_two_low = data["low"].rolling(window=52).min()

    data = data.with_columns(
        pl.Series("Ichimoku_Tenkan", (nine_high + nine_low) / 2),
        pl.Series("Ichimoku_Kijun", (twenty_six_high + twenty_six_low) / 2),
        pl.Series("Senkou_Span_A", ((data["Ichimoku_Tenkan"] + data["Ichimoku_Kijun"]) / 2).shift(26)),
        pl.Series("Senkou_Span_B", ((fifty_two_high + fifty_two_low) / 2).shift(26)),
    )

    return data


def calculate_intraday_indicators(data):
    """Indicatori ottimizzati per il trading intra-day."""
    data = calculate_scalping_indicators(data)
    data = calculate_historical_indicators(data)
    return data


def fetch_sentiment_data():
    """Recupera dati di sentiment analysis."""
    try:
        response = requests.get(SENTIMENT_API_URL)
        response.raise_for_status()
        sentiment_data = response.json()
        return sentiment_data["sentiment_score"]
    except requests.exceptions.RequestException as e:
        logging.error("❌ Errore API Sentiment Analysis: %s", e)
        return None


def get_indicators_list():
    """Restituisce una lista di tutti gli indicatori disponibili."""
    return [
        "RSI", "STOCH_K", "STOCH_D", "BB_Upper", "BB_Middle", "BB_Lower",
        "MACD", "MACD_Signal", "MACD_Hist", "EMA_50", "EMA_200", "SMA_100",
        "ADX", "OBV", "Ichimoku_Tenkan", "Ichimoku_Kijun", "Senkou_Span_A",
        "Senkou_Span_B", "SuperTrend_Upper", "SuperTrend_Lower"
    ]


class TradingIndicators:
    """Classe per calcolare e gestire gli indicatori di trading."""

    def __init__(self, data):
        self.data = data
        self.indicators = {}

    def calculate_all_indicators(self):
        """Calcola tutti gli indicatori disponibili."""
        self.indicators = calculate_intraday_indicators(self.data)

    def fetch_sentiment(self):
        """Ottiene dati di sentiment analysis per il mercato."""
        return fetch_sentiment_data()

    def list_available_indicators(self):
        """Restituisce la lista di tutti gli indicatori disponibili."""
        return get_indicators_list()
