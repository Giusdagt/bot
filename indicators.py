"""
Module for calculating various trading indicators.
"""

import logging
import numpy as np
import polars as pl
import requests
import talib

# 📌 Configurazione del logging avanzato
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 📌 URL per l'analisi del sentiment
SENTIMENT_API_URL = "https://your-sentiment-api.com/analyze"


def calculate_indicators(data):
    """Calcola tutti gli indicatori tecnici principali e li
    aggiunge ai dati di mercato.
    """

    # 📌 RSI (Relative Strength Index) per trend reversal e scalping
    data = data.with_columns(
        pl.Series("RSI", talib.RSI(data["close"], timeperiod=14))
    )

    # 📌 Bollinger Bands per volatilità
    upper_band, middle_band, lower_band = talib.BBANDS(
        data["close"], timeperiod=20, nbdevup=2, nbdevdn=2
    )
    data = data.with_columns(
        [
            pl.Series("BB_Upper", upper_band),
            pl.Series("BB_Middle", middle_band),
            pl.Series("BB_Lower", lower_band),
        ]
    )

    # 📌 MACD per trend detection e momentum
    macd, macdsignal, macdhist = talib.MACD(
        data["close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    data = data.with_columns(
        [
            pl.Series("MACD", macd),
            pl.Series("MACD_Signal", macdsignal),
            pl.Series("MACD_Hist", macdhist),
        ]
    )

    # 📌 EMA e SMA per trend-following
    data = data.with_columns(
        [
            pl.Series("EMA_50", talib.EMA(data["close"], timeperiod=50)),
            pl.Series("EMA_200", talib.EMA(data["close"], timeperiod=200)),
            pl.Series("SMA_100", talib.SMA(data["close"], timeperiod=100)),
        ]
    )

    # ADX per forza del trend
    data = data.with_columns(
        pl.Series(
            "ADX",
            talib.ADX(
                data["high"],
                data["low"],
                data["close"],
                timeperiod=14
            )
        )
    )

    # 📌 Ichimoku Cloud per trend analysis
    nine_high = data["high"].rolling(window=9).max()
    nine_low = data["low"].rolling(window=9).min()
    data = data.with_columns(
        pl.Series("Tenkan_Sen", (nine_high + nine_low) / 2)
    )

    twenty_six_high = data["high"].rolling(window=26).max()
    twenty_six_low = data["low"].rolling(window=26).min()
    data = data.with_columns(
        pl.Series("Kijun_Sen", (twenty_six_high + twenty_six_low) / 2)
    )

    fifty_two_high = data["high"].rolling(window=52).max()
    fifty_two_low = data["low"].rolling(window=52).min()
    data = data.with_columns(
        [
            pl.Series(
                "Senkou_Span_A",
                ((data["Tenkan_Sen"] + data["Kijun_Sen"]) / 2).shift(26)
            ),
            pl.Series(
                "Senkou_Span_B",
                ((fifty_two_high + fifty_two_low) / 2).shift(26)
            ),
        ]
    )

    # 📌 SuperTrend per segnali di acquisto e vendita
    atr = talib.ATR(data["high"], data["low"], data["close"], timeperiod=14)
    data = data.with_columns(
        [
            pl.Series("SuperTrend_Upper", data["close"] + (2 * atr)),
            pl.Series("SuperTrend_Lower", data["close"] - (2 * atr)),
        ]
    )

    # 📌 Sentiment Analysis da news e social media
    data = data.with_columns(
        pl.Series("Sentiment_Score", fetch_sentiment_data())
    )

    return data


def fetch_sentiment_data():
    """Recupera i dati di sentiment analysis dalle news e dai social media."""
    try:
        response = requests.get(SENTIMENT_API_URL)
        response.raise_for_status()
        sentiment_data = response.json()
        return sentiment_data["sentiment_score"]
    except requests.exceptions.HTTPError as http_err:
        logging.error("❌ HTTP error occurred: %s", http_err)
    except requests.exceptions.ConnectionError as conn_err:
        logging.error("❌ Connection error occurred: %s", conn_err)
    except requests.exceptions.Timeout as timeout_err:
        logging.error("❌ Timeout error occurred: %s", timeout_err)
    except requests.exceptions.RequestException as req_err:
        logging.error("❌ Error occurred: %s", req_err)
    return np.nan


def get_indicators_list():
    """Restituisce una lista di tutti gli indicatori disponibili."""
    return [
        "RSI", "BB_Upper", "BB_Middle", "BB_Lower", "MACD", "MACD_Signal",
        "MACD_Hist", "EMA_50", "EMA_200", "SMA_100", "ADX", "Tenkan_Sen",
        "Kijun_Sen", "Senkou_Span_A", "Senkou_Span_B", "SuperTrend_Upper",
        "SuperTrend_Lower", "Sentiment_Score"
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
