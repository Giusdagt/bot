#indicators
import numpy as np
import pandas as pd
import logging
import requests
import talib

# 📌 Configurazione del logging avanzato
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 📌 URL per l'analisi del sentiment
SENTIMENT_API_URL = "https://your-sentiment-api.com/analyze"

def calculate_indicators(data):
    """Calcola tutti gli indicatori tecnici principali e li aggiunge ai dati di mercato."""
    
    # 📌 RSI (Relative Strength Index) per trend reversal e scalping
    data['RSI'] = talib.RSI(data['close'], timeperiod=14)

    # 📌 Bollinger Bands per volatilità
    upper_band, middle_band, lower_band = talib.BBANDS(data['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    data['BB_Upper'] = upper_band
    data['BB_Middle'] = middle_band
    data['BB_Lower'] = lower_band

    # 📌 MACD per trend detection e momentum
    macd, macdsignal, macdhist = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['MACD'] = macd
    data['MACD_Signal'] = macdsignal
    data['MACD_Hist'] = macdhist

    # 📌 EMA e SMA per trend-following
    data['EMA_50'] = talib.EMA(data['close'], timeperiod=50)
    data['EMA_200'] = talib.EMA(data['close'], timeperiod=200)
    data['SMA_100'] = talib.SMA(data['close'], timeperiod=100)

    # 📌 ADX per forza del trend
    data['ADX'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)

    # 📌 Ichimoku Cloud per trend analysis
    nine_high = data['high'].rolling(window=9).max()
    nine_low = data['low'].rolling(window=9).min()
    data['Tenkan_Sen'] = (nine_high + nine_low) / 2

    twenty_six_high = data['high'].rolling(window=26).max()
    twenty_six_low = data['low'].rolling(window=26).min()
    data['Kijun_Sen'] = (twenty_six_high + twenty_six_low) / 2

    fifty_two_high = data['high'].rolling(window=52).max()
    fifty_two_low = data['low'].rolling(window=52).min()
    data['Senkou_Span_A'] = ((data['Tenkan_Sen'] + data['Kijun_Sen']) / 2).shift(26)
    data['Senkou_Span_B'] = ((fifty_two_high + fifty_two_low) / 2).shift(26)

    # 📌 SuperTrend per segnali di acquisto e vendita
    atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
    data['SuperTrend_Upper'] = data['close'] + (2 * atr)
    data['SuperTrend_Lower'] = data['close'] - (2 * atr)

    # 📌 Sentiment Analysis da news e social media
    data['Sentiment_Score'] = fetch_sentiment_data()

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