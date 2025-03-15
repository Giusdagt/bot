import os
import logging
from datetime import datetime
from pathlib import Path
import polars as pl
import numpy as np
import xgboost as xgb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from indicators import TradingIndicators
from data_handler import (
    get_normalized_market_data, process_historical_data, fetch_mt5_data
)
from drl_agent import DRLAgent
from gym_trading_env import TradingEnv
from risk_management import RiskManagement, VolatilityPredictor
from portfolio_optimization import PortfolioOptimizer

# 📌 Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 📌 Percorsi per il salvataggio dei modelli
MODEL_DIR = Path("/mnt/usb_trading_data/models") if Path(
    "/mnt/usb_trading_data"
).exists() else Path("D:/trading_data/models")
CLOUD_MODEL_DIR = Path("/mnt/google_drive/trading_models")
MODEL_FILE = MODEL_DIR / "trading_model.h5"
XGB_MODEL_FILE = MODEL_DIR / "xgb_trading_model.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
CLOUD_MODEL_DIR.mkdir(parents=True, exist_ok=True)


class AIModel:
    """
    Classe AI per il trading automatico. Utilizza modelli di deep learning e
    machine learning per analizzare i mercati e decidere le operazioni.
    """

    def __init__(self):
        self.volatility_predictor = VolatilityPredictor()
        self.risk_manager = RiskManagement()

    def decide_trade(self, symbol: str) -> bool:
        """
        Decide se eseguire un'operazione di trading basandosi sul rischio e
        sulla previsione di volatilità.
        Args:
            symbol (str): Simbolo dell'asset da analizzare.
        Returns:
            bool: True se il trade è consentito, False altrimenti.
        """
        market_data = get_best_market_data(symbol)
        if market_data is None or market_data.is_empty():
            logging.warning(f"⚠️ Nessun dato valido per {symbol}.")
            return False
        self.risk_manager.adjust_risk(symbol)
        risk = self.risk_manager.risk_settings["trailing_stop_pct"]

        if risk < 0.3:
            logging.info(f"AI:Esegui trade su {symbol} (Rischio: {risk:.2f})")
            return True
        else:
            logging.warning(f"⚠️ AI: Rischio alto ({risk:.2f}) per {symbol}")
            return False


def get_best_market_data(symbol: str) -> pl.DataFrame:
    """
    Recupera i migliori dati di mercato disponibili per un asset, scegliendo
    tra scalping e storico.
    Args:
        symbol (str): Simbolo dell'asset.
    Returns:
        pl.DataFrame: Dati di mercato ottimizzati.
    """
    data = fetch_mt5_data(symbol)
    if data is None or data.is_empty():
        logging.info(f"📡 Nessun dato di scalping per {symbol}. Uso storico.")
        data = get_normalized_market_data(symbol)
    if data is None or data.is_empty():
        logging.warning(f"⚠️Nessun dato valido X {symbol}, provo con storico.")
        process_historical_data()
        data = get_normalized_market_data(symbol)
    return data


def example_prediction(symbol: str):
    """
    Esegue una previsione AI adattiva con dati storici o di scalping.
    Args:
        symbol (str): Simbolo dell'asset su cui eseguire la previsione.
    """
    logging.info(f"📡 Recupero dati per {symbol}")
    data = get_best_market_data(symbol)
    if data is None or data.is_empty():
        logging.error(f"❌ Nessun dato disponibile per {symbol}")
        return
    if "close" not in data.columns or data["close"].null_count() > 0:
        logging.warning("⚠️ Dati di incompleti o nulli, impossibile predire.")
        return
    scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        data["close"].to_numpy().reshape(-1, 1)
    )
    X_lstm = np.array([scaled_data[-60:]]).reshape(-1, 60, 1)
    X_xgb = np.array([scaled_data[-60:]])
    lstm_model = load_model(MODEL_FILE) if MODEL_FILE.exists() else train_lstm_model()
    xgb_model = xgb.XGBRegressor()  
    if XGB_MODEL_FILE.exists():
        xgb_model.load_model(XGB_MODEL_FILE)
    else:
        xgb_model = train_xgboost_model()
    ai_model = AIModel()    
    if not ai_model.decide_trade(symbol):
        logging.warning(f"⚠️ Nessuna operazione su {symbol} a causa del rischio elevato.")
        return
    
    lstm_predictions = lstm_model.predict(X_lstm) if lstm_model else [None]
    xgb_predictions = xgb_model.predict(X_xgb) if xgb_model else [None]
    logging.info(f"📊 Previsione LSTM: {lstm_predictions[-1]}")
    logging.info(f"📊 Previsione XGBoost: {xgb_predictions[-1]}")
    optimizer = PortfolioOptimizer(data)
    optimized_allocation = optimizer.optimize()
    logging.info(f"Portafoglio ottimizzato X {symbol}: {optimized_allocation}")


if __name__ == "__main__":
    logging.info("🚀 Avvio del modello AI per il trading automatico.")
    example_prediction("EURUSD")
