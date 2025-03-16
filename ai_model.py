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

# Configurazione logging avanzata
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Percorsi per il salvataggio dei modelli e dei dati
MODEL_DIR = Path("/mnt/usb_trading_data/models") if Path("/mnt/usb_trading_data").exists() else Path("D:/trading_data/models")
DATA_FILE = MODEL_DIR / "ai_memory.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Definizione classe AIModel
class AIModel:
    def __init__(self):
        self.volatility_predictor = VolatilityPredictor()
        self.risk_manager = RiskManagement()
        self.memory = self.load_memory()

    def load_memory(self):
        """Carica la memoria dell'IA dal file Parquet con compressione Zstd."""
        if DATA_FILE.exists():
            logging.info("📥 Caricamento memoria IA da Parquet...")
            return pl.read_parquet(DATA_FILE)
        logging.info("🔄 Creazione nuova memoria IA...")
        return pl.DataFrame({"strategy": [], "performance": [], "adjustments": []})

    def save_memory(self):
        """Salva la memoria IA comprimendola con Parquet+Zstd."""
        self.memory.write_parquet(DATA_FILE, compression="zstd")
        logging.info("💾 Memoria IA aggiornata e compressa con Zstd.")

    def update_memory(self, strategy, performance, adjustments):
        """Aggiorna la memoria dell'IA migliorando le strategie esistenti."""
        new_data = pl.DataFrame({"strategy": [strategy], "performance": [performance], "adjustments": [adjustments]})
        self.memory = self.memory.vstack(new_data).unique(subset=["strategy"])  # Evita duplicati
        self.save_memory()

    def decide_trade(self, symbol):
        """L'IA prende una decisione basata su memoria, previsioni e rischio."""
        market_data = get_best_market_data(symbol)
        if market_data is None or market_data.is_empty():
            logging.warning(f"⚠️ Nessun dato per {symbol}. Nessuna decisione di trading.")
            return False

        prediction = self.volatility_predictor.predict(market_data)
        risk = self.risk_manager.adjust_risk(symbol)

        if risk < 0.3:
            logging.info(f"🚀 Trade approvato su {symbol} (Rischio: {risk:.2f})")
            self.update_memory(symbol, prediction, risk)
            return True
        else:
            logging.warning(f"⚠️ Trade rifiutato su {symbol} (Rischio: {risk:.2f})")
            return False

# Funzione per recuperare i migliori dati disponibili
def get_best_market_data(symbol):
    """Recupera i migliori dati disponibili per un asset."""
    data = fetch_mt5_data(symbol) or get_normalized_market_data(symbol)
    if data is None or data.is_empty():
        process_historical_data()
        data = get_normalized_market_data(symbol)
    return data

# Esecuzione della previsione AI
if __name__ == "__main__":
    logging.info("🚀 Avvio del modello AI con memoria compressa.")
    ai_model = AIModel()
    ai_model.decide_trade("EURUSD")
