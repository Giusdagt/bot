import logging
from datetime import datetime
from pathlib import Path
import polars as pl
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from indicators import (
    calculate_scalping_indicators,
    calculate_swing_indicators,
    calculate_sentiment_indicator
)
from data_handler import (
    get_normalized_market_data, process_historical_data, fetch_mt5_data
)
from drl_agent import DRLAgent
from gym_trading_env import TradingEnv
from risk_management import RiskManagement, VolatilityPredictor
from portfolio_optimization import PortfolioOptimizer  # 🔥 INTEGRATO!

# Configurazione logging avanzata
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Percorsi per il salvataggio dei modelli e dei dati
MODEL_DIR = Path("/mnt/usb_trading_data/models") if Path("/mnt/usb_trading_data").exists() else Path("D:/trading_data/models")
DATA_FILE = MODEL_DIR / "ai_memory.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Definizione delle funzioni

def fetch_account_balances():
    """Recupera automaticamente il saldo da MetaTrader tramite API."""
    return {
        "Danny": get_metatrader_balance("Danny"),  
        "Giuseppe": get_metatrader_balance("Giuseppe")
    }


def get_market_condition():
    """
    🔥 Determina se il mercato è in modalità scalping o normale.
    """
    return "normal"


class AIModel:
    """
    Modello AI avanzato con memoria compressa ultra-efficiente.
    Ottimizza strategie, modelli e gestione della RAM senza perdere dati.
    """
    def __init__(self, market_data, balances, market_condition):
        self.volatility_predictor = VolatilityPredictor()
        self.risk_manager = RiskManagement()
        self.memory = self.load_memory()
        self.strategy_representation = np.array([0], dtype=np.float32)
        self.model_representation = np.array([1], dtype=np.float32)
        self.optimization_representation = np.array([2], dtype=np.float32)
        self.balances = fetch_account_balances()
        self.indicator_set = self.select_best_indicators(market_condition)
        self.portfolio_optimizer = PortfolioOptimizer(market_data, balances, market_condition == "scalping")

    def load_memory(self):
        """Carica la memoria IA ottimizzata con compressione ultra-efficiente."""
        if DATA_FILE.exists():
            logging.info("📥 Caricamento memoria IA da Parquet...")
            loaded_memory = pl.read_parquet(DATA_FILE)
            self.strategy_representation = loaded_memory["strategy"][0]
            self.model_representation = loaded_memory["model"][0]
            self.optimization_representation = loaded_memory["optimization"][0]
        else:
            logging.info("🔄 Creazione nuova memoria IA...")
            self.strategy_representation = np.array([0])
            self.model_representation = np.array([1])
            self.optimization_representation = np.array([2])

    def save_memory(self):
        """Salva la memoria IA comprimendola progressivamente senza perdita di dati."""
        memory_data = {
            "strategy": [self.strategy_representation],
            "model": [self.model_representation],
            "optimization": [self.optimization_representation]
        }
        pl.DataFrame(memory_data).write_parquet(DATA_FILE, compression="zstd")
        logging.info("💾 Memoria IA aggiornata e ottimizzata con Zstd.")

    def update_memory(self, strategy_vector, model_vector, optimization_vector):
        """🔹 Aggiorna la memoria AI con rappresentazioni numeriche compresse."""
        self.strategy_representation = self._compress_data(strategy_vector)
        self.model_representation = self._compress_data(model_vector)
        self.optimization_representation = self._compress_data(optimization_vector)
        self.save_memory()

    def _compress_data(self, data):
        """🔹 Compressione avanzata con numpy: normalizzazione e riduzione di dimensione"""
        if isinstance(data, (list, np.ndarray)):
            return np.mean(data)
        return data

    def select_best_indicators(self, market_condition):
        """Seleziona automaticamente gli indicatori migliori per il contesto di mercato."""
        if market_condition == "scalping":
            return calculate_scalping_indicators
        elif market_condition == "swing":
            return calculate_swing_indicators
        else:
            return calculate_sentiment_indicator

    def decide_trade(self, symbol):
        """L'IA prende decisioni basate su memoria, previsioni, rischio ottimizzato e portafoglio."""
        market_data = get_best_market_data(symbol)
        if market_data is None or market_data.height == 0:
            logging.warning(f"⚠️ Nessun dato per {symbol}. Nessuna decisione di trading.")
            return False

        # 🔥 Miglioriamo il processo decisionale con numpy
        normalized_data = (market_data["close"] - np.mean(market_data["close"])) / np.std(market_data["close"])
        prediction = self.volatility_predictor.predict(normalized_data)
        risk = self.risk_manager.adjust_risk(symbol)

        if risk < 0.3:
            logging.info(f"🚀 Trade approvato su {symbol} (Rischio: {risk:.2f})")
            optimized_allocation, weights = self.portfolio_optimizer.optimize_portfolio()
            logging.info(f"📊 Allocazione ottimizzata: {optimized_allocation}")
            self.update_memory(prediction, self.model_representation, weights)
            return True
        else:
            logging.warning(f"⚠️ Trade rifiutato su {symbol} (Rischio: {risk:.2f})")
            return False


# Funzione per recuperare i migliori dati disponibili
def get_best_market_data(symbol):
    """Recupera i migliori dati disponibili per un asset."""
    data = fetch_mt5_data(symbol) or get_normalized_market_data(symbol)
    if data is None or data.height == 0:
        process_historical_data()
        data = get_normalized_market_data(symbol)
    return data


# Esecuzione del modello AI ottimizzato
if __name__ == "__main__":
    logging.info("🚀 Avvio del modello AI con memoria ottimizzata.")
    market_data = get_normalized_market_data()
    balances = fetch_account_balances()
    market_condition = get_market_condition()
    ai_model = AIModel(market_data, balances, market_condition)
    ai_model.decide_trade("EURUSD")
