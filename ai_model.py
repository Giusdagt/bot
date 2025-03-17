import logging
from datetime import datetime
from pathlib import Path
import polars as pl
import numpy as np
import asyncio
import MetaTrader5 as mt5
import sqlite3
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from indicators import (
    calculate_scalping_indicators,
    calculate_swing_indicators,
    calculate_sentiment_indicator
)
from data_handler import (
    get_normalized_market_data, process_historical_data, fetch_mt5_data
)
from drl_agent import DRLAgent  # 🔥 Deep Reinforcement Learning
from gym_trading_env import TradingEnv
from risk_management import RiskManagement, VolatilityPredictor
from portfolio_optimization import PortfolioOptimizer

# Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(asctime)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Percorsi per il salvataggio dei modelli e dei dati
MODEL_DIR = Path("/mnt/usb_trading_data/models") if Path("/mnt/usb_trading_data").exists() else Path("D:/trading_data/models")
DATA_FILE = MODEL_DIR / "ai_memory.parquet"
DB_FILE = MODEL_DIR / "trades.db"
PERFORMANCE_FILE = MODEL_DIR / "performance.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Creazione del database SQLite per il salvataggio delle operazioni
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    account TEXT,
                    symbol TEXT,
                    action TEXT,
                    lot_size REAL,
                    risk REAL,
                    status TEXT
                )''')
    conn.commit()
    conn.close()
init_db()

# Connessione sicura a MetaTrader 5
def initialize_mt5():
    for _ in range(3):
        if mt5.initialize():
            logging.info("✅ Connessione a MetaTrader 5 stabilita con successo.")
            return True
        logging.warning("⚠️ Tentativo di connessione a MT5 fallito, riprovo...")
    return False

# Recupero saldo da MetaTrader 5
def get_metatrader_balance():
    if not initialize_mt5():
        return 0
    account_info = mt5.account_info()
    return account_info.balance if account_info else 0

# Recupera automaticamente il saldo per ogni utente
def fetch_account_balances():
    return {
        "Danny": get_metatrader_balance(),
        "Giuseppe": get_metatrader_balance()
    }

def get_market_condition():
    return "normal"

class AIModel:
    def __init__(self, market_data, balances, market_condition):
        self.volatility_predictor = VolatilityPredictor()
        self.risk_manager = {acc: RiskManagement() for acc in balances}
        self.memory = self.load_memory()
        self.strategy_representation = np.array([0], dtype=np.float32)
        self.model_representation = np.array([1], dtype=np.float32)
        self.optimization_representation = np.array([2], dtype=np.float32)
        self.balances = balances
        self.indicator_set = self.select_best_indicators(market_condition)
        self.portfolio_optimizer = PortfolioOptimizer(market_data, balances, market_condition == "scalping")
        self.drl_agent = DRLAgent()

    def load_memory(self):
        if DATA_FILE.exists():
            logging.info("📥 Caricamento memoria IA da Parquet...")
            loaded_memory = pl.read_parquet(DATA_FILE)
            self.strategy_representation = loaded_memory["strategy"][0]
            self.model_representation = loaded_memory["model"][0]
            self.optimization_representation = loaded_memory["optimization"][0]

    def save_memory(self):
        memory_data = {
            "strategy": [self.strategy_representation],
            "model": [self.model_representation],
            "optimization": [self.optimization_representation]
        }
        pl.DataFrame(memory_data).write_parquet(DATA_FILE, compression="zstd")
        logging.info("💾 Memoria IA aggiornata e ottimizzata con Zstd.")

    def update_performance(self, account, profit):
        performance_data = {"timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                            "account": [account], "profit": [profit]}
        df = pl.DataFrame(performance_data)
        df.write_parquet(PERFORMANCE_FILE, compression="zstd", append=True)
        logging.info(f"📊 Performance aggiornata per {account}: Profit {profit}")

    def adapt_lot_size(self, balance, success_probability):
        base_lot = 0.02
        return min(balance * success_probability / 10, 1.0) if success_probability > 0.9 else base_lot * (balance / 100)

    def execute_trade(self, account, symbol, action, lot_size, risk):
        order = {
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_BUY if action == "buy" else mt5.ORDER_SELL,
            "price": mt5.symbol_info_tick(symbol).ask,
            "deviation": 10,
            "magic": 0,
            "comment": "AI Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }
        result = mt5.order_send(order)
        status = "executed" if result.retcode == mt5.TRADE_RETCODE_DONE else "failed"
        self.update_performance(account, result.profit if status == "executed" else 0)
        logging.info(f"✅ Trade {status} per {account} su {symbol}: {action} {lot_size} lotto")

    def backtest(self, symbol, historical_data):
        """Esegue un backtest utilizzando dati storici per migliorare il modello."""
        logging.info(f"🔄 Inizio del backtest su {symbol}...")
        for data in historical_data:
            success_probability = self.drl_agent.predict(symbol, data)
            lot_size = self.adapt_lot_size(self.balances["Danny"], success_probability)
            action = "buy" if success_probability > 0.5 else "sell"
            self.execute_trade("Backtest", symbol, action, lot_size, success_probability)
        logging.info(f"✅ Backtest completato su {symbol}.")

    def demo_trade(self, symbol, market_data):
        """Esegue una simulazione di trading per testare strategie senza rischiare fondi reali."""
        logging.info(f"🔄 Inizio della demo su {symbol}...")
        success_probability = self.drl_agent.predict(symbol, market_data)
        lot_size = self.adapt_lot_size(self.balances["Danny"], success_probability)
        action = "buy" if success_probability > 0.5 else "sell"
        logging.info(f"🧪 Demo trade per {symbol}: {action} {lot_size} lotto (Probabilità di successo: {success_probability:.2f})")
        logging.info(f"✅ Demo trade completato su {symbol}.")

    async def decide_trade(self, symbol):
        market_data = get_best_market_data(symbol)
        if market_data is None or market_data.height == 0:
            logging.warning(f"⚠️ Nessun dato per {symbol}. Nessuna decisione di trading.")
            return False
        for account in self.balances:
            success_probability = self.drl_agent.predict(symbol, market_data)
            lot_size = self.adapt_lot_size(self.balances[account], success_probability)
            action = "buy" if success_probability > 0.5 else "sell"
            self.execute_trade(account, symbol, action, lot_size, success_probability)

if __name__ == "__main__":
    ai_model = AIModel(get_normalized_market_data(), fetch_account_balances(), get_market_condition())
    asyncio.run(ai_model.decide_trade("EURUSD"))
    historical_data = [get_normalized_market_data()]
    ai_model.backtest("EURUSD", historical_data)
    ai_model.demo_trade("EURUSD", get_normalized_market_data())
