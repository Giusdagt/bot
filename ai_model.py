from demo_module import demo_trade
from backtest_module import run_backtest
from strategy_generator import StrategyGenerator
from price_prediction import PricePredictionModel
from optimizer_core import OptimizerCore
import logging
from datetime import datetime
from pathlib import Path
import polars as pl
import numpy as np
import asyncio
import MetaTrader5 as mt5
from data_handler import get_normalized_market_data
from drl_agent import DRLAgent  # Deep Reinforcement Learning
from risk_management import RiskManagement, VolatilityPredictor
from portfolio_optimization import PortfolioOptimizer

# Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Percorsi dei modelli e dei dati
MODEL_DIR = (
    Path("/mnt/usb_trading_data/models")
    if Path("/mnt/usb_trading_data").exists()
    else Path("D:/trading_data/models")
)
DATA_FILE = MODEL_DIR / "ai_memory.parquet"
DB_FILE = MODEL_DIR / "trades.db"
PERFORMANCE_FILE = MODEL_DIR / "performance.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Creazione database per il salvataggio delle operazioni
TRADE_FILE = MODEL_DIR / "trades.parquet"

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

class AIModel:
    def __init__(self, market_data, balances):
        self.volatility_predictor = VolatilityPredictor()
        self.risk_manager = {acc: RiskManagement() for acc in balances}
        self.memory = self.load_memory()
        self.strategy_strength = np.mean(self.memory) + 1  # 🔥 Valore infinito per migliorare strategia
        self.balances = balances
        self.portfolio_optimizer = PortfolioOptimizer(market_data, balances, True)
        self.price_predictor = PricePredictionModel()
        self.drl_agent = DRLAgent()
        self.active_assets = self.select_best_assets(market_data)  # Selezione automatica degli asset migliori
        self.strategy_generator = StrategyGenerator()
        selected_strategy, strategy_weight = self.strategy_generator.select_best_strategy(market_data) # generatore di strategie

    def load_memory(self):
        if DATA_FILE.exists():
            logging.info("📥 Caricamento memoria compressa...")
            loaded_memory = pl.read_parquet(DATA_FILE)["memory"].to_numpy()
            return np.mean(loaded_memory, axis=0, keepdims=True)
        return np.zeros(1, dtype=np.float32)

    def save_memory(self, new_value):
        df = pl.DataFrame({"memory": [new_value]})
        df.write_parquet(DATA_FILE, compression="zstd", mode="overwrite")
        logging.info("💾 Memoria IA aggiornata.")

    def update_performance(self, account, symbol, action, lot_size, profit, strategy):
        # Carica i dati esistenti
        if TRADE_FILE.exists():
            df = pl.read_parquet(TRADE_FILE)
        else:
            df = pl.DataFrame({"account": [], "symbol": [], "action": [],
                               "lot_size": [], "profit": [], "strategy": []})

        # Cerca se esiste già un trade per questo account e simbolo
        existing_trade = df.filter((df["account"] == account) & (df["symbol"] == symbol))

        if len(existing_trade) > 0:
            # Aggiorna il valore invece di creare una nuova riga
            df = df.with_columns([
                pl.when((df["account"] == account) & (df["symbol"] == symbol))
                .then(pl.lit(profit)).otherwise(df["profit"]).alias("profit")
            ])
        else:
            # Se non esiste, aggiunge una nuova entry
            new_entry = pl.DataFrame({"account": [account], "symbol": [symbol],
                                      "action": [action],"lot_size": [lot_size],
                                      "profit": [profit], "strategy": [strategy]})
            df = pl.concat([df, new_entry])

        df.write_parquet(TRADE_FILE, compression="zstd", mode="overwrite")
        logging.info(f"📊 Trade aggiornato per {account} su {symbol}: Profit {profit} | Strategia: {strategy}")

    def adapt_lot_size(self, balance, success_probability):
        max_lot_size = balance / 50  # 🔥 Adatta il lot size in base al saldo disponibile
        return min(balance * (success_probability * self.strategy_strength) / 10, max_lot_size)

    def execute_trade(self, account, symbol, action, lot_size, risk, strategy):
        order = {
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_BUY if action == "buy" else mt5.ORDER_SELL,
            "price": mt5.symbol_info_tick(symbol).ask,
            "deviation": 10,
            "magic": 0,
            "comment": f"AI Trade ({strategy})",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }
        result = mt5.order_send(order)
        status = "executed" if result.retcode == mt5.TRADE_RETCODE_DONE else "failed"
        self.update_performance(account, symbol, action, lot_size,
                                result.profit if status == "executed" else 0, strategy)
        self.save_memory(self.strategy_strength)  # 🔥 Aggiornamento della memoria dopo ogni trade
        self.strategy_generator.update_strategies(strategy, result.profit if status == "executed" else -10)
        logging.info(f"✅ Trade {status} per {account} su {symbol}: {action} {lot_size} lotto | Strategia: {strategy}")

    def select_best_assets(self, market_data):
        """Seleziona automaticamente gli asset con il miglior rendimento storico."""
        assets_performance = {asset: market_data[asset]["close"].pct_change().mean() for asset in market_data.keys()}
        sorted_assets = sorted(assets_performance, key=assets_performance.get, reverse=True)
        logging.info(f"📈 Asset selezionati per il trading: {sorted_assets[:5]}")
        return sorted_assets[:5]  # Seleziona i 5 asset migliori

    async def decide_trade(self, symbol):
        market_data = get_normalized_market_data(symbol)

        if market_data is None or market_data.height == 0:
            logging.warning(f"⚠️ Nessun dato per {symbol}. Avvio Backtest per auto-miglioramento.")
            self.backtest(symbol, [market_data])
            return False

        predicted_price = self.price_predictor.predict_price()

        for account in self.balances:
            success_probability = self.drl_agent.predict(symbol, market_data)
            lot_size = self.adapt_lot_size(self.balances[account], success_probability)
            action = "buy" if predicted_price > market_data["close"].iloc[-1] else "sell"

            # 🔥 Selezione della strategia migliore
            trade_profit = predicted_price - market_data["close"].iloc[-1]
            strategy, _ = self.strategy_generator.select_best_strategy(market_data)
            self.strategy_generator.update_knowledge(
                profit=trade_profit,
                win_rate=1 if trade_profit > 0 else 0,
                drawdown=abs(min(0, trade_profit)),
                volatility=market_data["volatility"].iloc[-1]
            )

            if success_probability > 0.5:
                self.execute_trade(account, symbol, action, lot_size, success_probability, strategy)
            else:
                logging.info(f"🚫 Nessun trade su {symbol} per {account}. Avvio Demo Trade per miglioramento.")
                self.demo_trade(symbol, market_data)

    def background_optimization_loop(ai_model_instance, interval_seconds=43200):  # ogni 12 ore
        optimizer = (
            OptimizerCore(
                strategy_generator=ai_model_instance.strategy_generator,
                ai_model=ai_model_instance
            )
        )
        while True:
            optimizer.run_full_optimization()
            time.sleep(interval_seconds)
            
if __name__ == "__main__":
    ai_model = AIModel(get_normalized_market_data(), fetch_account_balances())
    import threading
    thread = threading.Thread(
        target=background_optimization_loop,
        args=(ai_model,), daemon=True
    )
    thread.start()

    # 🔥 Loop di miglioramento continuo
    while True:
        for asset in ai_model.active_assets:
            asyncio.run(ai_model.decide_trade(asset))
        asyncio.sleep(10)
