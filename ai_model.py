"""
Modulo AI Model
Questo modulo definisce una classe AIModel
per il trading algoritmico basato su
intelligenza artificiale. Include funzionalità
per la gestione del rischio,
ottimizzazione del portafoglio, previsione dei prezzi,
esecuzione di trade
e monitoraggio delle prestazioni.
Include inoltre cicli di ottimizzazione in background
e strategie per migliorare
continuamente l'efficacia del modello.
"""
import threading
import time
import asyncio
import logging
from pathlib import Path
import polars as pl
import numpy as np
import MetaTrader5 as mt5
from drl_agent import DRLAgent  # Reinforcement Learning (mio file)
from drl_super_integration import DRLSuperManager
from demo_module import demo_trade
from backtest_module import run_backtest
from strategy_generator import StrategyGenerator
from price_prediction import PricePredictionModel
from optimizer_core import OptimizerCore
from data_handler import get_normalized_market_data, get_available_assets
from risk_management import RiskManagement
from volatility_tools import VolatilityPredictor
from portfolio_optimization import PortfolioOptimizer
from smart_features import apply_all_market_structure_signals
from market_fingerprint import get_embedding_for_symbol
from position_manager import PositionManager
from pattern_brain import PatternBrain

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


def initialize_mt5():
    """
    Connessione sicura a MetaTrader 5
    """
    for _ in range(3):
        if mt5.initialize():
            logging.info(
                "✅ Connessione a MetaTrader 5 stabilita con successo."
            )
            return True
        logging.warning(
            "⚠️ Tentativo di connessione a MT5 fallito, riprovo..."
        )
    return False


def get_metatrader_balance():
    """
    Recupero saldo da MetaTrader 5
    """
    if not initialize_mt5():
        return 0
    account_info = mt5.account_info()
    return account_info.balance if account_info else 0


def fetch_account_balances():
    """
    Recupera automaticamente il saldo per ogni utente
    """
    return {
        "Danny": get_metatrader_balance(),
        "Giuseppe": get_metatrader_balance()
    }


class AIModel:
    """
    Classe che rappresenta un modello di intelligenza artificiale
    per il trading.
    Questa classe gestisce il caricamento e il salvataggio della memoria,
    l'adattamento delle dimensioni del lotto,
    l'esecuzione di operazioni di trading
    e l'aggiornamento delle performance basandosi su strategie definite.
    strategy_strength (float): La forza della strategia attuale.
    strategy_generator (object): Generatore per le strategie di trading.
    """
    def __init__(self, market_data, balances):
        self.volatility_predictor = VolatilityPredictor()
        self.risk_manager = {acc: RiskManagement() for acc in balances}
        self.memory = self.load_memory()
        self.strategy_strength = np.mean(self.memory) + 1
        self.balances = balances
        self.portfolio_optimizer = PortfolioOptimizer(
            market_data, balances, True
        )
        self.price_predictor = PricePredictionModel()
        self.drl_agent = DRLAgent()
        self.active_assets = self.select_best_assets(market_data)
        self.pattern_brain = PatternBrain()
        self.strategy_generator = StrategyGenerator()
        self.drl_super_manager = DRLSuperManager()
        self.drl_super_manager.load_all()
        self.drl_super_manager.start_auto_training()

    def load_memory(self):
        """
        Carica i dati di memoria compressi da un file Parquet,
        se esistente.
        numpy.ndarray: La memoria caricata come un array Numpy.
        Se il file di memoria non esiste,
        restituisce un array vuoto predefinito.
        """
        if DATA_FILE.exists():
            logging.info("📥 Caricamento memoria compressa...")
            loaded_memory = pl.read_parquet(DATA_FILE)["memory"].to_numpy()
            return np.mean(loaded_memory, axis=0, keepdims=True)
        return np.zeros(1, dtype=np.float32)

    def save_memory(self, new_value):
        """
        Salva un nuovo valore nella memoria compressa,
        aggiornando il file Parquet.
        new_value (numpy.ndarray):
        Il nuovo valore da aggiungere alla memoria.
        """
        df = pl.DataFrame({"memory": [new_value]})
        df.write_parquet(DATA_FILE, compression="zstd", mode="overwrite")
        logging.info("💾 Memoria IA aggiornata.")

    def update_performance(
        self, account, symbol, action,
        lot_size, profit, strategy
    ):
        """
        Aggiorna le informazioni di performance relative a
        un'operazione di trading.
        account (str):Nome dell'account per cui aggiornare i dati.
        symbol (str):Simbolo dell'asset su cui è stata eseguita l'operazione.
        action (str):Tipo di operazione eseguita ("buy" o "sell").
        lot_size (float):La dimensione del lotto dell'operazione.
        profit (float):Il profitto generato dall'operazione.
        strategy (str):Nome della strategia utilizzata per l'operazione.
        Returns: None
        """
        # Carica i dati esistenti
        if TRADE_FILE.exists():
            df = pl.read_parquet(TRADE_FILE)
        else:
            df = pl.DataFrame({
                "account": [],
                "symbol": [],
                "action": [],
                "lot_size": [],
                "profit": [],
                "strategy": []
            })

        # Cerca se esiste già un trade per questo account e simbolo
        existing_trade = df.filter(
            (df["account"] == account) & (df["symbol"] == symbol)
        )

        if len(existing_trade) > 0:
            # Aggiorna il valore invece di creare una nuova riga
            df = df.with_columns([
                pl.when((df["account"] == account) & (df["symbol"] == symbol))
                .then(pl.lit(profit)).otherwise(df["profit"]).alias("profit")
            ])
        else:
            # Se non esiste, aggiunge una nuova entry
            new_entry = pl.DataFrame({
                "account": [account],
                "symbol": [symbol],
                "action": [action],
                "lot_size": [lot_size],
                "profit": [profit],
                "strategy": [strategy]
            })
            df = pl.concat([df, new_entry])

        df.write_parquet(TRADE_FILE, compression="zstd", mode="overwrite")
        logging.info(
            "📊 Trade aggiornato per %s su %s: Profit %s | Strategia: %s",
            account, symbol, profit, strategy
        )

    def adapt_lot_size(
        self, balance, success_probability, confidence_score, risk,
        predicted_volatility=None
    ):
        """
        Calcola la dimensione del lotto in modo ultra-adattivo.
        Tiene conto di strategia, confidenza, rischio e volatilità prevista.
        confidence_score (float): Confidenza del modello (es. DRL).
        """
        max_lot_size = balance / 50
        adjusted_lot_size = balance * (
            success_probability * self.strategy_strength * confidence_score *
            risk
        ) / 100
        # Se il sistema è molto sicuro, moltiplica dinamicamente la size
        if (
            success_probability > 0.9 and
            confidence_score > 0.9 and
            self.strategy_strength > 2.0
        ):
            adjusted_lot_size *= 10  # autorizza ad aprire lotti grandi

        if predicted_volatility is not None:
            adjusted_lot_size *= (
                np.clip(1 / (1 + predicted_volatility), 0.5, 2.0)
            )

        return max(0.01, min(adjusted_lot_size, max_lot_size))

    def execute_trade(self, account, symbol, action, lot_size, risk, strategy):
        """
        Esegue un'operazione di trading su MetaTrader 5
        in base ai parametri specificati.
        Args:
        account (str): Nome dell'account per cui eseguire il trade.
        symbol (str): Simbolo dell'asset su cui operare (es. EURUSD).
        action (str): Tipo di operazione da eseguire ("buy" o "sell").
        lot_size (float): Dimensione del lotto da negoziare.
        risk (float): Livello di rischio calcolato per il trade.
        strategy (str): Nome della strategia utilizzata per il trade.
        """
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
        status = (
            "executed" if result.retcode == mt5.TRADE_RETCODE_DONE
            else "failed"
        )
        self.update_performance(
            account, symbol, action, lot_size, result.profit
            if status == "executed" else 0, strategy
        )
        self.save_memory(self.strategy_strength)
        self.strategy_generator.update_strategies(
            strategy,
            result.profit if status == "executed" else -10
        )
        logging.info(
            "✅ Trade %s per %s su %s: %s %s lotto | Strat: %s | Rischio: %s",
            status, account, symbol, action, lot_size, strategy, risk
        )

    def select_best_assets(self, market_data):
        """
        Seleziona automaticamente gli asset con il miglior rendimento storico
        """
        assets_performance = {
            asset: market_data[asset]["close"].pct_change().mean()
            for asset in market_data.keys()
        }
        sorted_assets = sorted(
            assets_performance, key=assets_performance.get, reverse=True
        )
        logging.info(
            "📈 Asset selezionati per il trading: %s",
            sorted_assets[:5]
        )
        return sorted_assets[:5]  # Seleziona i 5 asset migliori

    async def decide_trade(self, symbol):
        market_data = get_normalized_market_data(symbol)

        if market_data is None or market_data.height == 0:
            logging.warning(
                "⚠️ Nessun dato per %s. Eseguo il backtest x migliorare", symbol
            )
            run_backtest(symbol, market_data)
            return False

        market_data = apply_all_market_structure_signals(market_data)

        # Calcolo degli embedding e del signal score
        signal_score = int(market_data[-1]["weighted_signal_score"])
        embeddings = [
            get_embedding_for_symbol(symbol, tf) for tf in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        ]
        pattern_data = [
            market_data["ILQ_Zone"][-1],
            market_data["fakeout_up"][-1],
            market_data["fakeout_down"][-1],
            market_data["volatility_squeeze"][-1],
            market_data["micro_pattern_hft"][-1]
        ]
        pattern_confidence = self.pattern_brain.predict_score(pattern_data)
        market_data_array = (
            market_data.select(pl.col(pl.NUMERIC_DTYPES)).to_numpy().flatten()
        )
        full_state = np.concatenate(
            [market_data_array, [signal_score], *embeddings]
        )
        full_state = np.clip(full_state, -1, 1)

        predicted_price = self.price_predictor.predict_price(symbol, full_state)

        for account in self.balances:
            last_close = market_data["close"][-1]
            risk = self.risk_manager[account].calculate_dynamic_risk(market_data)

            action_rl, confidence_score, algo_used = (
                self.drl_super_manager.get_best_action_and_confidence(full_state)
            )
            success_probability = confidence_score * pattern_confidence

            if action_rl == 1:
                action = "buy"
            elif action_rl == 2:
                action = "sell"
            else:
                logging.info(
                    "⚠️ AI ha suggerito HOLD. Nessuna operazione per %s.", symbol
                )
                return

            if signal_score < 2:
                logging.info(
                    "⚠️ Segnale troppo debole su %s (score=%s).", symbol, signal_score
                )
                return

            predicted_volatility = (
                self.volatility_predictor.predict_volatility(
                    full_state.reshape(1, -1)
                )[0]
            )

            lot_size = self.adapt_lot_size(
                self.balances[account],
                success_probability,
                confidence_score,
                risk,
                predicted_volatility
            )

            logging.info(
                f"🤖 Azione AI: {action} | Algoritmo: {algo_used} | Confidenza: {confidence_score:.2f} | Score: {signal_score}"
            )

            # Strategia
            trade_profit = predicted_price - market_data["close"].iloc[-1]
            strategy, strategy_weight = (
                self.strategy_generator.select_best_strategy(market_data)
            )
            self.strategy_strength = np.clip(
                self.strategy_strength * (1 + (strategy_weight - 0.5)),
                0.5, 3.0
            )
            self.strategy_generator.update_knowledge(
                profit=trade_profit,
                win_rate=1 if trade_profit > 0 else 0,
                drawdown=abs(min(0, trade_profit)),
                volatility=market_data["volatility"].iloc[-1]
            )
            self.volatility_predictor.update(
                full_state.reshape(1, -1),
                market_data["volatility"].to_numpy()[-1]
            )

            if pattern_confidence < 0.3:
                return

            if success_probability > 0.5:
                self.execute_trade(
                    account, symbol, action, lot_size,
                    success_probability, strategy
                )
                self.drl_agent.update(
                    full_state, 1 if trade_profit > 0 else 0
                )
                self.drl_super_manager.update_all(
                    full_state, 1 if trade_profit > 0 else 0
                )
                self.drl_super_manager.reinforce_best_agent(full_state, 1)
            else:
                logging.info(
                    "🚫 Nessun trade su %s per %s. Avvio Demo per miglioramento.",
                    symbol, account
                    )
                demo_trade(symbol, market_data)
                self.drl_agent.update(full_state, 0)
                self.drl_super_manager.update_all(full_state, 0)


def background_optimization_loop(
    ai_model_instance, interval_seconds=43200
):
    """
    Esegue un ciclo continuo per ottimizzare le strategie e il modello AI.
    Args:ai_model_instance (AIModel):Istanza del modello AI da ottimizzare.
    interval_seconds (int, opzionale):Intervallo di tempo in secondi tra
    due cicli di ottimizzazione. Default: 43200 secondi (12 ore).
    """
    optimizer = OptimizerCore(
        strategy_generator=ai_model_instance.strategy_generator,
        ai_model=ai_model_instance
    )
    while True:
        optimizer.run_full_optimization()
        time.sleep(interval_seconds)


def loop_position_monitor(position_manager_instance):
    """
    Controlla e gestisce tutte le posizioni aperte in autonomia.
    """
    while True:
        position_manager_instance.monitor_open_positions()
        time.sleep(10)


if __name__ == "__main__":
    # 🔄 Recupera tutti gli asset disponibili (preset o dinamici)
    assets = get_available_assets()

    # 📊 Crea un dizionario con i dati normalizzati per ciascun asset
    all_market_data = {
        symbol: data
        for symbol in assets
        if (data := get_normalized_market_data(symbol)) is not None
    }

    # ⚠️ IMPORTANTE: passare all_market_data, non market_data!
    ai_model = AIModel(all_market_data, fetch_account_balances())

    thread = threading.Thread(
        target=background_optimization_loop,
        args=(ai_model,), daemon=True
    )
    thread.start()

    pm = PositionManager()

    threading.Thread(
        target=lambda: loop_position_monitor(pm), daemon=True
    ).start()

    while True:
        for asset in ai_model.active_assets:
            asyncio.run(ai_model.decide_trade(asset))
        time.sleep(10)
