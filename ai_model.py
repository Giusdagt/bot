"""
Modulo AI Model.
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
import os
import logging
from pathlib import Path
import polars as pl
import numpy as np
import MetaTrader5 as mt5
from drl_agent import DRLAgent, DESIRED_STATE_SIZE, USE_DYNAMIC_STATE_SIZE
from drl_super_integration import DRLSuperManager
from demo_module import demo_trade
from backtest_module import run_backtest
from strategy_generator import StrategyGenerator
from price_prediction import PricePredictionModel
from optimizer_core import OptimizerCore
from data_handler import get_normalized_market_data, get_available_assets, process_historical_data
from risk_management import RiskManagement
from volatility_tools import VolatilityPredictor
from portfolio_optimization import PortfolioOptimizer
from smart_features import (
    apply_all_market_structure_signals, apply_all_advanced_features
)
from market_fingerprint import get_embedding_for_symbol
from position_manager import PositionManager
from pattern_brain import PatternBrain
from constants import USE_DYNAMIC_STATE_SIZE, SEQUENCE_LENGTH, DESIRED_STATE_SIZE
from state_utils import sanitize_full_state

print("ai_model.py caricato ✅")

# Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
PROCESSED_DATA_PATH = "D:/trading_data/processed_data.zstd.parquet"
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

if not os.path.exists(PROCESSED_DATA_PATH):
    print("⚠️ File dati processati non trovato, avvio generazione dati...")
    asyncio.run(process_historical_data())

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
        self.market_data = market_data
        self.volatility_predictor = VolatilityPredictor()
        self.risk_manager = {acc: RiskManagement() for acc in balances}
        self.memory = self.load_memory()
        self.strategy_strength = np.mean(self.memory) + 1
        self.balances = balances
        self.portfolio_optimizer = PortfolioOptimizer(
            market_data, balances, True
        )
        self.price_predictor = PricePredictionModel()
        state_size = DESIRED_STATE_SIZE if not USE_DYNAMIC_STATE_SIZE else self.calculate_dynamic_state_size()
        self.drl_agent = DRLAgent(state_size=state_size)
        self.active_assets = self.select_best_assets(market_data)
        self.pattern_brain = PatternBrain()
        self.strategy_generator = StrategyGenerator()
        self.drl_super_manager = DRLSuperManager(state_size=state_size)
        self.drl_super_manager.load_all()
        self.drl_super_manager.start_auto_training()

    def calculate_dynamic_state_size(self):
        """
        calcola la dimensione dello stato in modo dinamico
        """
        try:
            sample_asset = next(iter(self.active_assets))
            market_data = self.market_data[sample_asset]
            sequence_length = 10  # Assumendo che SEQUENCE_LENGTH sia 10 come in drl_agent.py
            num_features = market_data.select(pl.col(pl.NUMERIC_DTYPES)).shape[1]
            return sequence_length * num_features
        except Exception as e:
            logging.warning(f"⚠️ Errore nel calcolo dello state_size dinamico: {e}. Uso fallback {DESIRED_STATE_SIZE}.")
            return DESIRED_STATE_SIZE

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
        df.write_parquet(DATA_FILE, compression="zstd")
        logging.info("💾 Memoria IA aggiornata.")

    def update_performance(
        self, account, symbol, action,
        lot_size, profit, strategy
    ):
        """
        Aggiorna le informazioni di performance relative a
        un'operazione di trading.
        """
        # Carica i dati esistenti
        if TRADE_FILE.exists():
            df = pl.read_parquet(TRADE_FILE)
            df = df.with_columns([
                pl.col("lot_size").cast(pl.Float64),
                pl.col("profit").cast(pl.Float64)
            ])
        else:
            df = pl.DataFrame({
                "account": pl.Series([], dtype=pl.Utf8),
                "symbol": pl.Series([], dtype=pl.Utf8),
                "action": pl.Series([], dtype=pl.Utf8),
                "lot_size": pl.Series([], dtype=pl.Float64),
                "profit": pl.Series([], dtype=pl.Float64),
                "strategy": pl.Series([], dtype=pl.Utf8),
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
                "lot_size": [float(lot_size)],
                "profit": [float(profit)],
                "strategy": [strategy]
            })
            df = pl.concat([df, new_entry])

        df.write_parquet(TRADE_FILE, compression="zstd")
        logging.info(
            "📊 Trade aggiornato per %s su %s: Profit %s | Strategia: %s",
            account, symbol, profit, strategy
        )

    def adapt_lot_size(
        self, account, symbol, success_probability,
        confidence_score, predicted_volatility
    ):
        """
        Calcola la dimensione del lotto in modo ultra-intelligente e integrato.
        Combina risk manager, confidenza AI, ILQ zone, momentum e volatilità.
        """
        risk_manager = self.risk_manager[account]
        base_lot = risk_manager.calculate_position_size(
            self.balances[account], symbol
        )

        multiplier = 1.0
        if success_probability > 0.9 and confidence_score > 0.9:
            multiplier *= 1.5

        if predicted_volatility:
            multiplier *= np.clip(1 / (1 + predicted_volatility), 0.5, 1.2)

        multiplier *= np.clip(self.strategy_strength, 0.5, 3.0)

        final_lot = base_lot * multiplier

        max_lot = (
            self.balances[account] * risk_manager.risk_settings["max_exposure"]
        )
        return max(0.01, min(final_lot, max_lot))

    def execute_trade(
        self, account, symbol, action, lot_size, risk, strategy, sl, tp
    ):
        """
        Esegue un'operazione di trading su MetaTrader 5
        in base ai parametri specificati.
        """
        # Calcola Stop Loss e Take Profit
        current_price = mt5.symbol_info_tick(symbol).ask
        if hasattr(sl, "item"):
            sl = sl.item(-1)
        if hasattr(tp, "item"):
            tp = tp.item(-1)
        sl = sl if sl is not None else 0.0
        tp = tp if tp is not None else 0.0
        order = {
            "symbol": symbol,
            "volume": lot_size,
            "type": 0 if action == "buy" else 1,
            "price": current_price,
            "reference_price": mt5.symbol_info_tick(symbol).ask,
            "deviation": 10,
            "magic": 0,
            "comment": f"AI Trade ({strategy})",
            "type_time": 0,
            "type_filling": 1,
            "sl": sl,  # Stop Loss
            "tp": tp  # Take Profit
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
        """
        Analizza i dati di mercato per un determinato simbolo e decide
        se eseguire un'operazione di trading.
        Args:
        symbol (str): Il simbolo dell'asset da analizzare
        """
        market_data = get_normalized_market_data(symbol)

        if market_data is None or market_data.height == 0:
            logging.warning(
                "⚠️ Nessun dato per %s. Eseguo il backtest x migliorare",
                symbol
            )
            run_backtest(symbol, market_data)
            return False

        market_data = apply_all_advanced_features(market_data)
        market_data = apply_all_market_structure_signals(market_data)

        # PATCH TEST: forza la colonna dopo le feature da cancelare appena finito debug
        market_data = market_data.with_columns([
            pl.lit(10).alias("weighted_signal_score")
        ])
        print("DEBUG weighted_signal_score:", market_data["weighted_signal_score"][-5:])
        # fino a qua.

        # Calcolo degli embedding e del signal score
        if isinstance(market_data, dict):
            value = market_data.get(symbol, {}).get("weighted_signal_score", 0)
            if hasattr(value, "to_numpy"):
                try:
                    signal_score = int(value.to_numpy()[-1])
                except (IndexError, ValueError):
                    signal_score = 0
            else:
                try:
                    signal_score = int(value)
                except (ValueError, TypeError):
                    signal_score = 0
        else:
            try:
                signal_score = int(market_data.select("weighted_signal_score").to_numpy()[-1][0])
            except (KeyError, IndexError, TypeError):
                signal_score = 0

        embeddings = [
            get_embedding_for_symbol(symbol, tf) for tf in (
                ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            )
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
        full_state = sanitize_full_state(full_state)
        
        predicted_price = (
            self.price_predictor.predict_price(symbol, full_state)
        )

        for account in self.balances:
            if self.risk_manager[account].max_trades <= 0:
                logging.warning("❌ Max trades raggiunti per %s", account)
                continue

            action_rl, confidence_score, algo_used = (
                self.drl_super_manager.get_best_action_and_confidence(
                    full_state
                )
            )
            success_probability = confidence_score * pattern_confidence
            print(f"DEBUG: action_rl={action_rl}, confidence_score={confidence_score}, algo_used={algo_used}, pattern_confidence={pattern_confidence}")
            if action_rl == 1:
                action = "buy"
            elif action_rl == 2:
                action = "sell"
            else:
                logging.info(
                    "⚠️ AI ha suggerito HOLD. Nessuna operazione per %s.",
                    symbol
                )
                return

            if signal_score < 2:
                logging.info(
                    "⚠️ Segnale troppo debole su %s (score=%s).",
                    symbol, signal_score
                )
                return

            predicted_volatility = (
                self.volatility_predictor.predict_volatility(
                    full_state.reshape(1, -1)
                )[0]
            )

            sl, ts, tp = self.risk_manager[account].adaptive_stop_loss(
                market_data["close"].to_numpy()[-1], symbol
            )

            sl_val = float(sl) if isinstance(sl, (int, float)) else float(sl.to_numpy()[0])
            ts_val = float(ts) if isinstance(ts, (int, float)) else float(ts.to_numpy()[0])
            last_close = float(market_data["close"].to_numpy()[-1])
            if ts_val > sl_val and (ts_val - sl_val) < (0.002 * last_close):
                logging.info(
                    "⛔ Trailing Stop troppo stretto. Nessun trade su %s",
                    symbol
                )
                return

            lot_size = self.adapt_lot_size(
                account, symbol,
                success_probability, confidence_score,
                predicted_volatility
            )

            self.risk_manager[account].adjust_risk(symbol)

            if lot_size < 0.01:
                logging.warning(
                    "⛔ Lotto troppo piccolo, annullo trade su %s", symbol
                )
                return

            logging.info(
                "🤖 Azione AI: %s | Algo: %s | Confidenza: %.2f | Score: %d",
                action, algo_used, confidence_score, signal_score
            )

            # Strategia
            trade_profit = predicted_price - market_data["close"].to_numpy()[-1]
            strategy, strategy_weight = (
                self.strategy_generator.select_best_strategy(market_data)
            )
            self.execute_trade(
                account, symbol, action, lot_size,
                success_probability, strategy, sl, tp
            )
            self.strategy_strength = np.clip(
                self.strategy_strength * (1 + (strategy_weight - 0.5)),
                0.5, 3.0
            )
            self.strategy_generator.update_knowledge(
                profit=trade_profit,
                win_rate=1 if trade_profit > 0 else 0,
                drawdown=abs(min(0, trade_profit)),
                volatility=market_data["volatility"].item(-1)
            )
            self.volatility_predictor.update(
                full_state.reshape(1, -1),
                market_data["volatility"].item(-1)
            )

            if pattern_confidence < 0.3:
                return

            if success_probability > 0.5:

                self.risk_manager[account].max_trades -= 1
                self.drl_agent.update(
                    full_state, 1 if trade_profit > 0 else 0
                )
                self.drl_super_manager.update_all(
                    full_state, 1 if trade_profit > 0 else 0
                )
                # Controllo dimensione stato per evitare errori RL buffer
                expected_state_size = self.drl_super_manager.super_agents[algo_used].env.observation_space.shape[0]
                if full_state.shape[0] == expected_state_size:
                    self.drl_super_manager.reinforce_best_agent(full_state, 1)
                else:
                    logging.warning(
                        f"⚠️ Stato RL non coerente: atteso {expected_state_size}, ricevuto {full_state.shape[0]}. Reinforcement saltato."
                    )
            else:
                logging.info(
                    "🚫 Nessun trade su %s per %s. Avvio Demo.",
                    symbol, account
                    )
                demo_trade(symbol, market_data)
                self.drl_agent.update(full_state, 1 if trade_profit > 0 else 0)
                self.drl_super_manager.update_all(full_state, 1 if trade_profit > 0 else 0)


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

    threading.Thread(
        target=ai_model.strategy_generator.continuous_self_improvement,
        daemon=True
    ).start()

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
