# -*- coding: utf-8 -*-
"""
Script di test `test_bot_simulato.py`:
Simula in modo realistico ma controllato
il funzionamento del bot di trading.
Verifica inizializzazione, creazione pattern, gestione posizioni,
addestramento strategie, analisi tecnica e
integrazione con gestione del rischio (RM),
senza chiamate reali a MetaTrader5 o API esterne.
Stampa un report dettagliato di cosa è stato testato,
funzionamenti ed eventuali anomalie.
"""

import logging
import asyncio
import sys
import subprocess
import types
import pandas as pd
import numpy as np
import data_handler
import ai_model
import smart_features
import strategy_generator
import drl_agent
import drl_super_integration
import price_prediction
import volatility_tools
import risk_management
from position_manager import PositionManager
import pattern_brain
import demo_module

# Configura il logger per scrivere sia sulla console che in un file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Scrive sulla console
        logging.FileHandler("module_load_log.txt", mode="w")  # Scrive in un file
    ]
)


# Reindirizza i messaggi di print al logger
class PrintLogger:
    def write(self, message):
        if message.strip():  # Ignora righe vuote
            logging.info(message.strip())

    def flush(self):
        pass  # Necessario per compatibilità con sys.stdout


sys.stdout = PrintLogger()


# --- Simulazione ambiente MetaTrader5 e altre dipendenze esterne ---

# Dummy MetaTrader5: simula funzioni chiave (connessione, tick, posizioni, ordini)
class DummyMT5:
    ORDER_BUY = 0
    ORDER_SELL = 1
    TRADE_RETCODE_DONE = 0  # codice successo invio ordine

    class _TickInfo:
        def __init__(self, ask, bid):
            self.ask = ask
            self.bid = bid

    def __init__(self):
        self.positions = []  # lista posizioni aperte simulate

    def account_info(self):
        # Restituisce informazioni conto simulate (saldo)
        return types.SimpleNamespace(balance=10000.0)

    def initialize(self):
        # Simula inizializzazione riuscita
        return True

    def shutdown(self):
        # Simula chiusura connessione
        return True

    def symbol_info_tick(self, symbol):
        # Fornisce prezzi ask/bid fittizi per ciascun simbolo
        if symbol == "ASSET1":
            return DummyMT5._TickInfo(ask=112.0, bid=111.0)
        elif symbol == "ASSET2":
            return DummyMT5._TickInfo(ask=107.0, bid=106.0)
        else:
            return DummyMT5._TickInfo(ask=100.0, bid=100.0)

    def positions_get(self):
        # Restituisce l'elenco delle posizioni aperte aggiornando i profitti simulati
        for pos in self.positions:
            tick = self.symbol_info_tick(pos.symbol)
            if pos.type == 0:  # posizione long (buy)
                pos.profit = (tick.bid - pos.price_open) * pos.volume * 100
            else:  # posizione short (sell)
                pos.profit = (pos.price_open - tick.ask) * pos.volume * 100
        return list(self.positions)

    def order_send(self, request):
        # Simula invio ordini di apertura o chiusura posizione
        result = types.SimpleNamespace(retcode=DummyMT5.TRADE_RETCODE_DONE, profit=0.0)
        if 'action' in request:
            # Chiusura posizione: rimuove la posizione corrispondente
            symbol = request['symbol']
            volume = request['volume']
            self.positions = [
                pos for pos in self.positions
                if not (pos.symbol == symbol and abs(pos.volume - volume) < 1e-6)
            ]
        else:
            # Apertura nuova posizione: aggiunge alla lista
            symbol = request.get('symbol')
            volume = request.get('volume', 0.0)
            order_type = request.get('type')
            price = request.get('price', 0.0)
            pos_type = 0 if order_type == DummyMT5.ORDER_BUY else 1
            new_pos = types.SimpleNamespace(symbol=symbol, volume=volume, type=pos_type,
                                            price_open=price, profit=0.0)
            self.positions.append(new_pos)
        return result


# Inserisce DummyMT5 come modulo MetaTrader5 per l'uso nel codice del bot
dummy_mt5 = DummyMT5()
sys.modules['MetaTrader5'] = dummy_mt5

# Dummy per altre librerie esterne (tensorflow, sklearn, joblib) per evitare errori di import
# TensorFlow (usato da PatternBrain e PricePredictionModel) – definiamo solo quanto basta
dummy_tf = types.ModuleType("tensorflow")
dummy_tf.keras = types.SimpleNamespace(
    models=types.SimpleNamespace(load_model=lambda path: None),
    Sequential=lambda layers=None: types.SimpleNamespace(
        predict=lambda x, verbose=0: np.array([[0.5]]),  # sempre restituisce 0.5
        compile=lambda **kwargs: None
    )
)
sys.modules['tensorflow'] = dummy_tf

# Scikit-learn (MinMaxScaler in PricePredictionModel)
dummy_sklearn = types.ModuleType("sklearn")
dummy_preproc = types.ModuleType("sklearn.preprocessing")


class DummyScaler:
    def fit(self, X): return self
    def fit_transform(self, X): return X
    def transform(self, X): return X
    def inverse_transform(self, X): return X


dummy_preproc.MinMaxScaler = DummyScaler
dummy_sklearn.preprocessing = dummy_preproc
sys.modules['sklearn'] = dummy_sklearn
sys.modules['sklearn.preprocessing'] = dummy_preproc

# Joblib (usato per salvare/caricare modelli RL)
sys.modules['joblib'] = types.SimpleNamespace(
    dump=lambda obj, path: None,
    load=lambda path: types.SimpleNamespace()
)


# Polars (usato per dati e file Parquet) – simuliamo DataFrame e funzioni essenziali
class ExtendedDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return ExtendedDataFrame

    @property
    def height(self):
        # Proprietà simile a pl.DataFrame.height
        return self.shape[0]

    def is_empty(self):
        return self.shape[0] == 0

    def __getitem__(self, key):
        # Consente indicizzazione per riga come polars (es: df[-1] ultimo elemento)
        if isinstance(key, int):
            return self.iloc[key]
        else:

            return super().__getitem__(key)

    def write_parquet(self, file, compression="zstd", mode="overwrite"):
        # Simula salvataggio Parquet (non crea file reale, solo log)
        logging.info(f"(Dummy) Writing DataFrame to {file}")
        return True


dummy_polars = types.ModuleType("polars")
dummy_polars.DataFrame = ExtendedDataFrame
dummy_polars.Series = lambda name=None, data=None: pd.Series(data) if data is not None else pd.Series(dtype=float)
dummy_polars.read_parquet = lambda path: ExtendedDataFrame()
dummy_polars.exceptions = types.SimpleNamespace(PolarsError=Exception)
sys.modules['polars'] = dummy_polars


# Dummy subprocess.Popen (avvia Super Agent Runner) – evita apertura vero processo
class DummyPopen:

    def __init__(self, args, **kwargs):
        self.pid = 12345  # PID fittizio

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        return False  # non sopprime eccezioni


subprocess.Popen = DummyPopen


# --- Import moduli del bot con i dummy applicati ---

# --- Patch delle funzioni e classi del bot per usare i dummy e simulare i dati ---

# Configurazione: usa asset preset (evita caricamento dinamico da API)
data_handler.USE_PRESET_ASSETS = True
data_handler.load_preset_assets = lambda: {"preset": ["ASSET1", "ASSET2"]}
data_handler.dynamic_assets_loading = lambda mapping: None
data_handler.load_auto_symbol_mapping = lambda: {"ASSET1": "ASSET1", "ASSET2": "ASSET2"}
data_handler.get_available_assets = lambda: ["ASSET1", "ASSET2"]

# Gestione Rischio: restituisce un rischio dinamico fisso (0.2) per semplificare calcolo lot
risk_management.RiskManagement.calculate_dynamic_risk = lambda self, market_data: 0.2

# PatternBrain: sostituisce modello ML con dummy (confidence fissa 0.7)
pattern_brain.PatternBrain = type('DummyPatternBrain', (object,), {
    "__init__": lambda self, input_size=5: None,
    "predict_score": lambda self, pattern_array: 0.7
})

# DRLAgent: agent RL semplice che logga aggiornamenti
drl_agent.DRLAgent = type('DummyDRLAgent', (object,), {
    "__init__": lambda self: None,
    "update": lambda self, state, outcome: logging.info(f"DRLAgent updated with outcome {outcome}")
})

# DRLSuperManager: gestore RL avanzato con 4 algoritmi – simulato con output predefiniti
drl_super_integration.DRLSuperManager = type('DummyDRLSuperManager', (object,), {
    "__init__": lambda self, state_size=512: setattr(self, "call_count", 0) or None,
    "load_all": lambda self: logging.info("DRLSuperManager: load_all called (dummy)"),
    "start_auto_training": lambda self: logging.info("DRLSuperManager: start_auto_training called (dummy)"),
    "get_best_action_and_confidence": lambda self, full_state: (
        # Incrementa il contatore delle chiamate
        self.call_count += 1 or None,
        # Ritorna azioni e confidence diversi a seconda della chiamata (per testare vari scenari)
        [(1, 0.8, "PPO"),   # 1ª chiamata: suggerisce BUY con confidenza 0.8
         (1, 0.4, "PPO"),   # 2ª chiamata: suggerisce BUY con confidenza bassa 0.4
         (2, 0.6, "PPO"),   # 3ª chiamata: suggerisce SELL (chiusura) con confidenza 0.6
         (1, 0.2, "PPO")]   # 4ª chiamata: suggerisce BUY con confidenza molto bassa 0.2
        [min(self.call_count, 3)]  # usa indice in base a call_count incrementato
    ),
    "update_all": lambda self, full_state, outcome: logging.info(f"DRLSuperManager: update_all called with outcome {outcome}"),
    "reinforce_best_agent": lambda self, full_state, outcome: logging.info("DRLSuperManager: reinforce_best_agent called (dummy)")
})

# Indicatori tecnici: get_embedding_for_symbol restituisce embedding fittizio (array costante)
smart_features.get_embedding_for_symbol = lambda symbol, timeframe: np.array([0.1, 0.1, 0.1])

# Modello di Predizione Prezzi: simulato per restituire prezzo futuro leggermente diverso dall'ultimo
price_prediction.PricePredictionModel = type('DummyPricePredictor', (object,), {
    "__init__": lambda self: None,
    "predict_price": lambda self, asset, full_state=None: (
        float(market_data_map[asset]["close"].iloc[-1]) + 5.0  # prezzo futuro = ultimo close +5
        if asset == "ASSET1" else
        float(market_data_map[asset]["close"].iloc[-1]) - 2.0  # prezzo futuro = ultimo close -2
        if asset == "ASSET2" else 0.0
    )
})

# Ottimizzatore di portafoglio e core (background optimization) – dummy che logga le chiamate
if 'optimizer_core' in sys.modules:
    import optimizer_core
    optimizer_core.PortfolioOptimizer = type('DummyPortfolioOptimizer', (object,), {
        "__init__": lambda self, market_data, balances, flag: logging.info("PortfolioOptimizer initialized (dummy)"),
        "optimize": lambda self: logging.info("PortfolioOptimizer optimize called (dummy)")
    })
    optimizer_core.OptimizerCore = type('DummyOptimizerCore', (object,), {
        "__init__": lambda self, **kwargs: logging.info("OptimizerCore initialized (dummy)"),
        "run_full_optimization": lambda self: logging.info("OptimizerCore: run_full_optimization called (dummy)")
    })

# UpdatePerformance: evita scrittura file, logga l'aggiornamento performance trade
ai_model.AIModel.update_performance = lambda self, account, symbol, action, lot_size, profit, strategy: \
    logging.info(f"UpdatePerformance: {account} {symbol} Profit={profit:.2f} Strat={strategy}")

# --- Dati di mercato simulati (1-2 asset, 2 barre per asset) ---

# Creiamo dati di esempio per 2 asset (ASSET1 e ASSET2) con colonne richieste
data_asset1 = {
    "open": [100.0, 101.0],
    "high": [102.0, 113.0],
    "low": [99.0, 101.0],
    "close": [100.0, 110.0],           # prezzo chiusura sale da 100 a 110
    "volume": [1000.0, 1100.0],
    "price_change": [0.0, 10.0],       # variazione di prezzo (close diff)
    "rsi": [50.0, 20.0],               # RSI scende (esempio: oversold a 20)
    "bollinger_width": [0.05, 0.04],
    "spread": [0.0005, 0.0006],
    "latency": [50.0, 40.0],
    "ILQ_Zone": [0, 1],
    "fakeout_up": [0, 0],
    "fakeout_down": [0, 0],
    "volatility_squeeze": [0, 0],
    "micro_pattern_hft": [0, 0],
    "volatility": [1.2, 1.3],
    "weighted_signal_score": [3, 4]
}
data_asset2 = {
    "open": [100.0, 100.0],
    "high": [101.0, 107.0],
    "low": [99.0, 100.0],
    "close": [100.0, 105.0],           # prezzo chiusura sale da 100 a 105
    "volume": [800.0, 900.0],
    "price_change": [0.0, 5.0],
    "rsi": [50.0, 40.0],
    "bollinger_width": [0.05, 0.05],
    "spread": [0.0005, 0.0005],
    "latency": [50.0, 55.0],
    "ILQ_Zone": [0, 0],
    "fakeout_up": [0, 0],
    "fakeout_down": [0, 0],
    "volatility_squeeze": [0, 0],
    "micro_pattern_hft": [0, 1],
    "volatility": [1.0, 1.1],
    "weighted_signal_score": [3, 2]
}
df_asset1 = ExtendedDataFrame(data_asset1)
df_asset2 = ExtendedDataFrame(data_asset2)
market_data_map = {"ASSET1": df_asset1, "ASSET2": df_asset2}

# Patch final di get_normalized_market_data per restituire i dati simulati dal dizionario
data_handler.get_normalized_market_data = lambda symbol: market_data_map.get(symbol)

# Patch fetch_account_balances per evitare chiamate MetaTrader e usare valori fissi
ai_model.fetch_account_balances = lambda: {"Danny": 10000.0, "Giuseppe": 10000.0}

# --- Inizio test simulato ---
print("# Testing Trading Bot Simulation #")

# 1. Inizializzazione del sistema di trading
print("\n## Initializing Trading System ##")
try:
    trading_system = ai_model.AIModel(market_data_map, ai_model.fetch_account_balances())
    print("Initialization: SUCCESS - TradingSystem and components initialized.")
except Exception as e:
    print(f"Initialization: FAILURE - Exception occurred: {e}")
    trading_system = None

# Verifica asset attivi selezionati
if trading_system:
    try:
        assets_selected = trading_system.active_assets
        print(f"Active assets selected for trading: {assets_selected}")
    except Exception as e:
        print(f"Error during asset selection: {e}")

# 2. Simulazione ciclo decisionale trading sugli asset attivi
print("\n## Simulating Trading Decisions for Active Assets ##")


async def run_decisions():
    for asset in trading_system.active_assets:
        print(f"\n### Deciding trade for {asset} ###")
        try:
            result = await trading_system.decide_trade(asset)
            if result is False:
                print(
                    f"No trade executed for {asset} (no sufficient data or skipped)."
                )
            else:
                print(f"Trade decision completed for {asset}.")
        except Exception as e:
            print(f"Error during trade decision for {asset}: {e}")
asyncio.run(run_decisions())

if not any(pos.symbol == "ASSET2" for pos in dummy_mt5.positions):
    dummy_mt5.positions.append(
        types.SimpleNamespace(
            symbol="ASSET2", volume=0.1, type=0, price_open=100.0, profit=0.0
            )
    )
    print(
        """
        Note: Add a simulated existing BUY position for
        ASSET2 to test position management.
        """
    )

# Inizializza il gestore delle posizioni
position_manager = PositionManager()

# 4. Monitoraggio e gestione posizioni aperte
print("\n## Monitoring and Managing Open Positions ##")
try:
    position_manager.monitor_open_positions()
    print("Open positions monitored and managed.")
except Exception as e:
    print(f"Error during position monitoring: {e}")

# 5. Report finale posizioni ancora aperte o chiuse
remaining_positions = dummy_mt5.positions
if remaining_positions:
    print("\nPositions still open after monitoring:")
    for pos in remaining_positions:
        print(f" - {pos.symbol}: Volume={pos.volume}, Profit={pos.profit:.2f}")
else:
    print("\nAll positions are closed after monitoring.")

print("\n## Test Simulation Complete ##")
