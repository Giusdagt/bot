import logging
import gc
import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime

MODEL_DIR = (
    Path("/mnt/usb_trading_data/models") 
    if Path("/mnt/usb_trading_data").exists() 
    else Path("D:/trading_data/models")
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_FILE = MODEL_DIR / "strategies_compressed.parquet"
ANOMALY_FILE = MODEL_DIR / "market_anomalies.parquet"
KNOWLEDGE_FILE = MODEL_DIR / "compressed_knowledge.parquet"
AI_MEMORY_FILE = MODEL_DIR / "ai_memory.parquet"
TRADE_FILE = MODEL_DIR / "trades.parquet"
PERFORMANCE_FILE = MODEL_DIR / "performance.parquet"

class OptimizerCore:
    def __init__(self, strategy_generator=None, ai_model=None):
        self.sg = strategy_generator
        self.ai = ai_model

    def optimize_strategies(self):
        if not self.sg:
            logging.warning("⚠️ Nessun strategy_generator collegato.")
            return

        top_strategies = dict(list(self.sg.generated_strategies.items())[:5])
        self.sg.generated_strategies.clear()
        self.sg.generated_strategies.update(top_strategies)

        df = pl.DataFrame({"strategies": [str(self.sg.generated_strategies)]})
        df.write_parquet(STRATEGY_FILE, compression="zstd", mode="overwrite")
        logging.info("✅ Strategie ottimizzate e salvate.")

    def optimize_anomalies(self):
        if not self.sg:
            return
        limited_anomalies = self.sg.market_anomalies[-50:]  # Mantiene solo le ultime 50
        self.sg.market_anomalies = limited_anomalies
        df = pl.DataFrame({"anomalies": [limited_anomalies]})
        df.write_parquet(ANOMALY_FILE, compression="zstd", mode="overwrite")
        logging.info("✅ Anomalie salvate e ottimizzate.")

    def optimize_knowledge(self):
        if not self.sg:
            return

        ck = self.sg.compressed_knowledge
        while len(ck) > 25:
            ck = np.mean(ck.reshape(-1, 2), axis=1)
        self.sg.compressed_knowledge = ck

        df = pl.DataFrame({"knowledge": [ck.tobytes()]})
        df.write_parquet(KNOWLEDGE_FILE, compression="zstd", mode="overwrite")
        logging.info("🧠 Conoscenza compressa e salvata.")

    def optimize_ai_memory(self):
        if not self.ai:
            return

        if AI_MEMORY_FILE.exists():
            mem = pl.read_parquet(AI_MEMORY_FILE)["memory"].to_numpy()
            if len(mem) > 10:
                mem = mem[-10:]  # Mantieni solo le ultime 10 entry
            mean_mem = np.mean(mem)
            df = pl.DataFrame({"memory": [mean_mem]})
            df.write_parquet(AI_MEMORY_FILE, compression="zstd", mode="overwrite")
            logging.info("🧠 Memoria AI consolidata.")

    def optimize_trades(self):
        if not TRADE_FILE.exists():
            return
        df = pl.read_parquet(TRADE_FILE)
        df = df.sort("profit", descending=True).head(100)  # Mantieni solo i top 100 trade
        df.write_parquet(TRADE_FILE, compression="zstd", mode="overwrite")
        logging.info("📊 Trade compressi e ottimizzati.")

    def optimize_performance(self):
        if not PERFORMANCE_FILE.exists():
            return
        df = pl.read_parquet(PERFORMANCE_FILE)
        df = df.tail(100)  # Ultime 100 righe di performance
        df.write_parquet(PERFORMANCE_FILE, compression="zstd", mode="overwrite")
        logging.info("📈 Performance ottimizzate.")

    def evaluate_evolution(self, profit, win_rate, drawdown, volatility, strategy_strength):
        score = (
            (profit * 0.5) +
            (win_rate * 0.3) -
            (drawdown * 0.1) -
            (volatility * 0.1) +
            (strategy_strength * 0.5)
        )
        evolution_score = max(0, min(100, score * 10))
        logging.info(f"📊 Evoluzione Strategica: {evolution_score:.2f} / 100")
        return evolution_score

    def clean_ram(self):
        gc.collect()
        logging.info("🧹 RAM pulita.")

    def run_full_optimization(self):
        self.optimize_strategies()
        self.optimize_anomalies()
        self.optimize_knowledge()
        self.optimize_ai_memory()
        self.optimize_trades()
        self.optimize_performance()
        self.clean_ram()
        logging.info("✅ Ottimizzazione completa eseguita.")
