import logging
from datetime import datetime
from pathlib import Path
import polars as pl
import numpy as np

# 📂 Percorso del file di strategie
MODEL_DIR = Path("/mnt/usb_trading_data/models") if Path("/mnt/usb_trading_data").exists() else Path("D:/trading_data/models")
STRATEGY_FILE = MODEL_DIR / "strategies.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 🎯 Strategie disponibili (vengono generate e migliorate automaticamente)
strategies = {
    "scalping": {"volatility_threshold": 1.5, "base_weight": 0.6},
    "swing": {"volatility_range": (0.5, 1.5), "base_weight": 0.8},
    "mean_reversion": {"volatility_threshold": 0.5, "base_weight": 0.7},
    "trend_following": {"trend_strength_threshold": 0.75, "base_weight": 0.9}
}

class StrategyGenerator:
    def __init__(self):
        self.strategy_weights = self.load_strategies()

    def load_strategies(self):
        """Carica le strategie dal file, usa un valore compresso per evitare sprechi."""
        if STRATEGY_FILE.exists():
            logging.info("📥 Caricamento strategie da Parquet...")
            df = pl.read_parquet(STRATEGY_FILE)
            return {row["strategy"]: row["weight"] for row in df.iter_rows(named=True)}
        else:
            logging.info("📄 Nessuna strategia trovata. Creazione di default.")
            return {name: data["base_weight"] for name, data in strategies.items()}

    def save_strategies(self):
        """Salva le strategie in un file Parquet ottimizzato."""
        df = pl.DataFrame({"strategy": list(self.strategy_weights.keys()), "weight": list(self.strategy_weights.values())})
        df.write_parquet(STRATEGY_FILE, compression="zstd", mode="overwrite")
        logging.info("💾 Strategie salvate in formato compresso.")

    def update_strategies(self, strategy_name, performance):
        """
        Migliora il peso della strategia in base alle performance.
        Se una strategia è efficace, il suo peso aumenta, altrimenti si riduce.
        """
        if strategy_name in self.strategy_weights:
            new_weight = np.clip(self.strategy_weights[strategy_name] + (performance / 100), 0, 1)
            self.strategy_weights[strategy_name] = new_weight
            self.save_strategies()
            logging.info(f"📊 Strategia aggiornata: {strategy_name} → Nuovo peso: {new_weight:.3f}")
        else:
            logging.warning(f"⚠️ Strategia {strategy_name} non trovata!")

    def select_best_strategy(self, market_data):
        """
        Sceglie automaticamente la strategia migliore in base alla volatilità e alla tendenza.
        Ritorna il nome della strategia e il suo valore aggiornato.
        """
        volatility = market_data["volatility"].iloc[-1]
        trend_strength = market_data["trend_strength"].iloc[-1]

        if volatility > strategies["scalping"]["volatility_threshold"]:
            return "scalping", self.strategy_weights["scalping"]
        elif volatility < strategies["mean_reversion"]["volatility_threshold"]:
            return "mean_reversion", self.strategy_weights["mean_reversion"]
        elif trend_strength > strategies["trend_following"]["trend_strength_threshold"]:
            return "trend_following", self.strategy_weights["trend_following"]
        else:
            return "swing", self.strategy_weights["swing"]

# ✅ Test rapido del generatore di strategie
if __name__ == "__main__":
    sg = StrategyGenerator()
    print("📊 Strategie caricate:", sg.strategy_weights)
