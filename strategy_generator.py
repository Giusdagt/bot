import logging
from datetime import datetime
from pathlib import Path
import polars as pl
import numpy as np
import inspect
from indicators import TradingIndicators  # 📈 Importa tutti gli indicatori

# 📂 Percorso del file di strategie
MODEL_DIR = Path("/mnt/usb_trading_data/models") if Path("/mnt/usb_trading_data").exists() else Path("D:/trading_data/models")
STRATEGY_FILE = MODEL_DIR / "strategies_compressed.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class StrategyGenerator:
    def __init__(self):
        self.indicators = TradingIndicators()
        self.all_indicators = self.get_all_indicators()  # 🔥 Recupera automaticamente tutti gli indicatori
        self.compressed_knowledge = self.load_compressed_knowledge()
        self.market_anomalies = []  # Per registrare le falle del sistema
        self.generated_strategies = {}  # Per memorizzare le nuove strategie generate

    def get_all_indicators(self):
        """🔄 Recupera automaticamente tutti gli indicatori definiti in indicators.py"""
        return {name: method for name, method in inspect.getmembers(self.indicators, predicate=inspect.ismethod)}

    def load_compressed_knowledge(self):
        """📥 Carica la conoscenza delle strategie in formato compresso, evitando sprechi di dati"""
        if STRATEGY_FILE.exists():
            logging.info("📥 Caricamento conoscenza strategica compressa...")
            df = pl.read_parquet(STRATEGY_FILE)
            return np.frombuffer(df["knowledge"][0], dtype=np.float32)
        else:
            logging.info("📄 Nessuna conoscenza trovata. Creazione di default.")
            return np.zeros(100, dtype=np.float32)  # Array di default con 100 elementi

    def save_compressed_knowledge(self):
        """💾 Salva la conoscenza strategica in formato numerico ultra-compatto"""
        df = pl.DataFrame({"knowledge": [self.compressed_knowledge.tobytes()]})
        df.write_parquet(STRATEGY_FILE, compression="zstd", mode="overwrite")
        logging.info("💾 Conoscenza strategica aggiornata e compressa.")

    def detect_market_anomalies(self, market_data):
        """Analizza anomalie di mercato (Stop Hunt, Frontrunning, ecc.)."""
        high_volatility = market_data["volatility"].iloc[-1] > 2.0
        sudden_volume_spike = market_data["volume"].iloc[-1] > market_data["volume"].mean() * 3

        if high_volatility or sudden_volume_spike:
            self.market_anomalies.append("Manipolazione Rilevata")
            logging.warning("⚠️ POSSIBILE MANIPOLAZIONE DEL MERCATO RILEVATA!")

    def update_knowledge(self, profit, win_rate, drawdown, volatility):
        """🔥 Migliora la conoscenza strategica basandosi su performance reali"""
        efficiency_score = (profit * 0.5) + (win_rate * 0.3) - (drawdown * 0.1) - (volatility * 0.1)
        self.compressed_knowledge = np.clip(self.compressed_knowledge + (efficiency_score / 1000), 0, 1)
        self.save_compressed_knowledge()
        logging.info(f"📊 Conoscenza aggiornata → Nuovo valore: {self.compressed_knowledge.mean():.6f} | Score: {efficiency_score:.3f}")

    def generate_new_strategies(self, market_data):
        """Genera nuove strategie combinando dinamicamente gli indicatori esistenti."""
        new_strategies = {}
        indicator_values = {name: func(market_data) for name, func in self.all_indicators.items()}

        # Esempi di nuove strategie generate
        new_strategies["strategy_1"] = (indicator_values["RSI"] < 30 and 
                                        indicator_values["MACD"] > indicator_values["MACD_Signal"] and 
                                        indicator_values["ADX"] > 25)

        new_strategies["strategy_2"] = (indicator_values["RSI"] > 70 and 
                                        indicator_values["MACD"] < indicator_values["MACD_Signal"] and 
                                        indicator_values["BB_Upper"] > market_data["close"].iloc[-1])

        new_strategies["strategy_3"] = (indicator_values["EMA_50"] > indicator_values["EMA_200"] and 
                                        indicator_values["VWAP"] > market_data["close"].iloc[-1])

        # Aggiungi le nuove strategie generate alla collezione di strategie generate
        self.generated_strategies.update(new_strategies)
        logging.info("🔄 Nuove strategie generate e aggiunte alla collezione.")

    def select_best_strategy(self, market_data):
        """🔥 Crea e seleziona la strategia migliore in base ai valori di mercato attuali"""
        self.detect_market_anomalies(market_data)
        self.generate_new_strategies(market_data)
        
        indicator_values = {}
        for name, func in self.all_indicators.items():
            try:
                if "period" in inspect.signature(func).parameters:
                    indicator_values[name] = func(market_data, period=50)
                else:
                    indicator_values[name] = func(market_data)
            except Exception as e:
                logging.warning(f"⚠️ Errore nel calcolo dell'indicatore {name}: {e}")

        # 🔄 Strategie dinamiche basate su combinazioni di indicatori
        strategy_conditions = {
            **self.generated_strategies,
            "scalping": (indicator_values["RSI"] < 30 and 
                         indicator_values["MACD"] > indicator_values["MACD_Signal"] and 
                         indicator_values["ADX"] > 25 and 
                         self.compressed_knowledge.mean() > 0.6),

            "mean_reversion": (indicator_values["RSI"] > 70 and 
                               indicator_values["MACD"] < indicator_values["MACD_Signal"] and 
                               indicator_values["BB_Upper"] > market_data["close"].iloc[-1] and 
                               self.compressed_knowledge.mean() > 0.5),

            "trend_following": (indicator_values["EMA_50"] > 
                                indicator_values["EMA_200"] and 
                                indicator_values["VWAP"] > market_data["close"].iloc[-1] and 
                                self.compressed_knowledge.mean() > 0.4),

            "swing": (indicator_values["STOCH_K"] < 20 and 
                      indicator_values["STOCH_D"] < 20 and 
                      self.compressed_knowledge.mean() > 0.3),

            "momentum": (indicator_values["momentum"] > 100 and 
                         indicator_values["ADX"] > 20 and 
                         self.compressed_knowledge.mean() > 0.2),

            "breakout": (indicator_values["Donchian_Upper"] < market_data["high"].iloc[-1] and 
                         indicator_values["volatility"] > 1.5 and 
                         self.compressed_knowledge.mean() > 0.7),

            "ai_generated": (self.compressed_knowledge.mean() > 0.75)  # 🔥 Strategie auto-evolute
        }

        for strategy, condition in strategy_conditions.items():
            if condition:
                return strategy, self.compressed_knowledge.mean()

        return "swing", self.compressed_knowledge.mean()  # Default se nessuna condizione è soddisfatta

# ✅ Test rapido del generatore di strategie
if __name__ == "__main__":
    sg = StrategyGenerator()
    print("📊 Conoscenza strategica caricata:", sg.compressed_knowledge.mean())
