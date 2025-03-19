# strategy_generator.py
import logging
import inspect
from datetime import datetime
from pathlib import Path
import polars as pl
import numpy as np
from indicators import TradingIndicators  # Importa tutti gli indicatori

# 📂 Percorso del file di strategie
MODEL_DIR = Path("/mnt/usb_trading_data/models") if Path("/mnt/usb_trading_data").exists() else Path("D:/trading_data/models")
STRATEGY_FILE = MODEL_DIR / "strategies.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class StrategyGenerator:
    def __init__(self):
        self.indicators = TradingIndicators()
        self.all_indicators = self.get_all_indicators()  # 🔥 Recupera automaticamente tutti gli indicatori
        self.strategy_weights = self.load_strategies()

    def get_all_indicators(self):
        """🔄 Recupera automaticamente tutti gli indicatori definiti in indicators.py"""
        return {name: method for name, method in inspect.getmembers(self.indicators, predicate=inspect.ismethod)}

    def load_strategies(self):
        """📥 Carica le strategie dal file Parquet, usa un valore compresso per evitare sprechi"""
        if STRATEGY_FILE.exists():
            logging.info("📥 Caricamento strategie da Parquet...")
            df = pl.read_parquet(STRATEGY_FILE)
            return {row["strategy"]: row["weight"] for row in df.iter_rows(named=True)}
        else:
            logging.info("📄 Nessuna strategia trovata. Creazione di default.")
            return {}

    def save_strategies(self):
        """💾 Salva le strategie in un file Parquet ottimizzato"""
        df = pl.DataFrame({"strategy": list(self.strategy_weights.keys()), "weight": list(self.strategy_weights.values())})
        df.write_parquet(STRATEGY_FILE, compression="zstd", mode="overwrite")
        logging.info("💾 Strategie salvate in formato compresso.")

    def update_strategies(self, strategy_name, profit, win_rate, drawdown, volatility):
        """
        📊 Migliora il peso della strategia basandosi su più parametri:
        - Profitto
        - Win Rate (percentuale di successo)
        - Drawdown (perdita massima subita)
        - Volatilità (per evitare rischi eccessivi)
        """
        if strategy_name in self.strategy_weights:
            efficiency_score = (profit * 0.5) + (win_rate * 0.3) - (drawdown * 0.1) - (volatility * 0.1)
            new_weight = np.clip(self.strategy_weights[strategy_name] + (efficiency_score / 100), 0, 1)

            self.strategy_weights[strategy_name] = new_weight
            self.save_strategies()
            logging.info(f"📊 Strategia aggiornata: {strategy_name} → Nuovo peso: {new_weight:.3f} | Score: {efficiency_score:.2f}")
        else:
            logging.warning(f"⚠️ Strategia {strategy_name} non trovata!")

    def select_best_strategy(self, market_data):
        """
        🔥 Crea automaticamente strategie combinando **tutti** gli indicatori
        📊 Seleziona la strategia migliore in base ai valori di mercato attuali
        """
        indicator_values = {}
        for name, func in self.all_indicators.items():
            try:
                if "period" in inspect.signature(func).parameters:
                    indicator_values[name] = func(market_data, period=50)  # Se l'indicatore ha un parametro "period", usa 50 di default
                else:
                    indicator_values[name] = func(market_data)
            except Exception as e:
                logging.warning(f"⚠️ Errore nel calcolo dell'indicatore {name}: {e}")

        # 🔄 Strategie dinamiche basate su combinazioni di indicatori
        strategy_conditions = {
            "scalping": (indicator_values["calculate_rsi"] < 30 and 
                         indicator_values["calculate_macd"][0] > indicator_values["calculate_macd"][1] and 
                         indicator_values["calculate_adx"] > 25),

            "mean_reversion": (indicator_values["calculate_rsi"] > 70 and 
                               indicator_values["calculate_macd"][0] < indicator_values["calculate_macd"][1] and 
                               indicator_values["calculate_bollinger_bands"]["upper"] > market_data["close"].iloc[-1]),

            "trend_following": (indicator_values["calculate_ema"] > indicator_values["calculate_ema"](market_data, period=200) and 
                                indicator_values["calculate_vwap"] > market_data["close"].iloc[-1]),

            "swing": (indicator_values["calculate_stochastic"]["k"] < 20 and 
                      indicator_values["calculate_stochastic"]["d"] < 20),

            "momentum": (indicator_values["calculate_momentum"] > 100 and 
                         indicator_values["calculate_adx"] > 20),

            "breakout": (indicator_values["calculate_donchian_channels"]["upper"] < market_data["high"].iloc[-1] and 
                         indicator_values["calculate_volatility"] > 1.5),

            "ai_generated": (indicator_values["calculate_ai_score"] > 0.75)  # 🔥 Usa un punteggio AI per nuove strategie
        }

        # 🔥 Sceglie la strategia più forte in base ai dati di mercato
        for strategy, condition in strategy_conditions.items():
            if condition:
                return strategy, self.strategy_weights.get(strategy, 0.5)  # Se non ha peso, usa 0.5 come default

        return "swing", self.strategy_weights.get("swing", 0.5)  # Default se nessuna condizione è soddisfatta

# ✅ Test rapido del generatore di strategie
if __name__ == "__main__":
    sg = StrategyGenerator()
    print("📊 Strategie caricate:", sg.strategy_weights)
