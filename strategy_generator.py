"""
Modulo Strategy Generator
Questo modulo definisce la classe StrategyGenerator per la generazione
e l'ottimizzazione di strategie di trading algoritmico.
Include funzionalità
per rilevare anomalie di mercato, generare strategie basate su indicatori
tecnici e migliorare continuamente le strategie in base alle prestazioni.
"""
import inspect
import logging
import threading
import time
from pathlib import Path
import numpy as np
import polars as pl
from indicators import TradingIndicators

# Configura il logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

MODEL_DIR = (
    Path("/mnt/usb_trading_data/models")
    if Path("/mnt/usb_trading_data").exists()
    else Path("D:/trading_data/models")
)
STRATEGY_FILE = MODEL_DIR / "strategies_compressed.parquet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class StrategyGenerator:
    def __init__(self):
        self.indicators = TradingIndicators()
        self.all_indicators = self.get_all_indicators()
        self.compressed_knowledge = self.load_compressed_knowledge()
        self.market_anomalies = []
        self.generated_strategies = {}
        self.latest_market_data = None  # ✅ Aggiornato dinamicamente
        logger.info("StrategyGenerator initialized")

    def get_all_indicators(self):
        return (
            {name: method for name, method in inspect.getmembers(
                self.indicators, predicate=inspect.ismethod
            )}
        )

    def load_compressed_knowledge(self):
        if STRATEGY_FILE.exists():
            df = pl.read_parquet(STRATEGY_FILE)
            logging.info("Loaded compressed knowledge from file")
            return np.frombuffer(df["knowledge"][0], dtype=np.float32)
        else:
            logging.info("Nessun file di conoscenza esistente trovato")
            return np.zeros(100, dtype=np.float32)

    def save_compressed_knowledge(self):
        df = pl.DataFrame({"knowledge": [self.compressed_knowledge.tobytes()]})
        df.write_parquet(STRATEGY_FILE, compression="zstd", mode="overwrite")
        logging.info("Saved compressed knowledge to file")

    def detect_market_anomalies(self, market_data):
        """
        Rileva anomalie nel mercato basandosi su volatilità e volumi.
        Questo metodo analizza i dati di mercato per identificare condizioni anomale,
        come un'elevata volatilità o un improvviso picco nei volumi di scambio.
        Le anomalie rilevate vengono aggiunte alla lista `self.market_anomalies`.
        Args:
        market_data (DataFrame): Dati di mercato contenenti colonne come "volatility"
        e "volume" utilizzate per il rilevamento delle anomalie.
        """
        high_volatility = market_data["volatility"].iloc[-1] > 2.0
        sudden_volume_spike = (
            market_data["volume"].iloc[-1] >
            market_data["volume"].mean() * 3
        )
        if high_volatility or sudden_volume_spike:
            self.market_anomalies.append("Manipolazione Rilevata")
            logging.warning(
                "Rilevata anomalia di mercato: elevata volatilità o "
                "picco improvviso del volume"
            )

    def update_knowledge(self, profit, win_rate, drawdown, volatility):
        efficiency_score = (
            (profit * 0.5) + (win_rate * 0.3) -
            (drawdown * 0.1) - (volatility * 0.1)
        )
        self.compressed_knowledge = (
            np.clip(self.compressed_knowledge +
                    (efficiency_score / 1000), 0, 1)
        )
        logging.info(
            f"Knowledge updated: profit={profit}, win_rate={win_rate}, "
            f"drawdown={drawdown}, volatility={volatility}"
        )
        self.save_compressed_knowledge()

        # ✅ Compressione incrementale della conoscenza
        if len(self.compressed_knowledge) > 50:
            self.compressed_knowledge = np.mean(
                self.compressed_knowledge.reshape(-1, 2), axis=1
            )
            self.save_compressed_knowledge()
            logging.info("Compressed knowledge incrementally")

    def generate_new_strategies(self, market_data):
        """
        Genera nuove strategie di trading basate sui dati di mercato più recenti.
        Questo metodo utilizza i valori degli indicatori tecnici calcolati
        dai dati di mercato per creare nuove strategie di trading. Le strategie
        vengono salvate nel dizionario `self.generated_strategies`.
        Args:
        market_data (DataFrame): Dati di mercato utilizzati per calcolare gli 
        indicatori tecnici e generare le strategie.
        """
        indicator_values = (
            {name: func(market_data) for name,
             func in self.all_indicators.items()}
        )
        new_strategies = {
            "strategy_1": (
                indicator_values["RSI"] < 30 and
                indicator_values["MACD"] > indicator_values["MACD_Signal"] and
                indicator_values["ADX"] > 25
            ),
            "strategy_2": (
                indicator_values["RSI"] > 70 and
                indicator_values["MACD"] < indicator_values["MACD_Signal"] and
                indicator_values["BB_Upper"] > market_data["close"].iloc[-1]
            ),
            "strategy_3": (
                indicator_values["EMA_50"] > indicator_values["EMA_200"] and
                indicator_values["VWAP"] > market_data["close"].iloc[-1]
            )
        }
        self.generated_strategies.update(new_strategies)
        logging.info("Generated new strategies")

    def select_best_strategy(self, market_data):
        self.detect_market_anomalies(market_data)
        self.generate_new_strategies(market_data)
        indicator_values = (
            {name: func(market_data) for name,
             func in self.all_indicators.items()}
        )

        strategy_conditions = {
            **self.generated_strategies,
            "scalping": (
                indicator_values["RSI"] < 30 and
                indicator_values["MACD"] > indicator_values["MACD_Signal"] and
                indicator_values["ADX"] > 25 and
                self.compressed_knowledge.mean() > 0.6,
            ),
            "mean_reversion": (
                indicator_values["RSI"] > 70 and
                indicator_values["MACD"] < indicator_values["MACD_Signal"] and
                indicator_values["BB_Upper"] >
                market_data["close"].iloc[-1] and
                self.compressed_knowledge.mean() > 0.5,
            ),
            "trend_following": (
                indicator_values["EMA_50"] > indicator_values["EMA_200"] and
                indicator_values["VWAP"] > market_data["close"].iloc[-1] and
                self.compressed_knowledge.mean() > 0.4,
            ),
            "swing": (
                indicator_values["STOCH_K"] < 20 and
                indicator_values["STOCH_D"] < 20 and
                self.compressed_knowledge.mean() > 0.3,
            ),
            "momentum": (
                indicator_values["momentum"] > 100 and
                indicator_values["ADX"] > 20 and
                self.compressed_knowledge.mean() > 0.2,
            ),
            "breakout": (
                indicator_values["Donchian_Upper"] <
                market_data["high"].iloc[-1] and
                indicator_values["volatility"] > 1.5 and
                self.compressed_knowledge.mean() > 0.7,
            ),
            "ai_generated": self.compressed_knowledge.mean() > 0.75,
        }

        for strategy, condition in strategy_conditions.items():
            if condition:
                logging.info("Best strategy selected: %s", strategy)
                return strategy, self.compressed_knowledge.mean()

        logging.info("Default strategy selected")
        return "default_strategy", self.compressed_knowledge.mean()

    def fuse_top_strategies(self, top_n=5):
        """
        Combina le migliori strategie generate in una singola strategia "super".
        Questo metodo seleziona le migliori `top_n` strategie basandosi sui loro
        punteggi di performance, e le fonde in una nuova strategia chiamata
        "super_strategy". Ogni indicatore è mediato tra le strategie selezionate.
        Args:
        top_n (int): Numero di strategie migliori da considerare per la fusione. 
        Updates:
        - Aggiunge una nuova strategia "super_strategy" al dizionario 
        self.generated_strategies`.
        """
        sorted_strategies = sorted(
            self.generated_strategies.items(),
            key=lambda x: x[1].get('performance_score', 0.5),
            reverse=True
        )[:top_n]

        super_strategy = {}
        for _, strategy in sorted_strategies:
            for indicator, condition in strategy.items():
                super_strategy[indicator] = super_strategy.get(
                    indicator, 0
                ) + condition

        for indicator in super_strategy:
            super_strategy[indicator] /= top_n

        self.generated_strategies["super_strategy"] = super_strategy
        logging.info("Fused top strategies into super strategy")

    def exploit_market_anomalies(self, market_data):
        """
        Identifica e sfrutta le anomalie di mercato per generare strategie dedicate.
        Questo metodo analizza i dati di mercato per rilevare anomalie, come
        spread elevati o alta latenza. Per ogni anomalia rilevata, una strategia
        specifica viene generata e aggiunta al dizionario `self.generated_strategies`.
        Args:
        market_data (DataFrame): Dati di mercato contenenti colonne come "spread"
        e "latency" per l'analisi delle anomalie.
        """
        anomalies = []
        if market_data["spread"].iloc[-1] > market_data["spread"].mean() * 5:
            anomalies.append("Buco di Liquidità")
        if market_data["latency"].iloc[-1] > 200:
            anomalies.append("Lag nei Dati")
        for anomaly in anomalies:
            name = f"exploit_{anomaly.lower().replace(' ', '_')}"
            self.generated_strategies[name] = {"anomaly_detected": True}
            logging.warning("Exploiting market anomaly: %s", anomaly)

    def continuous_self_improvement(self, interval_seconds=1800):
        """
        Esegue un miglioramento continuo delle strategie di trading
        a intervalli regolari.
        Questo metodo genera nuove strategie,
        fonde le migliori strategie,
        sfrutta eventuali anomalie di mercato e aggiorna
        la conoscenza compressa
        basandosi su dati simulati. Viene eseguito in un loop
        continuo con una pausa
        specificata tra un ciclo e l'altro.
        Args:
        interval_seconds (int): Intervallo di tempo in secondi
        tra un ciclo e l'altro.
        """
        while True:
            if self.latest_market_data is not None:
                self.generate_new_strategies(self.latest_market_data)
                self.fuse_top_strategies()
                self.exploit_market_anomalies(self.latest_market_data)
                simulated_profit = np.random.uniform(-0.5, 1.5)
                simulated_win_rate = np.random.uniform(0.5, 0.8)
                simulated_drawdown = np.random.uniform(0, 0.1)
                simulated_volatility = np.random.uniform(0.01, 0.05)
                self.update_knowledge(
                    simulated_profit, simulated_win_rate, simulated_drawdown,
                    simulated_volatility
                )
            time.sleep(interval_seconds)

    def update_strategies(self, strategy_name, result):
        """
        Aggiorna internamente lo stato delle strategie dopo un trade.
        """
        logging.info(
            "📈 Strategia aggiornata: %s con profit: %s",
            strategy_name,
            result,
        )
        self.compressed_knowledge = np.clip(
            self.compressed_knowledge + (result / 1000), 0, 1
        )
        self.save_compressed_knowledge()


# ✅ Test rapido e avvio
if __name__ == "__main__":

    sg = StrategyGenerator()
    threading.Thread(
        target=sg.continuous_self_improvement, daemon=True
    ).start()
    logger.info(
        "📊 Conoscenza strategica caricata: %s", sg.compressed_knowledge.mean()
    )
