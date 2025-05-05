"""
risk_management.py
Modulo definitivo per la gestione intelligente del rischio e del capitale.
Ottimizzato per IA, Deep Reinforcement Learning (DRL) e trading adattivo su
più account. Supporta trading multi-strategia, gestione avanzata del rischio
e allocazione ottimale del capitale. Configurazione dinamica tramite
config.json per automazione totale.
"""

import logging
import numpy as np
from volatility_tools import VolatilityPredictor
from data_loader import (
    load_config,
    load_auto_symbol_mapping,
    USE_PRESET_ASSETS,
    load_preset_assets
)
import data_handler

print("risk_management.py caricato ✅")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Carica configurazione globale
config = load_config()


def get_tradable_assets():
    """Restituisce gli asset per il trading in base alle impostazioni."""
    auto_mapping = load_auto_symbol_mapping()
    return sum(load_preset_assets().values(), []) if USE_PRESET_ASSETS \
        else list(auto_mapping.values())


class RiskManagement:
    """Gestisce rischio, allocazione del capitale e trailing stop"""
    def __init__(self, max_drawdown=None):
        """Starta il sistema di gestione del rischio dalla configurazione"""
        settings = config["risk_management"]
        self.risk_settings = {
            "max_drawdown": (
                max_drawdown
                if max_drawdown is not None
                else settings["max_drawdown"]
            ),
            "trailing_stop_pct": settings["trailing_stop_pct"],
            "risk_per_trade": settings["risk_per_trade"],
            "max_exposure": settings["max_exposure"]
        }
        self.balance_info = {'min': float('inf'), 'max': 0}
        self.kill_switch_activated = False
        self.volatility_predictor = VolatilityPredictor()
        self.recovery_counter = 0

    def adaptive_stop_loss(self, entry_price, symbol):
        """Calcola stop-loss e trailing-stop basati su dati normalizzati."""
        market_data = data_handler.get_normalized_market_data(symbol)
        if not market_data or "volatility" not in market_data:
            logging.warning(
                "⚠️ Dati non disponibili X %s, uso default", symbol
            )
            return entry_price * 0.95, entry_price * 0.98
        volatility = market_data["volatility"]
        stop_loss = entry_price * (1 - (volatility * 1.5))
        trailing_stop = entry_price * (1 - (volatility * 0.8))
        return stop_loss, trailing_stop

    def adjust_risk(self, symbol):
        """Adatta trailing stop e il capitale usando dati normalizzati."""
        market_data = data_handler.get_normalized_market_data(symbol)
        required_keys = ["volume", "price_change", "rsi", "bollinger_width"]
        if not market_data or any(
            key not in market_data for key in required_keys
        ):
            logging.warning(
                "⚠️ Dati incompleti X %s, resto invariato", symbol
            )
            return  # Non modifica il rischio se i dati non sono completi

        future_volatility = self.volatility_predictor.predict_volatility(
            np.array([
                [
                    market_data["volume"],
                    market_data["price_change"],
                    market_data["rsi"],
                    market_data["bollinger_width"]
                ]
            ])
        )
        atr = future_volatility[0] * 100  # Previsione volatilità futura
        if atr > 15:
            self.risk_settings["trailing_stop_pct"] = 0.15
            self.risk_settings["risk_per_trade"] = 0.01
        elif atr > 10:
            self.risk_settings["trailing_stop_pct"] = 0.1
            self.risk_settings["risk_per_trade"] = 0.015
        else:
            self.risk_settings["trailing_stop_pct"] = 0.05
            self.risk_settings["risk_per_trade"] = 0.02

    def calculate_position_size(self, balance, symbol):
        """Dimensione ottimale della posizione in base al saldo e ai dati"""
        market_data = data_handler.get_normalized_market_data(symbol)

        if balance <= 0:
            logging.warning(
                "⚠️ Saldo non valido (%s) per %s, imposta 0.",
                balance,
                symbol
            )
            return 0

        if not market_data or "momentum" not in market_data:
            logging.warning(
                "⚠️ Momentum non disponibile per %s, uso valore base.",
                symbol
            )
            momentum_factor = 1  # Default
        else:
            momentum_factor = 1 + market_data["momentum"]

        base_position_size = balance * self.risk_settings["risk_per_trade"]
        adjusted_position_size = base_position_size * momentum_factor
        max_allowed = balance * self.risk_settings["max_exposure"]
        return min(adjusted_position_size, max_allowed)
