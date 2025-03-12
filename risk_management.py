"""
risk_management.py
Modulo definitivo per la gestione intelligente del rischio e del capitale.
Ottimizzato per IA, Deep Reinforcement Learning (DRL) e trading adattivo su più account.
Supporta trading multi-strategia, gestione avanzata del rischio e allocazione ottimale del capitale.
Configurazione dinamica tramite config.json per automazione totale.
"""

import logging
import numpy as np
from data_loader import (
    load_config,
    load_auto_symbol_mapping,
    USE_PRESET_ASSETS,
    load_preset_assets
)
from ai_model import VolatilityPredictor
import data_handler  # Importa i dati normalizzati per ottimizzare la gestione del rischio

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Carica configurazione globale
config = load_config()


def get_tradable_assets():
    """Restituisce gli asset disponibili per il trading in base alle impostazioni."""
    auto_mapping = load_auto_symbol_mapping()
    return sum(load_preset_assets().values(), []) if USE_PRESET_ASSETS \
        else list(auto_mapping.values())


class RiskManagement:
    """Gestisce il rischio, l'allocazione del capitale e il trailing stop avanzato."""
    
    def __init__(self):
        """Inizializza il sistema di gestione del rischio basato su configurazione."""
        settings = config["risk_management"]
        self.max_drawdown = settings["max_drawdown"]
        self.trailing_stop_pct = settings["trailing_stop_pct"]
        self.risk_per_trade = settings["risk_per_trade"]
        self.max_exposure = settings["max_exposure"]
        self.min_balance = float('inf')
        self.highest_balance = 0
        self.kill_switch_activated = False
        self.volatility_predictor = VolatilityPredictor()
        self.recovery_counter = 0

    def adaptive_stop_loss(self, entry_price, symbol):
        """Calcola stop-loss e trailing-stop basati su dati normalizzati."""
        market_data = data_handler.get_normalized_market_data(symbol)  # Ottiene dati normalizzati
        volatility = market_data["volatility"]
        stop_loss = entry_price * (1 - (volatility * 1.5))
        trailing_stop = entry_price * (1 - (volatility * 0.8))
        return stop_loss, trailing_stop

    def adjust_risk(self, symbol):
        """Adatta dinamicamente il trailing stop e il capitale usando dati normalizzati."""
        market_data = data_handler.get_normalized_market_data(symbol)  # Usa dati normalizzati

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
            self.trailing_stop_pct = 0.15
            self.risk_per_trade = 0.01
        elif atr > 10:
            self.trailing_stop_pct = 0.1
            self.risk_per_trade = 0.015
        else:
            self.trailing_stop_pct = 0.05
            self.risk_per_trade = 0.02

    def calculate_position_size(self, balance, symbol):
        """Determina la dimensione ottimale della posizione in base al saldo e ai dati normalizzati."""
        market_data = data_handler.get_normalized_market_data(symbol)  # Usa dati normalizzati
        base_position_size = balance * self.risk_per_trade
        adjusted_position_size = base_position_size * (1 + market_data["momentum"])
        max_allowed = balance * self.max_exposure
        return min(adjusted_position_size, max_allowed)
