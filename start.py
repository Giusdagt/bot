"""
start.py
File principale di avvio per il trading system completo.
Inizializza tutto, carica i dati,
avvia il bot AI, il PositionManager,
il ciclo di ottimizzazione e il super agent runner.
"""

import asyncio
import threading
import logging
import subprocess
import time
import contextlib
from data_loader import (
    load_config, load_preset_assets,
    load_auto_symbol_mapping, dynamic_assets_loading,
    USE_PRESET_ASSETS
)
from data_handler import (
    get_available_assets, get_normalized_market_data
)
from ai_model import (
    AIModel, fetch_account_balances, background_optimization_loop
)
from position_manager import PositionManager

# Configurazione logging globale
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class TradingSystem:
    """
    Sistema principale che avvia:
    - AIModel (trading decisionale)
    - PositionManager (gestione posizioni aperte)
    - Super agent runner (DRLSuperManager continuo)
    """
    def __init__(self):
        self.config = load_config()
        logging.info("✅ Configurazione caricata: %s", self.config)

        mapping = load_auto_symbol_mapping()
        if USE_PRESET_ASSETS:
            preset_assets = load_preset_assets()
        else:
            dynamic_assets_loading(mapping)

        self.assets = get_available_assets()
        self.market_data = {
            symbol: data for symbol in self.assets
            if (data := get_normalized_market_data(symbol)) is not None
        }
        self.balances = fetch_account_balances()

        self.ai_model = AIModel(self.market_data, self.balances)
        self.position_manager = PositionManager()

    def start_background_tasks(self):
        """
        Avvia i task in background:
        - Ottimizzazione continua del modello AI.
        - Monitoraggio delle posizioni aperte.
        - Esecuzione del Super Agent Runner.
        """
        threading.Thread(
            target=background_optimization_loop,
            args=(self.ai_model,), daemon=True
        ).start()
        logging.info("🔁 Ottimizzazione AI Model avviata.")

        threading.Thread(
            target=self.monitor_positions_loop, daemon=True
        ).start()
        logging.info("🛡️ Monitoraggio posizioni attivo.")

        with contextlib.suppress(Exception):
            subprocess.Popen(["python", "super_agent_runner.py"])
        logging.info("🚀 Super Agent Runner avviato.")

    def monitor_positions_loop(self):
        """
        Monitora continuamente le posizioni aperte
        e aggiorna lo stato.
        """
        while True:
            self.position_manager.monitor_open_positions()
            time.sleep(10)

    async def run(self):
        """
        Loop principale per decidere i trade su asset attivi.
        """
        self.start_background_tasks()
        logging.info(
            "🏁 Trading system avviato. Monitoraggio asset in corso..."
        )

        while True:
            for asset in self.ai_model.active_assets:
                await self.ai_model.decide_trade(asset)
            await asyncio.sleep(10)


if __name__ == "__main__":
    system = TradingSystem()
    asyncio.run(system.run())
