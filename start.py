"""
start.py
Questo script rappresenta il punto di ingresso
principale per il sistema di trading automatizzato.
Include la configurazione iniziale, il caricamento dei dati,
la gestione dei modelli di intelligenza artificiale,
e l'integrazione con i componenti avanzati basati su DRL
(Deep Reinforcement Learning).
"""
import asyncio
import threading
import logging
from data_loader import (
    load_config, load_preset_assets,
    load_auto_symbol_mapping, dynamic_assets_loading, USE_PRESET_ASSETS
)
from data_handler import get_available_assets, get_normalized_market_data
from ai_model import (
    AIModel, fetch_account_balances, background_optimization_loop
)
from drl_agent import DRLSuperAgent
import subprocess

# Logging di sistema
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class TradingSystem:
    """
    Sistema principale che inizializza e coordina tutti i componenti:
    - AIModel (trading classico)
    - DRLSuperAgent (modello avanzato con PPO, DQN, A2C, SAC)
    """
    def __init__(self):
        self.config = load_config()  # Utilizzo di load_config
        logging.info(f"Configurazione caricata: {self.config}")
        self.assets = get_available_assets()
        self.market_data = {
            symbol: data for symbol in self.assets
            if (data := get_normalized_market_data(symbol)) is not None
        }
        self.balances = fetch_account_balances()
        self.ai_model = AIModel(self.market_data, self.balances)
        self.drl_super_agent = DRLSuperAgent()

    # 🔄 Caricamento dinamico o da preset
    mapping = load_auto_symbol_mapping()
    if USE_PRESET_ASSETS:
        preset_assets = load_preset_assets()
        from data_handler import save_preset_assets_from_dict
        save_preset_assets_from_dict(preset_assets)
    else:
        dynamic_assets_loading(mapping)

    def start_optimization_loop(self):
        thread = threading.Thread(
            target=background_optimization_loop,
            args=(self.ai_model,), daemon=True
        )
        thread.start()
        logging.info("🔁 Ottimizzazione AIModel avviata in background")

    async def run(self):
        """Loop principale asincrono per il trading continuo."""
        self.start_optimization_loop()
        logging.info("🚀 Sistema di trading avviato.")
        while True:
            for asset in self.ai_model.active_assets:
                await self.ai_model.decide_trade(asset)
            await asyncio.sleep(10)


subprocess.Popen(["python", "super_agent_runner.py"])

if __name__ == "__main__":
    system = TradingSystem()
    asyncio.run(system.run())
