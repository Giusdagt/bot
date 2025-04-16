import asyncio
import threading
import logging
from data_handler import get_available_assets, get_normalized_market_data
from ai_model import AIModel, fetch_account_balances, background_optimization_loop
from drl_agent import DRLSuperAgent

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
        self.assets = get_available_assets()
        self.market_data = {
            symbol: data for symbol in self.assets
            if (data := get_normalized_market_data(symbol)) is not None
        }
        self.balances = fetch_account_balances()
        self.ai_model = AIModel(self.market_data, self.balances)
        self.drl_super_agent = DRLSuperAgent()

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


if __name__ == "__main__":
    system = TradingSystem()
    asyncio.run(system.run())
