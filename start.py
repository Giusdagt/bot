import logging
import asyncio
from data_loader import load_config
from data_handler import get_normalized_market_data
from ai_model import AIModel
from risk_management import fetch_account_balances

# Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def main():
    logging.info("🚀 Avvio del bot AI...")
    # Caricamento configurazione
    config = load_config()
    market_data = get_normalized_market_data()
    balances = fetch_account_balances()
    market_condition = config.get("market_condition", "normal")
    # Inizializzazione AI Model
    ai_model = AIModel(market_data, balances, market_condition)
    # Avvio del trading
    asyncio.run(ai_model.decide_trade("EURUSD"))


if __name__ == "__main__":
    main()
