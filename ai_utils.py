"""
ai_utils.py
"""
from data_handler import (
    get_normalized_market_data, get_available_assets
)
from ai_model import AIModel, fetch_account_balances

print("ai_utils.py caricato ✅")


def prepare_ai_model():
    """
    Prepara il modello AI recuperando i bilanci dell'account,
    gli asset disponibili e i relativi dati di mercato normalizzati.
    Restituisce:
    tuple: Una tupla contenente l'istanza di AIModel e
    il dizionario dei dati di mercato.
    """
    balances = fetch_account_balances()
    all_assets = get_available_assets()
    market_data = {
        symbol: get_normalized_market_data(symbol)
        for symbol in all_assets
    }
    return AIModel(market_data, balances), market_data


if __name__ == "__main__":
    try:
        thread = threading.Thread(
            target=auto_train_super_agent,
            daemon=True
        )
        thread.start()
        thread.join()  # opzionale: mantiene vivo il main thread
    except KeyboardInterrupt:
        logging.info("🛑 Interrotto manualmente.")
    except (
        ValueError, KeyError, IndexError, AttributeError, TypeError, RuntimeError
    ) as e:
        logging.error("❌ Errore nel thread principale: %s", e)
