from pathlib import Path
import textwrap

# Creazione file di test completo per AIModel e DRL System
test_script = textwrap.dedent("""
    import asyncio
    import logging
    import traceback
    from ai_model import AIModel, fetch_account_balances
    from data_handler import get_normalized_market_data, get_available_assets
    from drl_agent import DRLAgent
    from drl_super_integration import DRLSuperManager
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def run_test():
        try:
            print("🔍 TEST STARTED: Caricamento asset...")
            assets = get_available_assets()
            assert isinstance(assets, list) and assets, "❌ Lista asset vuota o non valida."

            print(f"✅ Asset disponibili: {assets[:5]}")

            print("🔍 Caricamento dati di mercato normalizzati...")
            all_market_data = {}
            for symbol in assets[:3]:  # Testa solo su 3 asset per velocità
                data = get_normalized_market_data(symbol)
                assert data is not None and not data.is_empty(), f"❌ Dati non validi per {symbol}"
                all_market_data[symbol] = data
            print("✅ Dati caricati correttamente.")

            print("🔍 Recupero bilanci account...")
            balances = fetch_account_balances()
            assert isinstance(balances, dict) and balances, "❌ Bilanci non validi"
            print(f"✅ Bilanci: {balances}")

            print("🧠 Inizializzazione AIModel...")
            ai_model = AIModel(all_market_data, balances)
            print("✅ AIModel inizializzato correttamente.")

            for symbol in all_market_data:
                print(f"🧪 Test trading decision su {symbol}...")
                asyncio.run(ai_model.decide_trade(symbol))
                print(f"✅ Completato test trade per {symbol}.")

            print("✅✅✅ TEST COMPLETO ESEGUITO CON SUCCESSO ✅✅✅")

        except Exception as e:
            print("❌ ERRORE DURANTE IL TEST:")
            traceback.print_exc()

    if __name__ == "__main__":
        run_test()
""")

# Salvataggio file
test_path = Path("c:/bot/debug_system.py")
test_path.write_text(test_script, encoding="utf-8")

test_path.name  # Mostra solo il nome del file creato per l'utente
