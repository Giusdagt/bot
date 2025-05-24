import asyncio
import logging
import traceback
import polars as pl
from ai_model import AIModel, fetch_account_balances
from data_handler import get_normalized_market_data, get_available_assets
from drl_agent import DRLAgent
from drl_super_integration import DRLSuperManager
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FORZA_SEGNALI_FORTI = True

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

            # --- FORZATURA SEGNALE DOPO OGNI MODIFICA ---
            if FORZA_SEGNALI_FORTI:
                # Forza la colonna dopo eventuali modifiche interne
                modified_data = ai_model.market_data[symbol].with_columns([
                    pl.lit(10).alias("weighted_signal_score")
                ])
                print(f"👁️ Forzato signal_score per {symbol}:", modified_data["weighted_signal_score"][-5:])
                ai_model.market_data[symbol] = modified_data

                # Monkey patch: forza pattern_brain e RL
                ai_model.pattern_brain.predict_score = lambda pattern_data: 100
                ai_model.drl_super_manager.get_best_action_and_confidence = lambda full_state: (1, 0.99, "PPO")
                ai_model.risk_manager["Danny"].max_trades = 5
                ai_model.risk_manager["Giuseppe"].max_trades = 5

            asyncio.run(ai_model.decide_trade(symbol))
            print(f"✅ Completato test trade per {symbol}.")

        print("✅✅✅ TEST COMPLETO ESEGUITO CON SUCCESSO ✅✅✅")

    except Exception as e:
        print("❌ ERRORE DURANTE IL TEST:")
        traceback.print_exc()

if __name__ == "__main__":
    run_test()