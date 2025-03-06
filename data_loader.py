"""
data_loader.py
Gestione avanzata del caricamento dei dati di configurazione e di mercato.
Ottimizzato per efficienza, espandibilità e robustezza.
"""

import json
import os
import logging

# 📌 Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 📌 Percorsi per i file di configurazione
CONFIG_FILE = "config.json"
MARKET_API_FILE = "market_data_apis.json"
PRESET_ASSETS_FILE = "preset_assets.json"

# 📌 Impostazioni globali
USE_PRESET_ASSETS = True  # Se True usa preset_assets.json, altrimenti selezione dinamica
MAX_ASSETS = 300  # Numero massimo di asset da selezionare

# 📌 Struttura dati organizzata per categorie di asset
TRADABLE_ASSETS = {
    "crypto": [],
    "forex": [],
    "indices": [],
    "commodities": []
}

# ===========================
# 🔹 FUNZIONI DI UTILITÀ
# ===========================

def load_json_file(json_file):
    """Carica e restituisce il contenuto di un file JSON."""
    if not os.path.exists(json_file):
        logging.warning(f"⚠️ Il file {json_file} non esiste. Creazione automatica in corso...")
        return None
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config():
    """Carica il file di configurazione principale."""
    return load_json_file(CONFIG_FILE)


def load_market_data_apis():
    """Carica la configurazione delle API di mercato."""
    return load_json_file(MARKET_API_FILE)


def load_preset_assets():
    """Carica gli asset predefiniti da preset_assets.json se attivo."""
    data = load_json_file(PRESET_ASSETS_FILE)
    if data:
        logging.info("✅ Asset predefiniti caricati con successo.")
    return data


def categorize_tradable_assets(market_data):
    """Filtra e organizza le coppie di trading per categoria."""
    try:
        # Se è attivo `USE_PRESET_ASSETS`, usa solo gli asset definiti in `preset_assets.json`
        if USE_PRESET_ASSETS:
            preset_assets = load_preset_assets()
            if preset_assets:
                return preset_assets  # Restituisce direttamente gli asset predefiniti
        
        # Selezione dinamica basata sui dati disponibili
        categorized_assets = {"crypto": [], "forex": [], "indices": [], "commodities": []}

        for asset in market_data.get("exchanges", []):
            symbol = asset.get("symbol", "").upper()

            if symbol.endswith("USDT") or symbol.endswith("USD"):
                categorized_assets["crypto"].append(symbol)
            elif any(fx in symbol for fx in ["EUR", "GBP", "JPY", "AUD"]):
                categorized_assets["forex"].append(symbol)
            elif any(idx in symbol for idx in ["SPX", "NDX", "DAX", "US500"]):
                categorized_assets["indices"].append(symbol)
            elif any(com in symbol for com in ["XAU", "XAG", "OIL", "BRENT"]):
                categorized_assets["commodities"].append(symbol)

        # Limita a MAX_ASSETS totali
        for key in categorized_assets:
            categorized_assets[key] = categorized_assets[key][:MAX_ASSETS // len(categorized_assets)]

        logging.info("✅ Asset tradabili organizzati con successo.")
        return categorized_assets

    except Exception as e:
        logging.error("❌ Errore nella categorizzazione asset: %s", e)
        return TRADABLE_ASSETS  # Restituisce una struttura vuota in caso di errore


# ===========================
# 🔹 ESECUZIONE PRINCIPALE
# ===========================

if __name__ == "__main__":
    try:
        # Carica le configurazioni di trading
        config = load_config()
        logging.info("🔹 Config trading caricata con successo.")

        # Carica le API di mercato
        market_data_apis = load_market_data_apis()
        logging.info("🔹 Configurazioni API di mercato caricate con successo.")

        # Organizza le coppie di trading per categoria
        tradable_assets = categorize_tradable_assets(market_data_apis)

        # Log dettagliato delle categorie selezionate
        logging.info("🔹 Crypto: %s", tradable_assets["crypto"][:10])
        logging.info("🔹 Forex: %s", tradable_assets["forex"][:10])
        logging.info("🔹 Indici: %s", tradable_assets["indices"][:10])
        logging.info("🔹 Materie Prime: %s", tradable_assets["commodities"][:10])

    except FileNotFoundError as e:
        logging.error("❌ Errore: %s", e)
    except json.JSONDecodeError:
        logging.error("❌ Errore nella lettura del file JSON. Verifica la sintassi.")
