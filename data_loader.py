"""
data_loader.py
Gestione avanzata del caricamento e normalizzazione automatica degli asset.
"""

import json
import os
import logging
import re

# 📌 Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 📌 Percorsi per i file di configurazione
CONFIG_FILE = "config.json"
MARKET_API_FILE = "market_data_apis.json"
PRESET_ASSETS_FILE = "preset_assets.json"
AUTO_MAPPING_FILE = "auto_symbol_mapping.json"

# 📌 Struttura dinamica per asset
TRADABLE_ASSETS = {
    "crypto": [],
    "forex": [],
    "indices": [],
    "commodities": []
}

AUTO_SYMBOL_MAPPING = {}

# 📌 Lista delle valute principali
SUPPORTED_CURRENCIES = [
    "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "HKD", "SGD",
    "NOK", "SEK", "DKK", "MXN", "ZAR", "TRY", "CNY", "RUB", "PLN", "HUF",
    "INR", "IDR", "VND", "THB", "KRW", "PHP"
]

# ===========================
# 🔹 FUNZIONI DI UTILITÀ
# ===========================

def load_json_file(json_file):
    """Carica e restituisce il contenuto di un file JSON."""
    if not os.path.exists(json_file):
        logging.warning(f"⚠️ Il file {json_file} non esiste, ne verrà creato uno nuovo.")
        return {}
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_file(data, json_file):
    """Salva i dati in un file JSON."""
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        logging.info(f"✅ Dati salvati in {json_file}")

def load_config():
    """Carica il file di configurazione principale."""
    return load_json_file(CONFIG_FILE)

def load_market_data_apis():
    """Carica la configurazione delle API di mercato."""
    return load_json_file(MARKET_API_FILE)

def load_preset_assets():
    """Carica gli asset predefiniti per il trading."""
    return load_json_file(PRESET_ASSETS_FILE)

def load_auto_symbol_mapping():
    """Carica la mappatura automatica dei simboli."""
    global AUTO_SYMBOL_MAPPING
    AUTO_SYMBOL_MAPPING = load_json_file(AUTO_MAPPING_FILE)

def save_auto_symbol_mapping():
    """Salva la mappatura automatica dei simboli."""
    save_json_file(AUTO_SYMBOL_MAPPING, AUTO_MAPPING_FILE)

def standardize_symbol(symbol):
    """
    Converte automaticamente un simbolo nel formato corretto
    in base alla valuta e all'exchange di riferimento.
    """
    if symbol in AUTO_SYMBOL_MAPPING:
        return AUTO_SYMBOL_MAPPING[symbol]

    # Rimozione di caratteri speciali
    normalized_symbol = re.sub(r"[/\-_]", "", symbol).upper()

    # Identificazione della valuta finale
    for currency in SUPPORTED_CURRENCIES:
        if normalized_symbol.endswith(currency):
            base_symbol = normalized_symbol[:-len(currency)]
            normalized_symbol = f"{base_symbol}{currency}"
            break

    AUTO_SYMBOL_MAPPING[symbol] = normalized_symbol
    save_auto_symbol_mapping()
    return normalized_symbol

def categorize_tradable_assets(preset_assets):
    """Filtra e organizza le coppie di trading per categoria, con conversione automatica."""
    try:
        for category, assets in preset_assets.items():
            TRADABLE_ASSETS[category] = [standardize_symbol(asset) for asset in assets]

        logging.info("✅ Asset tradabili organizzati e normalizzati con successo.")
    except Exception as e:
        logging.error("❌ Errore nella categorizzazione asset: %s", e)

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

        # Carica gli asset predefiniti
        preset_assets = load_preset_assets()
        logging.info("🔹 Asset predefiniti caricati con successo.")

        # Carica la mappatura automatica
        load_auto_symbol_mapping()

        # Organizza le coppie di trading per categoria con conversione automatica
        categorize_tradable_assets(preset_assets)

        # Log delle categorie finali
        logging.info("🔹 Crypto: %s", TRADABLE_ASSETS["crypto"][:10])
        logging.info("🔹 Forex: %s", TRADABLE_ASSETS["forex"][:10])
        logging.info("🔹 Indici: %s", TRADABLE_ASSETS["indices"][:10])
        logging.info("🔹 Materie Prime: %s", TRADABLE_ASSETS["commodities"][:10])

    except FileNotFoundError as e:
        logging.error("❌ Errore: %s", e)
    except json.JSONDecodeError:
        logging.error("❌ Errore nella lettura del file JSON. Verifica la sintassi.")
