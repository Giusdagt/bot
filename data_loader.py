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
        logging.warning(f"⚠️ Il file {json_file} non c'è, ne creo uno nuovo.")
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


def load_auto_symbol_mapping(auto_mapping_file):
    """Carica la mappatura automatica dei simboli."""
    return load_json_file(auto_mapping_file)

def save_auto_symbol_mapping(auto_symbol_mapping, auto_mapping_file):
    """Salva la mappatura automatica dei simboli."""
    save_json_file(auto_symbol_mapping, auto_mapping_file)

def standardize_symbol(symbol, auto_symbol_mapping, auto_mapping_file):
    """
    Converte automaticamente un simbolo nel formato corretto
    basandosi sulle fonti dei dati storici e broker.
    """
    if symbol in auto_symbol_mapping:
        return auto_symbol_mapping[symbol]

    # Regole di conversione automatica
    normalized_symbol = symbol.replace("/", "").replace("-", "").upper()

    if normalized_symbol.endswith("USD"):
        normalized_symbol = normalized_symbol[:-3] + "USD"

    auto_symbol_mapping[symbol] = normalized_symbol
    save_auto_symbol_mapping(auto_symbol_mapping, auto_mapping_file)
    return normalized_symbol

def categorize_tradable_assets(preset_assets, auto_symbol_mapping, auto_mapping_file):
    """Filtra e organizza le coppie di trading per categoria, con conversione automatica."""
    try:
        for category, assets in preset_assets.items():
            TRADABLE_ASSETS[category] = [standardize_symbol(asset, auto_symbol_mapping, auto_mapping_file)
                                         for asset in assets]

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
        loaded_preset_assets = load_preset_assets()
        logging.info("🔹 Asset predefiniti caricati con successo.")

        # Carica la mappatura automatica
        auto_symbol_mapping = load_auto_symbol_mapping(AUTO_MAPPING_FILE)

        # Organizza le coppie di trading per categoria con conversione automatica
        categorize_tradable_assets(loaded_preset_assets, auto_symbol_mapping, AUTO_MAPPING_FILE)

        # Log delle categorie finali
        logging.info("🔹 Crypto: %s", TRADABLE_ASSETS["crypto"][:10])
        logging.info("🔹 Forex: %s", TRADABLE_ASSETS["forex"][:10])
        logging.info("🔹 Indici: %s", TRADABLE_ASSETS["indices"][:10])
        logging.info("🔹 Materie Prime: %s", TRADABLE_ASSETS["commodities"][:10])

    except FileNotFoundError as e:
        logging.error("❌ Errore: %s", e)
    except json.JSONDecodeError:
        logging.error("❌ Errore nella lettura del file JSON. Verifica la sintassi.")
