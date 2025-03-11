"""
data_loader.py
Gestione avanzata del caricamento e normalizzazione automatica degli asset.
Ultra-intelligente, dinamico e flessibile.
"""

import json
import os
import logging
import re
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

CONFIG_FILE = "config.json"
MARKET_API_FILE = "market_data_apis.json"
PRESET_ASSETS_FILE = "preset_assets.json"
AUTO_MAPPING_FILE = "auto_symbol_mapping.json"

USE_PRESET_ASSETS = True  # usa preset_assets.json, False dinamico senza limiti

TRADABLE_ASSETS = {"crypto": [], "forex": [], "indices": [], "commodities": []}

SUPPORTED_CURRENCIES = [
    "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "HKD", "SGD",
    "NOK", "SEK", "DKK", "MXN", "ZAR", "TRY", "CNY", "RUB", "PLN", "HUF",
    "INR", "IDR", "VND", "THB", "KRW", "PHP"
]


def load_json_file(json_file, default=None):
    """Carica e restituisce il contenuto di un file JSON."""
    if not os.path.exists(json_file):
        logging.warning("⚠️ File %s non trovato, ne creo uno nuovo", json_file)
        return {} if default is None else default
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data, json_file):
    """Salva i dati in un file JSON."""
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logging.info("✅ Dati salvati in %s", json_file)


def load_config():
    """Carica il file di configurazione principale."""
    return load_json_file(CONFIG_FILE)


def load_market_data_apis():
    """Carica la configurazione delle API/NO di mercato."""
    return load_json_file(MARKET_API_FILE)


def load_preset_assets():
    """Carica gli asset predefiniti per il trading."""
    return load_json_file(PRESET_ASSETS_FILE)


def load_auto_symbol_mapping():
    """Carica la mappatura automatica dei simboli."""
    return load_json_file(AUTO_MAPPING_FILE, default={})


def save_auto_symbol_mapping(mapping):
    """Salva la mappatura automatica dei simboli."""
    save_json_file(mapping, AUTO_MAPPING_FILE)


def standardize_symbol(symbol, mapping):
    """Converte automaticamente un simbolo nel formato corretto."""
    if symbol in mapping:
        return mapping[symbol]

    normalized_symbol = re.sub(r"[^\w]", "", symbol).upper()

    for currency in SUPPORTED_CURRENCIES:
        if normalized_symbol.endswith(currency):
            base_symbol = normalized_symbol[:-len(currency)]
            normalized_symbol = f"{base_symbol}{currency}"
            break

    mapping[symbol] = normalized_symbol
    save_auto_symbol_mapping(mapping)
    return normalized_symbol


def categorize_tradable_assets(assets, mapping):
    """Organizza le coppie di trading per categoria automaticamente."""
    try:
        for category, asset_list in assets.items():
            TRADABLE_ASSETS[category] = [
                standardize_symbol(asset, mapping) for asset in asset_list
            ]
        logging.info("✅ Asset organizzati e normalizzati con successo.")
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
        logging.error("❌ Errore nella categorizzazione asset: %s", e)


# Ultra-intelligente: prioritariamente no-api, fallback API solo se necessario
def dynamic_assets_loading(mapping):
    """Carica dinamicamente gli asset da fonti senza API prioritariamente."""
    market_data_apis = load_market_data_apis()
    assets = {"crypto": [], "forex": [], "indices": [], "commodities": []}

    # Usa prioritariamente fonti senza API
    no_api_sources = market_data_apis.get("data_sources", {}).get("no_api", {})
    for source_name, base_url in no_api_sources.items():
        try:
            response = requests.get(base_url)
            response.raise_for_status()
            data = response.json()
            for item in data:
                symbol = standardize_symbol(item["symbol"], mapping)
                asset_type = exchange_asset_type(symbol)
                if asset_type:
                    assets[asset_type].append(symbol)
            logging.info("✅ Dati senza API da %s caricati.", source_name)
        except requests.RequestException as e:
            logging.warning("⚠️ Fonte no-api '%s' fallita: %s", source_name, e)

    # Se nessun asset recuperato senza API, usa fonti API
    if not any(assets.values()):
        for exchange in market_data_apis["exchanges"]:
            try:
                api_url = exchange["api_url"].replace("{currency}", "USD")
                response = requests.get(api_url)
                response.raise_for_status()
                data = response.json()
                for item in data:
                    symbol = standardize_symbol(item["symbol"], mapping)
                    asset_type = exchange_asset_type(symbol)
                    if asset_type:
                        assets[asset_type].append(symbol)
                logging.info("✅ Dati API da %s caricati.", exchange["name"])
            except requests.RequestException as e:
                logging.error(
                    "❌ Errore caricamento API '%s': %s", exchange["name"], e)

    categorize_tradable_assets(assets, mapping)


def exchange_asset_type(symbol):
    """Determina il tipo di asset in base al simbolo."""
    if symbol.endswith(tuple(SUPPORTED_CURRENCIES)):
        return "forex"
    if symbol.startswith(("BTC", "ETH", "BNB")):
        return "crypto"
    if symbol.startswith(("XAU", "XAG", "WTI")):
        return "commodities"
    if symbol in ("US30", "NAS100", "SPX"):
        return "indices"
    return None


if __name__ == "__main__":
    auto_symbol_mapping = load_auto_symbol_mapping()
    if USE_PRESET_ASSETS:
        preset_assets = load_preset_assets()
        categorize_tradable_assets(preset_assets, auto_symbol_mapping)
    else:
        dynamic_assets_loading(auto_symbol_mapping)

    logging.info("🔹 Crypto: %s", TRADABLE_ASSETS["crypto"][:10])
    logging.info("🔹 Forex: %s", TRADABLE_ASSETS["forex"][:10])
    logging.info("🔹 Indici: %s", TRADABLE_ASSETS["indices"][:10])
    logging.info("🔹 Materie Prime: %s", TRADABLE_ASSETS["commodities"][:10])
