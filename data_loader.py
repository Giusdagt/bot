"""
data_loader.py
Gestione avanzata, ultra-intelligente e dinamica degli asset.
Supporta preset, caricamento dinamico e trading reale da config.json.
"""

import json
import os
import logging
import re
import requests

print("data_loader.py caricato ✅")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

CONFIG_FILE = "config.json"
MARKET_API_FILE = "market_data_apis.json"
PRESET_ASSETS_FILE = "preset_assets.json"
AUTO_MAPPING_FILE = "auto_symbol_mapping.json"

USE_PRESET_ASSETS = True  # True preset_assets.json, False dinamico illimitato

TRADABLE_ASSETS = {"crypto": [], "forex": [], "indices": [], "commodities": []}

SUPPORTED_CURRENCIES = [
    "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "HKD", "SGD",
    "NOK", "SEK", "DKK", "MXN", "ZAR", "TRY", "CNY", "RUB", "PLN", "HUF",
    "INR", "IDR", "VND", "THB", "KRW", "PHP"
]


def load_json_file(json_file, default=None):
    """Carica un file JSON e restituisce i dati."""
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
    """Carica il file config.json per trading reale."""
    return load_json_file(CONFIG_FILE)


def load_market_data_apis():
    """Carica configurazioni delle API di mercato."""
    return load_json_file(MARKET_API_FILE)


def load_preset_assets():
    """Carica gli asset predefiniti per trading da preset."""
    assets = load_json_file(PRESET_ASSETS_FILE)
    logging.info("✅ Asset caricati da preset_assets.json: %s", assets)
    return assets


def load_auto_symbol_mapping():
    """Carica la mappatura automatica simboli."""
    return load_json_file(AUTO_MAPPING_FILE, default={})


def save_auto_symbol_mapping(mapping):
    """Salva la mappatura automatica simboli."""
    save_json_file(mapping, AUTO_MAPPING_FILE)


def standardize_symbol(symbol, mapping):
    """Normalizza intelligentemente il simbolo."""
    if symbol in mapping:
        logging.info(
            "✅ Simbolo trovato nella mappatura: %s -> %s",
            symbol, mapping[symbol]
        )
        return mapping[symbol]

    normalized_symbol = re.sub(r"[^\w]", "", symbol).upper()
    logging.info("🔄 Simbolo normalizzato: %s -> %s",
                 symbol, normalized_symbol
    )

    for currency in SUPPORTED_CURRENCIES:
        if normalized_symbol.endswith(currency):
            base_symbol = normalized_symbol[:-len(currency)]
            normalized_symbol = f"{base_symbol}{currency}"
            break

    mapping[symbol] = normalized_symbol
    save_auto_symbol_mapping(mapping)
    return normalized_symbol


def categorize_tradable_assets(assets, mapping):
    """Categorizza automaticamente gli asset forniti."""
    try:
        for category, asset_list in assets.items():
            TRADABLE_ASSETS[category] = [
                standardize_symbol(asset, mapping) for asset in asset_list
            ]
        logging.info("✅ Asset organizzati e normalizzati con successo.")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        logging.error("❌ Errore categorizzazione asset: %s", e)


# Ultra-avanzato: priorità no-api, fallback API solo se necessario
def dynamic_assets_loading(mapping):
    """Caricamento dinamico intelligente degli asset."""
    market_data_apis = load_market_data_apis()
    assets = {"crypto": [], "forex": [], "indices": [], "commodities": []}

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
            logging.info("✅ Dati no-api da %s caricati.", source_name)
        except requests.RequestException as e:
            logging.warning("⚠️ Fonte no-api '%s' fallita: %s", source_name, e)

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
                logging.error("❌ Errore API '%s': %s", exchange["name"], e)

    categorize_tradable_assets(assets, mapping)


def exchange_asset_type(symbol):
    """Determina tipo di asset basato su simbolo."""
    if symbol.endswith(tuple(SUPPORTED_CURRENCIES)):
        return "forex"
    if symbol.startswith(("BTC", "ETH", "BNB")):
        return "crypto"
    if symbol.startswith(("XAU", "XAG", "WTI")):
        return "commodities"
    if symbol in ("US30", "NAS100", "SPX"):
        return "indices"
    return None
