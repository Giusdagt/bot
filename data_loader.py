"""
data_loader.py
Modulo per la gestione del caricamento dei dati di configurazione e di mercato.
"""

import json
import os
import logging

# Configurazione logging avanzato
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 📌 Percorsi per i file di configurazione
CONFIG_FILE = "config.json"
MARKET_API_FILE = "market_data_apis.json"

# ===========================
# 🔹 FUNZIONI DI UTILITÀ
# ===========================


def load_config(json_file=CONFIG_FILE):
    """Carica il file di configurazione."""
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"❌ Il file {json_file} non esiste.")
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_market_data_apis(json_file=MARKET_API_FILE):
    """Carica la configurazione delle API di mercato."""
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"❌ Il file {json_file} non esiste.")
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================
# 🔹 ESEMPIO DI UTILIZZO
# ===========================


if __name__ == "__main__":
    try:
        # Carica le configurazioni di trading
        config = load_config()
        logging.info("🔹 Config trading caricata con successo.")

        # Carica le API di mercato
        market_data_apis = load_market_data_apis()
        logging.info("🔹 Configurazioni API di mercato caricate con successo.")

    except FileNotFoundError as e:
        logging.error("❌ Errore: %s", e)
