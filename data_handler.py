"""
data_handler.py
Modulo per la gestione dei dati di mercato, inclusa la normalizzazione, 
il backup su cloud, l'elaborazione dei dati grezzi e il supporto al trading.
"""

import os
import shutil
import json
import logging
import asyncio
import websockets
import pandas as pd
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import data_api_module
from indicators import TradingIndicators
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# 📌 Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 📌 Percorsi per backup e dati
SAVE_DIRECTORY = (
    "/mnt/usb_trading_data/processed_data"
    if os.path.exists("/mnt/usb_trading_data") else "D:/trading_data"
)
HISTORICAL_DATA_FILE = os.path.join(SAVE_DIRECTORY, "historical_data.parquet")
SCALPING_DATA_FILE = os.path.join(SAVE_DIRECTORY, "scalping_data.parquet")
RAW_DATA_FILE = "market_data.json"
MAX_AGE = 30 * 24 * 60 * 60  # 30 giorni in secondi
CLOUD_BACKUP = "/mnt/google_drive/trading_backup"

# 📌 WebSocket per dati in tempo reale (Scalping)
WEBSOCKET_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"

# 📌 Autenticazione Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)


def upload_to_drive(filepath):
    """Carica un file su Google Drive."""
    try:
        file_drive = drive.CreateFile({'title': os.path.basename(filepath)})
        file_drive.SetContentFile(filepath)
        file_drive.Upload()
        logging.info("✅ File caricato su Google Drive: %s", filepath)
    except Exception as e:
        logging.error("❌ Errore caricamento Google Drive: %s", e)


def download_from_drive(filename, save_path):
    """Scarica un file da Google Drive."""
    try:
        file_list = drive.ListFile({'q': f"title = '{filename}'"}).GetList()
        if file_list:
            file_drive = file_list[0]
            file_drive.GetContentFile(save_path)
            logging.info("✅ File scaricato da Google Drive: %s", save_path)
    except Exception as e:
        logging.error("❌ Errore download Google Drive: %s", e)


def ensure_directory_exists(directory):
    """Crea la directory se non esiste."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def should_update_data(filename=HISTORICAL_DATA_FILE, max_age_days=1):
    """Controlla se i dati devono essere aggiornati."""
    file_path = os.path.join(SAVE_DIRECTORY, filename)
    if not os.path.exists(file_path):
        return True
    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
    return datetime.now() - file_time > timedelta(days=max_age_days)


def fetch_and_prepare_data():
    """Scarica, elabora e salva i dati di mercato."""
    try:
        if not should_update_data():
            logging.info("✅ Dati aggiornati. Carico i dati esistenti.")
            return load_processed_data()

        logging.info("📥 Avvio del processo di scaricamento ed elaborazione...")
        ensure_directory_exists(SAVE_DIRECTORY)

        if not os.path.exists(RAW_DATA_FILE):
            logging.warning("⚠️ File dati di mercato non trovato. Scaricamento in corso...")
            asyncio.run(data_api_module.main_fetch_all_data("eur"))

        return process_raw_data()
    except Exception as e:
        logging.error("❌ Errore durante il processo di dati: %s", e)
        return pd.DataFrame()


def process_raw_data():
    """Elabora i dati dal file JSON grezzo e li salva come parquet."""
    try:
        with open(RAW_DATA_FILE, "r") as json_file:
            raw_data = json.load(json_file)

        if not raw_data or not isinstance(raw_data, list):
            raise ValueError("Errore: Dati di mercato non disponibili o non validi.")

        df_historical = pd.DataFrame(
            [{"timestamp": datetime.utcfromtimestamp(entry["timestamp"] / 1000),
              "coin_id": crypto.get("id", "unknown"),
              "close": entry["close"]}
             for crypto in raw_data for entry in crypto.get("historical_prices", [])]
        )
        df_historical.set_index("timestamp", inplace=True)
        df_historical.sort_index(inplace=True)

        save_processed_data(df_historical)
        return df_historical
    except Exception as e:
        logging.error("❌ Errore elaborazione dati grezzi: %s", e)
        return pd.DataFrame()


def save_processed_data(df, filename=HISTORICAL_DATA_FILE):
    """Salva i dati elaborati in formato parquet."""
    try:
        ensure_directory_exists(SAVE_DIRECTORY)
        file_path = os.path.join(SAVE_DIRECTORY, filename)
        df.to_parquet(file_path, index=True)
        logging.info("✅ Dati salvati in: %s", file_path)
    except Exception as e:
        logging.error("❌ Errore durante il salvataggio dei dati: %s", e)


def delete_old_data():
    """Elimina i file più vecchi di MAX_AGE."""
    now = time.time()
    for filename in os.listdir(SAVE_DIRECTORY):
        file_path = os.path.join(SAVE_DIRECTORY, filename)
        if os.path.isfile(file_path) and (now - os.path.getmtime(file_path) > MAX_AGE):
            os.remove(file_path)
            logging.info("🗑 File eliminato: %s", file_path)

if __name__ == "__main__":
    logging.info("🔄 Avvio della sincronizzazione dei dati...")
    fetch_and_prepare_data()
    delete_old_data()
