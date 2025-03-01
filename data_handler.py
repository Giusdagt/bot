"""
data_handler.py
Modulo per la gestione dei dati di mercato, inclusa la normalizzazione e 
il salvataggio su vari dispositivi (locale, USB, Google Drive).
"""

import os
import shutil
import json
import logging
import asyncio
import websockets
import pandas as pd
import time
from datetime import datetime
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
HISTORICAL_DATA_FILE = os.path.join(
    SAVE_DIRECTORY, "historical_data.parquet"
)
SCALPING_DATA_FILE = os.path.join(
    SAVE_DIRECTORY, "scalping_data.parquet"
)
RAW_DATA_FILE = "market_data.json"
CLOUD_BACKUP = "/mnt/google_drive/trading_backup"

# 📌 WebSocket per dati in tempo reale (Scalping)
WEBSOCKET_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"

# 📌 Creazione dello scaler per normalizzazione
scaler = MinMaxScaler()

# 📌 Autenticazione Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

def upload_to_drive(filepath):
    """Carica un file su Google Drive."""
    try:
        file_drive = drive.CreateFile({
            'title': os.path.basename(filepath)
        })
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

def sync_with_drive(local_path, drive_filename):
    """Sincronizza i file tra il sistema locale e Google Drive."""
    if os.path.exists(local_path):
        upload_to_drive(local_path)
    else:
        download_from_drive(drive_filename, local_path)

def move_file(src, dest):
    """Sposta un file da src a dest."""
    try:
        shutil.move(src, dest)
        logging.info("✅ File spostato da %s a %s", src, dest)
    except Exception as e:
        logging.error("❌ Errore nello spostamento del file: %s", e)

def copy_file(src, dest):
    """Copia un file da src a dest."""
    try:
        shutil.copy(src, dest)
        logging.info("✅ File copiato da %s a %s", src, dest)
    except Exception as e:
        logging.error("❌ Errore nella copia del file: %s", e)

def delete_file(filepath):
    """Elimina un file specificato."""
    try:
        os.remove(filepath)
        logging.info("✅ File eliminato: %s", filepath)
    except Exception as e:
        logging.error("❌ Errore nell'eliminazione del file: %s", e)

def delete_directory(directory):
    """Elimina una directory e tutto il suo contenuto."""
    try:
        shutil.rmtree(directory)
        logging.info("✅ Directory eliminata: %s", directory)
    except Exception as e:
        logging.error("❌ Errore nell'eliminazione della directory: %s", e)

sync_with_drive(HISTORICAL_DATA_FILE, "historical_data.parquet")
sync_with_drive(SCALPING_DATA_FILE, "scalping_data.parquet")

async def consume_websocket():
    """Consuma dati dal WebSocket per operazioni di scalping."""
    while True:
        try:
            async with websockets.connect(WEBSOCKET_URL) as websocket:
                logging.info("✅ Connessione WebSocket stabilita.")
                async for message in websocket:
                    await process_websocket_message(message)
        except Exception as e:
            logging.error("❌ Errore WebSocket: %s", e)
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(consume_websocket())
