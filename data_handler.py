"""
data_handler.py
Modulo per la gestione dei dati di mercato, inclusa la normalizzazione,
la sincronizzazione su cloud, l'elaborazione dei dati grezzi e il supporto
al trading.
"""

import os
import logging
import asyncio
import time
from datetime import datetime
import websockets
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import data_api_module
from indicators import TradingIndicators, calculate_indicators
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# 📌 Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 📌 Percorsi per sincronizzazione e dati
SAVE_DIRECTORY = (
    "/mnt/usb_trading_data/processed_data"
    if os.path.exists("/mnt/usb_trading_data") else "D:/trading_data"
)
HISTORICAL_DATA_FILE = os.path.join(SAVE_DIRECTORY,
                                    "historical_data.parquet")
SCALPING_DATA_FILE = os.path.join(SAVE_DIRECTORY,
                                  "scalping_data.parquet")
RAW_DATA_FILE = "market_data.parquet"
CLOUD_SYNC = "/mnt/google_drive/trading_sync"

# 📌 WebSocket per dati in tempo reale (Scalping)
WEBSOCKET_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"

# 📌 Autenticazione Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)
scaler = MinMaxScaler()

def upload_to_drive(filepath):
    """Sincronizza un file su Google Drive."""
    try:
        file_drive = drive.CreateFile({'title': os.path.basename(filepath)})
        file_drive.SetContentFile(filepath)
        file_drive.Upload()
        logging.info("✅ File sincronizzato su Google Drive: %s", filepath)
    except IOError as e:
        logging.error("❌ Errore sincronizzazione Google Drive: %s", e)

def load_processed_data(filename=HISTORICAL_DATA_FILE):
    """Carica i dati elaborati da un file parquet."""
    try:
        if os.path.exists(filename):
            return pd.read_parquet(filename)
        logging.warning("⚠️ Nessun file trovato: %s", filename)
        return pd.DataFrame()
    except ValueError as e:
        logging.error("❌ Errore caricamento dati: %s", e)
        return pd.DataFrame()

def normalize_data(df):
    """Normalizza i dati di mercato e garantisce tutte le colonne."""
    try:
        required_columns = [
            'coin_id', 'symbol', 'name', 'image', 'last_updated',
            'historical_prices', 'timestamp', "close", "open", "high",
            "low", "volume"
        ]
        for col in required_columns:
            if col not in df.columns:
                df[col] = None  # Assicura che tutte le colonne siano presenti
        df = calculate_indicators(df)  # Calcola tutti gli indicatori avanzati
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df
    except ValueError as e:
        logging.error("❌ Errore normalizzazione dati: %s", e)
        return df

def fetch_and_prepare_data():
    """Scarica, elabora e salva i dati di mercato."""
    try:
        logging.info("📥 Avvio elaborazione dati...")
        if not os.path.exists(RAW_DATA_FILE):
            logging.warning("⚠️ File dati di mercato non trovato.")
            data_api_module.main()
        return process_raw_data()
    except ValueError as e:
        logging.error("❌ Errore elaborazione dati: %s", e)
        return pd.DataFrame()

def process_raw_data():
    """Elabora i dati dal file Parquet e salva come file parquet elaborati."""
    try:
        df_historical = pd.read_parquet(RAW_DATA_FILE)
        df_historical.set_index("timestamp", inplace=True)
        df_historical.sort_index(inplace=True)
        df_historical = normalize_data(df_historical)
        save_processed_data(df_historical)
        return df_historical
    except (ValueError, KeyError) as e:
        logging.error("❌ Errore elaborazione dati grezzi: %s", e)
        return pd.DataFrame()

def save_processed_data(df, filename=HISTORICAL_DATA_FILE):
    """Salva i dati elaborati in formato parquet."""
    try:
        df.to_parquet(filename, index=True)
        logging.info("✅ Dati salvati in: %s", filename)
        upload_to_drive(filename)
    except ValueError as e:
        logging.error("❌ Errore salvataggio dati: %s", e)

def process_websocket_message(message):
    """Elabora il messaggio ricevuto dal WebSocket per dati real-time."""
    try:
        data = pd.DataFrame([message])
        data = calculate_indicators(data)  # Calcolo indicatori su dati WebSocket
        save_processed_data(data, SCALPING_DATA_FILE)
        logging.info("✅ Dati scalping aggiornati: %s", data.tail(1))
    except (ValueError, KeyError) as e:
        logging.error("❌ Errore elaborazione WebSocket: %s", e)

async def consume_websocket():
    """Consuma dati dal WebSocket per operazioni di scalping."""
    async with websockets.connect(WEBSOCKET_URL) as websocket:
        logging.info("✅ Connessione WebSocket stabilita.")
        try:
            async for message in websocket:
                await process_websocket_message(message)
        except websockets.ConnectionClosed:
            logging.warning("⚠️ Connessione WebSocket chiusa. Riconnessione...")
            await asyncio.sleep(5)
            await consume_websocket()
        except ValueError as e:
            logging.error("❌ Errore WebSocket: %s", e)
            await asyncio.sleep(5)
            await consume_websocket()

if __name__ == "__main__":
    logging.info("🔄 Avvio sincronizzazione dati...")
    fetch_and_prepare_data()
    asyncio.run(consume_websocket())
