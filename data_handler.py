"""
data_handler.py
Modulo per la gestione dei dati di mercato, inclusa la normalizzazione,
la sincronizzazione su cloud, l'elaborazione dei dati grezzi e il supporto
al trading.
"""

import os
import logging
import asyncio
from datetime import datetime
import websockets
import pandas as pd
from column_definitions import required_columns
from sklearn.preprocessing import MinMaxScaler
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from indicators import calculate_indicators
import data_api_module

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
HISTORICAL_DATA_FILE = os.path.join(SAVE_DIRECTORY, "historical_data.parquet")
SCALPING_DATA_FILE = os.path.join(SAVE_DIRECTORY, "scalping_data.parquet")
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
    except Exception as e:
        logging.error("❌ Errore sincronizzazione Google Drive: %s", e)


def save_processed_data(df, filename):
    """Salva i dati elaborati in formato Parquet."""
    try:
        df.to_parquet(filename, index=False)
        logging.info("✅ Dati salvati in %s.", filename)
        upload_to_drive(filename)
    except Exception as e:
        logging.error("❌ Errore nel salvataggio dei dati: %s", e)


def process_websocket_message(message):
    """Elabora il messaggio ricevuto dal WebSocket per dati real-time."""
    try:
        data = pd.DataFrame([message])
        data["timestamp"] = datetime.utcnow()
        data = calculate_indicators(data)  # Calcola indicatori per WebSocket
        save_processed_data(data, SCALPING_DATA_FILE)  # 🔥 Salva i dati scalping!
        logging.info("✅ Dati scalping aggiornati con indicatori: %s",
                     data.tail(1))
    except Exception as e:
        logging.error("❌ Errore elaborazione WebSocket: %s", e)


async def consume_websocket():
    """Consuma dati dal WebSocket per operazioni di scalping."""
    async with websockets.connect(WEBSOCKET_URL) as websocket:
        logging.info("✅ Connessione WebSocket stabilita.")
        try:
            async for message in websocket:
                await process_websocket_message(message)
        except websockets.ConnectionClosed:
            logging.warning("⚠️ Connessione WebSocket chiusa. Riconnessione..")
            await asyncio.sleep(5)
            await consume_websocket()
        except Exception as e:
            logging.error("❌ Errore WebSocket: %s", e)
            await asyncio.sleep(5)
            await consume_websocket()


def fetch_and_prepare_data():
    """Scarica e prepara i dati di mercato se non già disponibili."""
    try:
        if not os.path.exists(RAW_DATA_FILE):
            logging.info("📥 Dati non trovati, avvio il download...")
            asyncio.run(data_api_module.main())
        logging.info("✅ Dati di mercato aggiornati.")
    except Exception as e:
        logging.error("❌ Errore durante il fetch dei dati: %s", e)


def normalize_data(df):
    """Normalizza i dati di mercato e carica le colonne richieste."""
    try:
        if df.empty:
            logging.warning("⚠️ Il DataFrame è vuoto. Nessuna normalizzazione eseguita.")
            return df

        for col in required_columns:
            if col not in df.columns:
                df[col] = pd.NA  # Usa pd.NA invece di None per valori mancanti
        
        df = calculate_indicators(df)  # Calcola gli indicatori avanzati
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.empty:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        return df
    except Exception as e:
        logging.error("❌ Errore normalizzazione dati: %s", e)
        return df


if __name__ == "__main__":
    logging.info("🔄 Avvio sincronizzazione dati...")
    fetch_and_prepare_data()
    asyncio.run(consume_websocket())
