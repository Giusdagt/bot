"""
data_handler.py
Modulo per la gestione dei dati di mercato, inclusa la normalizzazione e il salvataggio.
"""

import os
import shutil
import json
import logging
import asyncio
import websockets
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import data_api_module
from indicators import TradingIndicators

# 📌 Configurazione logging avanzata
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# 📌 Percorsi per backup e dati
SAVE_DIRECTORY = "/mnt/usb_trading_data/processed_data" if os.path.exists(
    "/mnt/usb_trading_data") else "D:/trading_data/processed_data"
HISTORICAL_DATA_FILE = os.path.join(SAVE_DIRECTORY, "historical_data.parquet")
SCALPING_DATA_FILE = os.path.join(SAVE_DIRECTORY, "scalping_data.parquet")
RAW_DATA_FILE = "market_data.json"
CLOUD_BACKUP = "/mnt/google_drive/trading_backup"

# 📌 WebSocket per dati in tempo reale (Scalping)
WEBSOCKET_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"

# 📌 Creazione dello scaler per normalizzazione
scaler = MinMaxScaler()


async def process_websocket_message(message):
    """Elabora il messaggio ricevuto dal WebSocket per dati real-time di scalping."""
    try:
        data = json.loads(message)
        price = float(data["p"])
        timestamp = datetime.fromtimestamp(data["T"] / 1000.0)

        df = pd.DataFrame([[timestamp, price]], columns=["timestamp", "price"])
        df.set_index("timestamp", inplace=True)

        # 📊 Calcolo indicatori tecnici
        df["rsi"] = TradingIndicators.relative_strength_index(df)
        df["macd"], df["macd_signal"] = TradingIndicators.moving_average_convergence_divergence(df)
        df["ema"] = TradingIndicators.exponential_moving_average(df)
        df["bollinger_upper"], df["bollinger_lower"] = TradingIndicators.bollinger_bands(df)

        # 📌 Normalizzazione e salvataggio dati
        df = normalize_data(df)
        save_processed_data(df, SCALPING_DATA_FILE)
        logging.info("✅ Dati scalping aggiornati: %s", df.tail(1))

    except (json.JSONDecodeError, KeyError, ValueError) as e:
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
        except Exception as e:
            logging.error("❌ Errore WebSocket: %s", e)
            await asyncio.sleep(5)
            await consume_websocket()


async def fetch_and_prepare_historical_data():
    """Scarica, elabora e normalizza i dati storici."""
    try:
        if not should_update_data(HISTORICAL_DATA_FILE):
            logging.info("✅ Dati storici già aggiornati.")
            return load_processed_data(HISTORICAL_DATA_FILE)

        logging.info("📥 Scaricamento ed elaborazione dati storici...")
        ensure_directory_exists(SAVE_DIRECTORY)

        if not os.path.exists(RAW_DATA_FILE):
            logging.warning("⚠️ File dati di mercato non trovato. Avvio recupero...")
            await data_api_module.main_fetch_all_data("eur")

        return process_historical_data()

    except Exception as e:
        logging.error("❌ Errore elaborazione dati storici: %s", e)
        return pd.DataFrame()


def process_historical_data():
    """Elabora e normalizza i dati storici."""
    try:
        with open(RAW_DATA_FILE, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        historical_data_list = []
        for crypto in raw_data:
            prices = crypto.get("historical_prices", [])
            for entry in prices:
                try:
                    timestamp = entry.get("timestamp")
                    close_price = entry.get("close")
                    if timestamp and close_price:
                        historical_data_list.append({
                            "timestamp": timestamp,
                            "coin_id": crypto.get("id", "unknown"),
                            "close": close_price
                        })
                except KeyError as e:
                    logging.error("⚠️ Errore parsing dati storici %s: %s",
                                  crypto.get('id', 'unknown'), e)

        df = pd.DataFrame(historical_data_list)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # 📊 Calcolo indicatori tecnici
        df["rsi"] = TradingIndicators.relative_strength_index(df)
        df["macd"], df["macd_signal"] = TradingIndicators.moving_average_convergence_divergence(df)
        df["ema"] = TradingIndicators.exponential_moving_average(df)
        df["bollinger_upper"], df["bollinger_lower"] = TradingIndicators.bollinger_bands(df)

        df = normalize_data(df)

        save_processed_data(df, HISTORICAL_DATA_FILE)
        logging.info("✅ Dati storici normalizzati e salvati.")
        return df

    except Exception as e:
        logging.error("❌ Errore elaborazione dati storici: %s", e)
        return pd.DataFrame()


def normalize_data(df):
    """Normalizza i dati per il trading AI."""
    try:
        cols_to_normalize = [
            "close", "rsi", "macd", "macd_signal", "ema",
            "bollinger_upper", "bollinger_lower"
        ]
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        return df
    except Exception as e:
        logging.error("❌ Errore normalizzazione dati: %s", e)
        return df


def save_processed_data(df, filename):
    """Salva i dati elaborati."""
    df.to_parquet(filename)


def ensure_directory_exists(directory):
    """Crea la directory se non esiste."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def should_update_data(filename):
    """Verifica se i dati devono essere aggiornati."""
    if not os.path.exists(filename):
        return True
    file_age = time.time() - os.path.getmtime(filename)
    return file_age > 30 * 24 * 60 * 60  # 30 giorni


def load_processed_data(filename):
    """Carica i dati elaborati se disponibili."""
    if os.path.exists(filename):
        return pd.read_parquet(filename)
    return pd.DataFrame()


def backup_file(file_path):
    """Effettua il backup del file su Google Drive e USB."""
    try:
        shutil.copy(file_path, CLOUD_BACKUP)
        shutil.copy(file_path, SAVE_DIRECTORY)
        logging.info("✅ Backup completato per %s", file_path)
    except Exception as e:
        logging.error("❌ Errore backup: %s", e)


if __name__ == "__main__":
    asyncio.run(consume_websocket())
