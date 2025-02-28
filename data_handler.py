# data_handler normalizzazione dati 
import os
import json
import logging
import asyncio
import websockets  
import shutil
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import data_api_module
from indicators import TradingIndicators

# 📌 Configurazioni di salvataggio e backup
SAVE_DIR = "/mnt/usb_trading_data/processed" if os.path.exists(
    "/mnt/usb_trading_data") else "D:/trading_data/processed"
HIST_FILE = os.path.join(SAVE_DIR, "historical_data.parquet")
SCALP_FILE = os.path.join(SAVE_DIR, "scalping_data.parquet")
RAW_FILE = "market_data.json"
CLOUD_BACKUP = "/mnt/google_drive/trading_backup/"
MAX_AGE = 30 * 24 * 60 * 60  # 30 giorni in secondi

# WebSocket URL per dati in tempo reale per scalping (BTC/EUR come esempio)
WEBSOCKET_URL = "wss://stream.binance.com:9443/ws/btceur@trade"

# 📌 Configurazione logging avanzato
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

scaler = MinMaxScaler()


async def fetch_market_data():
    """Scarica i dati se non sono disponibili."""
    if not os.path.exists(RAW_FILE):
        logging.warning("⚠️ File dati mercato assente, avvio recupero dati.")
        await data_api_module.main_fetch_all_data("eur")


async def consume_websocket():
    """Consuma i dati WebSocket per il trading in tempo reale (scalping)."""
    async with websockets.connect(WEBSOCKET_URL) as websocket:
        logging.info("✅ Connessione WebSocket stabilita per dati real-time.")
        try:
            async for message in websocket:
                await process_websocket_message(message)
        except websockets.ConnectionClosed:
            logging.warning(
                "⚠️ Connessione WebSocket chiusa. Riconnessione in corso..."
            )
            await asyncio.sleep(5)
            await consume_websocket()
        except Exception as e:
            logging.error(f"❌ Errore durante la ricezione WebSocket: {e}")
            await asyncio.sleep(5)
            await consume_websocket()


async def process_websocket_message(message):
    """Elabora i messaggi ricevuti dal WebSocket."""
    try:
        data = json.loads(message)
        price = float(data["p"])  # Prezzo dell'ultima operazione
        timestamp = datetime.fromtimestamp(data["T"] / 1000.0)

        df = pd.DataFrame([[timestamp, price]], columns=["timestamp", "price"])
        df.set_index("timestamp", inplace=True)

        # Calcolo indicatori tecnici per scalping
        df["rsi"] = TradingIndicators.relative_strength_index(df)
        df["macd"], df["macd_signal"] = (
            TradingIndicators.moving_average_convergence_divergence(df))
        df["ema"] = TradingIndicators.exponential_moving_average(df)
        df["bollinger_upper"], df["bollinger_lower"] = (
            TradingIndicators.bollinger_bands(df))

        # Normalizzazione dei dati
        df = normalize_data(df)

        # Salvataggio dati scalping
        save_data(df, SCALP_FILE)
        logging.info(f"✅ Dati scalping aggiornati: {df.tail(1)}")

    except Exception as e:
        logging.error(
            f"❌ Errore nell'elaborazione del messaggio WebSocket: {e}"
        )


def normalize_data(df):
    """Normalizza i dati per l'AI."""
    try:
        cols = ["close", "open", "high", "low", "volume", "rsi",
                "macd", "macd_signal", "ema", "bollinger_upper",
                "bollinger_lower"]
        df[cols] = scaler.fit_transform(df[cols])
        return df
    except Exception as e:
        logging.error(f"❌ Errore normalizzazione dati: {e}")
        return df


def process_historical_data():
    """Processa e normalizza i dati storici."""
    try:
        with open(RAW_FILE, "r") as file:
            raw_data = json.load(file)

        hist_list = []
        for crypto in raw_data:
            for entry in crypto.get("historical_prices", []):
                try:
                    hist_list.append({
                        "timestamp": entry["timestamp"],
                        "coin_id": crypto.get("id", "unknown"),
                        "close": entry["close"],
                        "open": entry["open"],
                        "high": entry["high"],
                        "low": entry["low"],
                        "volume": entry["volume"],
                    })
                except Exception as e:
                    logging.error(f"⚠️ Errore parsing dati: {e}")

        df = pd.DataFrame(hist_list)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        df["rsi"] = TradingIndicators.relative_strength_index(df)
        df["macd"], df["macd_signal"] = (
            TradingIndicators.moving_average_convergence_divergence(df))
        df["ema"] = TradingIndicators.exponential_moving_average(df)
        df["bollinger_upper"], df["bollinger_lower"] = (
            TradingIndicators.bollinger_bands(df))

        df = normalize_data(df)

        save_data(df, HIST_FILE)
        logging.info(f"✅ Dati storici salvati: {HIST_FILE}")
        return df

    except Exception as e:
        logging.error(f"❌ Errore elaborazione storici: {e}")
        return pd.DataFrame()


def save_data(df, filename):
    """Salva i dati in formato parquet e gestisce il backup."""
    df.to_parquet(filename)
    backup_file(filename)


def backup_file(file_path):
    """Gestisce il backup locale e su Google Drive."""
    try:
        shutil.copy(file_path, CLOUD_BACKUP)
        logging.info(f"☁️ Backup su Google Drive: {file_path}")
    except Exception as e:
        logging.error(f"❌ Errore backup cloud: {e}")


if __name__ == "__main__":
    asyncio.run(fetch_market_data())
    asyncio.run(consume_websocket())  # ✅ Avvia il WebSocket per scalping
    process_historical_data()
