"""
Modulo completo per la gestione delle Delivery Zone.
Calcola in modo intelligente le zone di presa profitto in base a volumi, ILQ zone,
reazioni storiche e previsione dei prezzi.
"""
import numpy as np
import polars as pl
from data_handler import get_normalized_market_data
from smart_features import add_ilq_zone
from price_prediction import PricePredictionModel

print("delivery_zone_manager.py caricato ✅")


class DeliveryZoneManager:
    def __init__(self):
        self.price_predictor = PricePredictionModel()

    def calculate_delivery_zone(self, symbol, action, lookback=300, volume_factor=2.0):
        df = get_normalized_market_data(symbol)
        if df is None or df.height < lookback:
            return None  # Dati insufficienti

        # Applica ILQ Zone
        df = add_ilq_zone(df, volume_factor=volume_factor)
        ilq_df = df.filter(pl.col("ILQ_Zone") == 1)

        if ilq_df.is_empty():
            return None  # Nessuna zona liquida rilevata

        # Analisi dei volumi per la zona di Delivery
        high_volume_zones = ilq_df.filter(pl.col("volume") > ilq_df["volume"].mean() * volume_factor)

        if high_volume_zones.is_empty():
            return None

        # Calcola livelli di prezzo target
        if action == "buy":
            delivery_level = float(high_volume_zones["high"].max())
        else:
            delivery_level = float(high_volume_zones["low"].min())

        # Raffinamento con la previsione del prezzo
        market_data_array = df.select(df.columns).to_numpy().flatten()
        full_state = np.clip(market_data_array, -1, 1)
        predicted_price = self.price_predictor.predict_price(full_state.reshape(1, -1))[0]

        # Fusione dei livelli: 70% storico, 30% previsione AI
        final_delivery_zone = delivery_level * 0.7 + predicted_price * 0.3

        return round(final_delivery_zone, 5)

    def get_delivery_zone(self, symbol, action):
        zone = self.calculate_delivery_zone(symbol, action)
        if zone is None:
            print(f"⚠️ Nessuna Delivery Zone trovata per {symbol}")
        return zone


if __name__ == "__main__":
    dzm = DeliveryZoneManager()
    test_zone = dzm.get_delivery_zone("EURUSD", "buy")
    print(f"Delivery Zone suggerita: {test_zone}")
