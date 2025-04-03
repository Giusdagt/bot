import polars as pl
import numpy as np


def add_candle_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggiunge colonne avanzate per analisi professionale delle candele:
    - Dimensione corpo, upper/lower wick, body_ratio, indecision, engulfing, inside bar.
    """
    body_size = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["close", "open"]].max(axis=1)
    lower_wick = df[["close", "open"]].min(axis=1) - df["low"]
    total_range = df["high"] - df["low"] + 1e-9  # evita divisione per zero

    df = df.with_columns([
        pl.Series(name="body_size", values=body_size),
        pl.Series(name="upper_wick", values=upper_wick),
        pl.Series(name="lower_wick", values=lower_wick),
        pl.Series(name="body_ratio", values=(body_size / total_range)),
        pl.Series(name="is_indecision", values=((body_size / total_range) < 0.2)),
    ])

    # Engulfing pattern
    engulfing = [
        df["open"][i] > df["close"][i] and df["close"][i + 1] > df["open"][i + 1] and df["close"][i + 1] > df["open"][i] and df["open"][i + 1] < df["close"][i]
        for i in range(len(df) - 1)
    ] + [False]
    
    df = df.with_columns([
        pl.Series(name="engulfing", values=engulfing)
    ])

    # Inside bar detection
    inside_bar = [
        df["high"][i] < df["high"][i - 1] and df["low"][i] > df["low"][i - 1]
        if i > 0 else False for i in range(len(df))
    ]
    df = df.with_columns([
        pl.Series(name="inside_bar", values=inside_bar)
    ])

    return df


def apply_all_advanced_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Applica tutte le trasformazioni avanzate al DataFrame per
    diventare un trader istituzionale:
    - Candlestick pattern
    - Body/wick ratio
    - Pattern comportamentali
    - Volatilità locale
    - Momentum
    """
    df = add_candle_features(df)

    df = df.with_columns([
        (df["close"] - df["close"].shift(1)).alias("momentum"),
        df["close"].rolling_std(window_size=5).alias("local_volatility"),
        df["volume"].rolling_mean(window_size=3).alias("avg_volume_3"),
    ])

    df = add_ilq_zone(df)

    return df

def add_ilq_zone(df: pl.DataFrame, spread_thresh=0.02, volume_factor=2.0) -> pl.DataFrame:
    """
    Aggiunge la colonna 'ILQ_Zone':
    1 se spread è basso e volume è alto → zona liquida
    0 altrimenti
    """
    avg_volume = df["volume"].mean()
    ilq = (
        (df["spread"] < spread_thresh) &
        (df["volume"] > avg_volume * volume_factor)
    ).cast(pl.Int8)

    return df.with_columns([
        ilq.alias("ILQ_Zone")
    ])
