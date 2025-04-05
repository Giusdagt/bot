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
    total_range = df["high"] - df["low"] + 1e-9

    df = df.with_columns([
        pl.Series(name="body_size", values=body_size),
        pl.Series(name="upper_wick", values=upper_wick),
        pl.Series(name="lower_wick", values=lower_wick),
        pl.Series(name="body_ratio", values=(body_size / total_range)),
        pl.Series(name="is_indecision", values=((body_size / total_range) < 0.2)),
    ])

    engulfing = [
        df["open"][i] > df["close"][i] and df["close"][i + 1] > df["open"][i + 1] and df["close"][i + 1] > df["open"][i] and df["open"][i + 1] < df["close"][i]
        for i in range(len(df) - 1)
    ] + [False]
    df = df.with_columns([
        pl.Series(name="engulfing", values=engulfing)
    ])

    inside_bar = [
        df["high"][i] < df["high"][i - 1] and df["low"][i] > df["low"][i - 1]
        if i > 0 else False for i in range(len(df))
    ]
    df = df.with_columns([
        pl.Series(name="inside_bar", values=inside_bar)
    ])

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


def detect_fakeouts(df: pl.DataFrame, threshold=0.5) -> pl.DataFrame:
    highs = df["high"]
    lows = df["low"]
    closes = df["close"]
    prev_highs = highs.shift(1)
    prev_lows = lows.shift(1)
    fakeout_up = ((highs > prev_highs) & (closes < prev_highs)).cast(pl.Int8)
    fakeout_down = ((lows < prev_lows) & (closes > prev_lows)).cast(pl.Int8)
    return df.with_columns([
        fakeout_up.alias("fakeout_up"),
        fakeout_down.alias("fakeout_down")
    ])


def detect_volatility_squeeze(df: pl.DataFrame, window=20, threshold=0.01) -> pl.DataFrame:
    vol = df["close"].rolling_std(window_size=window)
    squeeze = (vol < threshold).cast(pl.Int8)
    return df.with_columns([squeeze.alias("volatility_squeeze")])


def detect_micro_patterns(df: pl.DataFrame) -> pl.DataFrame:
    volume_spike = (df["volume"] > df["volume"].rolling_mean(3) * 1.5)
    price_jump = ((df["close"] - df["open"]).abs() > df["close"].rolling_std(5))
    tight_spread = (df["spread"] < 0.01)
    micro_pattern = (volume_spike & price_jump & tight_spread).cast(pl.Int8)
    return df.with_columns([micro_pattern.alias("micro_pattern_hft")])


def apply_all_market_structure_signals(df: pl.DataFrame) -> pl.DataFrame:
    df = detect_fakeouts(df)
    df = detect_volatility_squeeze(df)
    df = detect_micro_patterns(df)
    return df


def apply_all_advanced_features(df: pl.DataFrame) -> pl.DataFrame:
    df = add_candle_features(df)
    df = df.with_columns([
        (df["close"] - df["close"].shift(1)).alias("momentum"),
        df["close"].rolling_std(window_size=5).alias("local_volatility"),
        df["volume"].rolling_mean(window_size=3).alias("avg_volume_3"),
    ])
    df = add_ilq_zone(df)
    df = apply_all_market_structure_signals(df)
    
    signal_score = (
        df["ILQ_Zone"] +
        df["fakeout_up"] +
        df["fakeout_down"] +
        df["volatility_squeeze"] +
        df["micro_pattern_hft"]
    ).alias("signal_score")
    
    return df.with_columns([signal_score])
