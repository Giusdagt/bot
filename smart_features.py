import polars as pl
import numpy as np


def add_candle_features(df: pl.DataFrame) -> pl.DataFrame:
    body_size = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["close", "open"]].max(axis=1)
    lower_wick = df[["close", "open"]].min(axis=1) - df["low"]
    total_range = df["high"] - df["low"] + 1e-9

    df = df.with_columns([
        pl.Series("body_size", body_size),
        pl.Series("upper_wick", upper_wick),
        pl.Series("lower_wick", lower_wick),
        pl.Series("body_ratio", body_size / total_range),
        ((body_size / total_range) < 0.2).alias("is_indecision")
    ])

    engulfing = [
        df["open"][i] > df["close"][i] and
        df["close"][i + 1] > df["open"][i + 1] and
        df["close"][i + 1] > df["open"][i] and
        df["open"][i + 1] < df["close"][i]
        for i in range(len(df) - 1)
    ] + [False]

    inside_bar = [
        df["high"][i] < df["high"][i - 1] and df["low"][i] > df["low"][i - 1]
        if i > 0 else False for i in range(len(df))
    ]

    df = df.with_columns([
        pl.Series("engulfing", engulfing),
        pl.Series("inside_bar", inside_bar)
    ])

    return df


def add_ilq_zone(
    df: pl.DataFrame, spread_thresh=0.02, volume_factor=2.0
) -> pl.DataFrame:
    avg_volume = df["volume"].mean()
    ilq = (
        (
            (df["spread"] < spread_thresh)
            & (df["volume"] > avg_volume * volume_factor)
        )
        .cast(pl.Int8)
    )
    return df.with_columns([ilq.alias("ILQ_Zone")])


def detect_fakeouts(df: pl.DataFrame) -> pl.DataFrame:
    threshold = (df["high"].max() - df["low"].min()) * 0.05
    highs = df["high"]
    lows = df["low"]
    closes = df["close"]
    prev_highs = highs.shift(1)
    prev_lows = lows.shift(1)

    fakeout_up_strength = ((highs - prev_highs).clip(min=0)) / threshold
    fakeout_up = (
        (fakeout_up_strength > 1) & (closes < prev_highs)
    ).cast(pl.Int8)

    fakeout_down_strength = ((prev_lows - lows).clip(min=0)) / threshold
    fakeout_down = (
        (fakeout_down_strength > 1) & (closes > prev_lows)
    ).cast(pl.Int8)

    return df.with_columns([
        fakeout_up.alias("fakeout_up"),
        fakeout_down.alias("fakeout_down")
    ])


def detect_volatility_squeeze(
    df: pl.DataFrame, window=20, threshold=0.01
) -> pl.DataFrame:
    vol = df["close"].rolling_std(window_size=window)
    squeeze = (vol < threshold).cast(pl.Int8)
    return df.with_columns([squeeze.alias("volatility_squeeze")])


def detect_micro_patterns(df: pl.DataFrame) -> pl.DataFrame:
    volume_spike = df["volume"] > df["volume"].rolling_mean(3) * 1.5
    price_jump = (df["close"] - df["open"]).abs() > df["close"].rolling_std(5)
    tight_spread = df["spread"] < 0.01
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

    try:
        vector = extract_multi_timeframe_vector(df)
        df = df.with_columns([
            pl.Series("mtf_vector", [vector.tobytes()])
        ])
    except Exception as e:
        print("⚠️ Errore durante estrazione MTF:", e)

    # Score classico
    signal_score = (
        df["ILQ_Zone"] +
        df["fakeout_up"] +
        df["fakeout_down"] +
        df["volatility_squeeze"] +
        df["micro_pattern_hft"]
    ).alias("signal_score")

    df = df.with_columns([signal_score])

    # Score pesato dinamico (opzionale ma consigliato)
    df = compute_weighted_signal_score(df)

    return df


def extract_multi_timeframe_vector(
    df: pl.DataFrame, timeframes=("1m", "5m", "15m", "30m", "1h", "4h", "1d")
) -> np.ndarray:
    if "timeframe" not in df.columns:
        return np.zeros(len(timeframes) * 3, dtype=np.float32)

    vectors = []
    for tf in timeframes:
        tf_df = df.filter(pl.col("timeframe") == tf)
        if tf_df.is_empty():
            vectors.extend([0, 0, 0])
            continue

        closes = tf_df["close"].to_numpy()
        mean = np.mean(closes)
        std = np.std(closes)
        value_range = closes.max() - closes.min()
        vectors.extend([mean, std, value_range])

    return np.array(vectors, dtype=np.float32)


def compute_weighted_signal_score(
    df: pl.DataFrame, weights: dict = None
) -> pl.DataFrame:
    default_weights = {
        "ILQ_Zone": 1.0,
        "fakeout_up": 1.0,
        "fakeout_down": 1.0,
        "volatility_squeeze": 1.0,
        "micro_pattern_hft": 1.0
    }

    w = weights if weights else default_weights

    score = (
        df["ILQ_Zone"] * w.get("ILQ_Zone", 1.0) +
        df["fakeout_up"] * w.get("fakeout_up", 1.0) +
        df["fakeout_down"] * w.get("fakeout_down", 1.0) +
        df["volatility_squeeze"] * w.get("volatility_squeeze", 1.0) +
        df["micro_pattern_hft"] * w.get("micro_pattern_hft", 1.0)
    )

    return df.with_columns(score.alias("weighted_signal_score"))
