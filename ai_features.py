"""
ai_features.py
Definizione dei set di feature AI
usati per SCALPING, SWING e MACRO trading.
Il sistema può scegliere automaticamente
il set corretto in base al tipo di strategia.
"""

print("ai_features.py caricato ✅")

AI_FEATURES_SCALPING = [
    "open", "high", "low", "close", "volume",
    "spread",
    "price_change_percentage_24h",
    "market_cap_change_percentage_24h"
]

AI_FEATURES_SWING = [
    "open", "high", "low", "close", "volume",
    "market_cap", "spread",
    "price_change_percentage_24h",
    "market_cap_change_percentage_24h",
    "ath_change_percentage",
    "atl_change_percentage"
]

AI_FEATURES_MACRO = [
    "open", "close", "volume",
    "market_cap", "price_change_percentage_24h",
    "market_cap_change_percentage_24h",
    "ath_change_percentage",
    "atl_change_percentage"
]

# Dizionario centrale per selezione automatica
FEATURE_SET_MAP = {
    "scalping": AI_FEATURES_SCALPING,
    "swing": AI_FEATURES_SWING,
    "macro": AI_FEATURES_MACRO
}

def get_features_by_strategy_type(strategy_type: str):
    """
    Restituisce l'elenco di feature
    corretto in base al tipo di strategia.
    Args:
    strategy_type (str):
    Tipo di strategia ("scalping", "swing", "macro")
    Returns:
    list[str]: Elenco di colonne da usare per IA/DRL
    """
    return FEATURE_SET_MAP.get(strategy_type.lower(), AI_FEATURES_SWING)

def get_ai_features_from_df(df, strategy_type: str):
    """
    Filtra un DataFrame mantenendo
    solo le colonne AI in base al tipo di strategia.
    Args:
    df (polars.DataFrame or pandas.DataFrame): DataFrame completo
    strategy_type (str): "scalping", "swing" o "macro"
    Returns:
    DataFrame filtrato con solo le colonne AI disponibili
    """
    features = get_features_by_strategy_type(strategy_type)
    available = [col for col in features if col in df.columns]
    return df[available]
