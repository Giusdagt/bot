"""
ai_utils.py
"""
from data_handler import (
    get_normalized_market_data, get_available_assets
)
from ai_model import AIModel, fetch_account_balances

def prepare_ai_model():
    balances = fetch_account_balances()
    all_assets = get_available_assets()
    market_data = {
        symbol: get_normalized_market_data(symbol)
        for symbol in all_assets
    }
    return AIModel(market_data, balances), market_data
