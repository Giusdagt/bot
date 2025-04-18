import MetaTrader5 as mt5
import logging
import numpy as np
from market_fingerprint import get_embedding_for_symbol
from smart_features import apply_all_market_structure_signals
from data_handler import get_normalized_market_data
from price_prediction import PricePredictionModel
from volatility_tools import VolatilityPredictor
from drl_super_integration import DRLSuperManager


class PositionManager:
    def __init__(self):
        self.price_predictor = PricePredictionModel()
        self.volatility_predictor = VolatilityPredictor()
        self.drl_super_manager = DRLSuperManager()
        self.drl_super_manager.load_all()

    def monitor_open_positions(self):
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            return

        for pos in positions:
            symbol = pos.symbol
            volume = pos.volume
            action = "buy" if pos.type == 0 else "sell"
            entry_price = pos.price_open
            current_price = (
                mt5.symbol_info_tick(symbol).bid if
                action == "buy" else
                mt5.symbol_info_tick(symbol).ask
            )
            profit = pos.profit

            # Recupero dati di mercato e segnali
            market_data = get_normalized_market_data(symbol)
            if market_data is None or market_data.height == 0:
                continue
            market_data = apply_all_market_structure_signals(market_data)

            embedding = np.concatenate([
                get_embedding_for_symbol(symbol, tf) for tf in
                ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            ])
            last_row = market_data[-1]
            signal_score = int(last_row["weighted_signal_score"])

            market_data_array = (
                market_data.select(market_data.columns).to_numpy().flatten()
            )
            full_state = (
                np.clip(np.concatenate(
                    [market_data_array, [signal_score], embedding]
                ), -1, 1)
            )

            predicted_price = (
                self.price_predictor.predict_price(symbol, full_state)
            )
            predicted_volatility = (
                self.volatility_predictor.predict_volatility(
                    full_state.reshape(1, -1)
                )[0]
            )
            action_rl, confidence_score, algo_used = (
                self.drl_super_manager.get_best_action_and_confidence(
                    full_state
                )
            )

            if action_rl == 2 and action == "buy":
                self.close_position(pos)

            elif action_rl == 1 and action == "sell":
                self.close_position(pos)

            elif confidence_score < 0.3:
                self.close_position(pos)

            # Strategia di chiusura intelligente
            trailing_stop_trigger = 0.5 * predicted_volatility * 10000
            gain = (
                current_price - entry_price if
                action == "buy" else entry_price - current_price
            )

            if profit > 0:
                if gain * 100000 > trailing_stop_trigger and signal_score < 1:
                    self.close_position(pos)
                    logging.info(
                        "🚨 EXIT | %s | Profit: %.2f | Segnali in calo" % (
                            symbol, profit
                        )
                    )
            elif profit < 0:
                if abs(profit) > 0.02 * volume * 100000:  # stop loss dinamico
                    self.close_position(pos)
                    logging.info(
                        "🚑 STOP | %s | Perd: %.2f | Prot." % (symbol, profit)
                    )
            else:
                # Se il segnale cambia direzione bruscamente
                if (
                    action == "buy" and predicted_price < current_price
                ) or (
                    action == "sell" and predicted_price > current_price
                ):
                    self.close_position(pos)
                    logging.info(
                        "📊 EXIT | %s | Profit: %.2f | inversione" % (
                            symbol, profit
                        )
                    )

    def close_position(self, pos):
        symbol = pos.symbol
        action = mt5.ORDER_SELL if pos.type == 0 else mt5.ORDER_BUY
        price = (
            mt5.symbol_info_tick(symbol).bid
            if action == mt5.ORDER_BUY else
            mt5.symbol_info_tick(symbol).ask
        )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": action,
            "price": price,
            "deviation": 10,
            "magic": 0,
            "comment": "AI Auto Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(
                f"🔍 Posizione chiusa: {symbol} | Volume: {pos.volume}"
            )
        else:
            logging.warning(
                "❌ Errore chiusura posizione su %s | Retcode: %d" % (
                    symbol, result.retcode
                )
            )
