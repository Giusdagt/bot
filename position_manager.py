"""
Modulo per la gestione delle posizioni di trading.
Questo modulo include la classe PositionManager,
che utilizza modelli di machine learning
e segnali di mercato per monitorare e chiudere
automaticamente le posizioni di trading.
"""
import logging
import MetaTrader5 as mt5
import numpy as np
from market_fingerprint import get_embedding_for_symbol
from smart_features import apply_all_market_structure_signals
from data_handler import get_normalized_market_data
from price_prediction import PricePredictionModel
from volatility_tools import VolatilityPredictor
from drl_super_integration import DRLSuperManager

print("position_manager.py caricato ✅")


class PositionManager:
    """
    Gestisce le posizioni di trading aperte, monitorando i segnali di mercato,
    la volatilità prevista e le azioni suggerite da modelli di machine learning
    per applicare strategie di chiusura automatizzate.
    """
    def __init__(self):
        self.price_predictor = PricePredictionModel()
        self.volatility_predictor = VolatilityPredictor()
        self.drl_super_manager = DRLSuperManager()
        self.drl_super_manager.load_all()

    def monitor_open_positions(self):
        """
        Monitora le posizioni aperte e applica
        strategie di chiusura basate su segnali di mercato,
        volatilità prevista e azioni suggerite
        da un modello di reinforcement learning.
        """
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
            if market_data is None or market_data.is_empty():
                logging.warning(
                    "⚠️ Dati mancanti/vuoti per %s. Salto gestione posizione.",
                    symbol
                )
                continue
            market_data = apply_all_market_structure_signals(market_data)

            timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            embedding = np.concatenate([
                get_embedding_for_symbol(symbol, tf) for tf in timeframes
            ])
            last_row = market_data[-1]
            signal_score = int(last_row["weighted_signal_score"])
            gain = (
                current_price - entry_price if
                action == "buy" else entry_price - current_price
            )
            market_data_array = (
                market_data.select(market_data.columns).to_numpy().flatten()
            )
            full_state = (
                np.clip(np.concatenate(
                    [market_data_array, [signal_score], embedding]
                ), -1, 1)
            )
            predicted_volatility = (
                self.volatility_predictor.predict_volatility(
                    full_state.reshape(1, -1)
                )[0]
            )

            # 📉 Engulfing ribassista → chiude BUY
            if last_row.get("engulfing_bearish", 0) == 1 and action == "buy":
                self.close_position(pos)
                logging.info(
                    "📉 Engulfing ribassista → chiudo BUY su %s", symbol
                )
                continue

            # 📈 Engulfing rialzista → chiude SELL
            if last_row.get("engulfing_bullish", 0) == 1 and action == "sell":
                self.close_position(pos)
                logging.info(
                    "📈 Engulfing rialzista → chiudo SELL su %s", symbol
                )
                continue

            # 🔒 Inside Bar chiude posizione prudenzialmente se già in profitto
            if last_row.get("inside_bar", 0) == 1 and profit > 0:
                self.close_position(pos)
                logging.info(
                    "📦 Inside Bar rilevata chiudo posizione in profitto %s",
                    symbol
                )
                continue

            # 🔺 Fakeout up → chiude BUY (possibile inversione)
            if last_row.get("fakeout_up", 0) == 1 and action == "buy":
                self.close_position(pos)
                logging.info("🧨 Fakeout UP → chiudo BUY su %s", symbol)
                continue

            # 🔻 Fakeout down → chiude SELL (possibile inversione)
            if last_row.get("fakeout_down", 0) == 1 and action == "sell":
                self.close_position(pos)
                logging.info("🧨 Fakeout DOWN → chiudo SELL su %s", symbol)
                continue

            # 💥 Volatility Squeeze → chiude posizione per evitare breakout
            if last_row.get("volatility_squeeze", 0) == 1:
                self.close_position(pos)
                logging.info(
                    "💥 Volatility Squeeze → chiudo %s su %s",
                    action.upper(), symbol
                )
                continue

            # 📊 Trailing Stop dinamico se in profitto
            if profit > 0:
                highest_price = (
                    pos.price_current if action == "buy" else pos.price_current
                )
                ts_price = (
                    entry_price + (gain * 0.7)
                    if action == "buy" else entry_price - (gain * 0.7)
                )
                if (
                    (action == "buy" and current_price < ts_price)
                    or (action == "sell" and current_price > ts_price)
                ):
                    self.close_position(pos)
                    logging.info(
                        "📉 Trailing Stop attivato su %s | Profit: %.2f",
                        symbol,
                        profit
                    )
                if gain is None or entry_price is None:
                    logging.error("Valori non validi per gain o entry_price.")
                    continue

            # 🟩 Break-even intelligente se in forte profitto + segnali deboli
            if profit > 0 and signal_score < 1:
                if gain * 100000 > 2 * predicted_volatility * 10000:
                    self.close_position(pos)
                    logging.info(
                        "⚖️ Break-even → chiudo %s su %s in profitto",
                        action.upper(), symbol
                    )
                    continue

            predicted_price = (
                self.price_predictor.predict_price(symbol, full_state)
            )

            action_rl, confidence_score, algo_used = (
                self.drl_super_manager.get_best_action_and_confidence(
                    full_state
                )
            )
            logging.info(
                "Algoritmo usato: %s | Azione: %d | Confidence: %.2f",
                algo_used, action_rl, confidence_score
            )

            if action_rl == 2 and action == "buy":
                self.close_position(pos)

            elif action_rl == 1 and action == "sell":
                self.close_position(pos)

            elif confidence_score < 0.3:
                self.close_position(pos)

            # Strategia di chiusura intelligente
            trailing_stop_trigger = 0.5 * predicted_volatility * 10000

            # 📊 Trailing Stop dinamico fisico su MT5
            if profit > 0:
                if action == "buy":
                    trailing_sl = entry_price + (gain * 0.7)
                    if current_price < trailing_sl:
                        self.close_position(pos)
                        logging.info("📉 Trailing Stop attivato su %s | Profit: %.2f", symbol, profit)
                        continue
                    if trailing_sl > pos.sl:
                        self.update_trailing_stop(pos, trailing_sl)
                else:
                    trailing_sl = entry_price - (gain * 0.7)
                    if current_price > trailing_sl:
                        self.close_position(pos)
                        logging.info("📉 Trailing Stop attivato su %s | Profit: %.2f", symbol, profit)
                        continue
                    if trailing_sl < pos.sl:
                        self.update_trailing_stop(pos, trailing_sl)

    def update_trailing_stop(self, pos, new_sl):
        mt5.order_modify(
            ticket=pos.ticket,
            price=pos.price_open,
            stoplimit=0,
            sl=new_sl,
            tp=pos.tp,
            deviation=10,
            type_time=mt5.ORDER_TIME_GTC,
            type_filling=mt5.ORDER_FILLING_IOC
        )


    def close_position(self, pos):
        """
        Chiude una posizione di trading aperta.
        Argomenti:
        pos: L'oggetto posizione che contiene
        i dettagli della trade da chiudere.
        """
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
                "🔍 Posizione chiusa: %s | Volume: %.2f",
                symbol, pos.volume
            )
        else:
            logging.warning(
                "❌ Errore chiusura posizione su %s | Retcode: %d",
                symbol,
                result.retcode,
            )
