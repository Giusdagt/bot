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
    Gestisce le posizioni di trading aperte,
    monitorando i segnali di mercato,
    la volatilità prevista e le azioni
    suggerite da modelli di machine learning
    per applicare strategie di chiusura automatizzate.
    """
    def __init__(self):
        self.price_predictor = PricePredictionModel()
        self.volatility_predictor = VolatilityPredictor()
        self.drl_super_manager = DRLSuperManager()
        self.drl_super_manager.load_all()
        self.max_prices = {}  # Per posizioni BUY
        self.min_prices = {}  # Per posizioni SELL

    def should_close_position(
        self, pos, last_row, action, profit,
        signal_score, predicted_volatility
    ):
        """
        Determina se una posizione deve essere chiusa in base
        a segnali di mercato e condizioni.
        """
        if last_row.get("engulfing_bearish", 0) == 1 and action == "buy":
            logging.info(
                "📉 Engulfing ribassista → chiudo BUY su %s", pos.symbol
            )
            return True
        if last_row.get("engulfing_bullish", 0) == 1 and action == "sell":
            logging.info(
                "📈 Engulfing rialzista → chiudo SELL su %s",
                pos.symbol
            )
            return True
        if last_row.get("inside_bar", 0) == 1 and profit > 0:
            logging.info(
                "📦 Inside Bar rilevata → chiudo posizione in profitto %s",
                pos.symbol
            )
            return True
        if last_row.get("fakeout_up", 0) == 1 and action == "buy":
            logging.info("🧨 Fakeout UP → chiudo BUY su %s", pos.symbol)
            return True
        if last_row.get("fakeout_down", 0) == 1 and action == "sell":
            logging.info(
                "🧨 Fakeout DOWN → chiudo SELL su %s", pos.symbol
            )
            return True
        if last_row.get("volatility_squeeze", 0) == 1:
            logging.info(
                "💥 Volatility Squeeze → chiudo %s su %s",
                action.upper(), pos.symbol
            )
            return True
        if (
            profit > 0
            and signal_score < 1
            and profit * 100000 > 2 * predicted_volatility * 10000
        ):
            logging.info(
                "⚖️ Break-even → chiudo %s su %s in profitto",
                action.upper(), pos.symbol
            )
            return True
        return False

    def handle_trailing_stop(
        self, pos, action, current_price, predicted_volatility, profit
    ):
        """
        Gestisce il trailing stop per una posizione aperta.
        """
        gain = (
            current_price - pos.price_open if
            action == "buy" else pos.price_open - current_price
        )
        if action == "buy":
            trailing_sl = (
                self.max_prices[pos.ticket] - (predicted_volatility * 0.5)
            )
            if current_price < trailing_sl:
                self.close_position(pos)
                logging.info(
                    "📉 Trailing Stop %s attivato su %s | Profit: %.2f | Gain: %.5f | Volume: %.2f",
                    action.upper(), pos.symbol, profit, gain, pos.volume
                )
                return True
            if pos.sl is None or trailing_sl > pos.sl:
                self.update_trailing_stop(pos, trailing_sl)
        else:
            trailing_sl = (
                self.min_prices[pos.ticket] + (predicted_volatility * 0.5)
            )
            if current_price > trailing_sl:
                self.close_position(pos)
                logging.info(
                    "📉 Trailing Stop %s attivato su %s | Profit: %.2f | Gain: %.5f | Volume: %.2f",
                    action.upper(), pos.symbol, profit, gain, pos.volume
                )
                return True
            if pos.sl is None or trailing_sl < pos.sl:
                self.update_trailing_stop(pos, trailing_sl)
        return False

    def dynamic_stop_management(self, pos, action, current_price, predicted_volatility, profit):
        """
        Gestione dinamica di stop loss e take profit su MT5.
        Modifica i parametri in base a condizioni di mercato
        per massimizzare i profitti e ridurre le perdite.
        """
        gain = current_price - pos.price_open if action == "buy" else pos.price_open - current_price

        # Calcolo nuovi SL e TP in modo dinamico
        dynamic_sl = pos.price_open - (predicted_volatility * 0.5) if action == "buy" else pos.price_open + (predicted_volatility * 0.5)
        dynamic_tp = pos.price_open + (gain * 1.5) if action == "buy" else pos.price_open - (gain * 1.5)

        current_tp = pos.tp if pos.tp else 0
        if (action == "buy" and dynamic_tp > current_tp) or (action == "sell" and dynamic_tp < current_tp):
            self.update_take_profit(pos, dynamic_tp)

        # Protezione a Break-Even dopo un certo guadagno
        if profit > 0 and gain > predicted_volatility * 2:
            dynamic_sl = pos.price_open  # Break-even

        # Aggiornamento diretto su MT5 solo se migliora la protezione
        if (pos.sl is None or (action == "buy" and dynamic_sl > pos.sl) or (action == "sell" and dynamic_sl < pos.sl)):
            self.update_trailing_stop(pos, dynamic_sl)

    def update_take_profit(self, pos, new_tp):
        result = mt5.order_modify(
            ticket=pos.ticket,
            price=pos.price_open,
            stoplimit=0,
            sl=pos.sl,
            tp=new_tp,
            deviation=10,
            type_time=mt5.ORDER_TIME_GTC,
            type_filling=mt5.ORDER_FILLING_IOC
        )
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info("📈 TP aggiornato per %s: %.5f", pos.symbol, new_tp)
        else:
            logging.warning("⚠️ Errore aggiornamento TP su %s | Retcode: %d", pos.symbol, result.retcode)

    def pyramiding_strategy(self, symbol, action, current_price):
        """
        Aggiunge posizioni alla posizione vincente in caso di forte trend.
        """
        positions = mt5.positions_get(symbol=symbol)
        positions_open = len([p for p in positions if p.symbol == symbol])
        total_profit = sum(pos.profit for pos in positions if pos.symbol == symbol)
        base_volume = 0.01  # Volume minimo
        volume_increment = base_volume + (total_profit / 1000)  # Scala dinamicamente
        trend_strength = self.volatility_predictor.predict_volatility(np.array([[current_price]]))[0]
        if positions_open < 5 and trend_strength > 0.7:
            if trend_strength > 0.7:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume_increment,
                    "type": mt5.ORDER_BUY if action == "buy" else mt5.ORDER_SELL,
                    "price": current_price,
                    "deviation": 10,
                    "magic": 0,
                    "comment": "Pyramiding AI",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                mt5.order_send(request)

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
            entry_price = pos.price_open
            action = "buy" if pos.type == 0 else "sell"
            current_price = (
                mt5.symbol_info_tick(symbol).bid if
                action == "buy" else mt5.symbol_info_tick(symbol).ask
            )
            profit = pos.profit

            # Inizializza prezzo massimo/minimo se non presente
            if pos.ticket not in self.max_prices:
                self.max_prices[pos.ticket] = current_price
            if pos.ticket not in self.min_prices:
                self.min_prices[pos.ticket] = current_price

            # Aggiorna i prezzi massimi/minimi raggiunti
            if action == "buy" and current_price > self.max_prices[pos.ticket]:
                self.max_prices[pos.ticket] = current_price
            if (
                action == "sell" and
                current_price < self.min_prices[pos.ticket]
            ):
                self.min_prices[pos.ticket] = current_price

            # Recupero dati di mercato e segnali
            market_data = get_normalized_market_data(symbol)
            if market_data is None or market_data.height == 0:
                logging.warning(
                    "⚠️ Dati mancanti/vuoti per %s. Salto gestione posizione.",
                    symbol
                )
                continue
            market_data = apply_all_market_structure_signals(market_data)

            timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            embedding = np.concatenate(
                [get_embedding_for_symbol(symbol, tf) for tf in timeframes]
            )
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
            if profit > predicted_volatility * 2 and gain > predicted_volatility:
                self.pyramiding_strategy(symbol, action, current_price)

            # Verifica condizioni di chiusura
            if self.should_close_position(
                pos, last_row, action, profit,
                signal_score, predicted_volatility
            ):
                self.close_position(pos)
                continue

            if profit > 0:
                self.dynamic_stop_management(
                    pos, action, current_price,
                    predicted_volatility, profit
                )

            if profit > 0 and self.handle_trailing_stop(
                pos, action, current_price, predicted_volatility, profit
            ):
                continue

            if profit > predicted_volatility * 2:
                self.pyramiding_strategy(symbol, action, current_price)

            action_rl, confidence_score, algo_used = (
            self.drl_super_manager.get_best_action_and_confidence(full_state)
            )
            if self.should_close_based_on_rl(pos, action, action_rl, confidence_score):
                self.close_position(pos)
                continue

        # Valutazione e ribilanciamento del portafoglio
        self.evaluate_portfolio_risk()
        self.rebalance_portfolio()

    def update_trailing_stop(self, pos, new_sl):
        """
        Aggiorna lo stop loss direttamente su MT5
        per proteggere la posizione in corso.
        """
        result = mt5.order_modify(
            ticket=pos.ticket,
            price=pos.price_open,
            stoplimit=0,
            sl=new_sl,
            tp=pos.tp,
            deviation=10,
            type_time=mt5.ORDER_TIME_GTC,
            type_filling=mt5.ORDER_FILLING_IOC
        )
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(
                "🔁 SL aggiornato per %s: %.5f", pos.symbol, new_sl
            )
        else:
            logging.warning(
                "⚠️ Errore aggiornamento SL su %s | Retcode: %d",
                pos.symbol, result.retcode
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
                symbol, result.retcode
            )
        # Pulisce la memoria dei prezzi
        self.clear_price_memory(pos)

    def evaluate_portfolio_risk(self):
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            return

        total_profit = sum(pos.profit for pos in positions)
        total_volume = sum(pos.volume for pos in positions)
        exposure_limit = 0.05 * total_volume * 100000  # Soglia dinamica

        if abs(total_profit) > exposure_limit:
            logging.info("🚨 Stop Portafoglio attivato! Profitto: %.2f", total_profit)
            for pos in positions:
                self.close_position(pos)

    def rebalance_portfolio(self):
        positions = mt5.positions_get()
        if not positions:
            return
        
        profits = [pos.profit for pos in positions]
        if not profits:
            return
        
        max_profit_pos = max(positions, key=lambda p: p.profit)
        if max_profit_pos.profit > 2 * abs(sum(p.profit for p in positions if p != max_profit_pos)):
            self.close_position(max_profit_pos)
            logging.info("📊 Ribilanciamento: chiusa posizione in forte profitto %s", max_profit_pos.symbol)
            return
        
    def clear_price_memory(self, pos):
        if pos.ticket in self.max_prices:
            del self.max_prices[pos.ticket]
        if pos.ticket in self.min_prices:
            del self.min_prices[pos.ticket]
