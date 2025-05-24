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
from smart_features import apply_all_market_structure_signals, get_ilq_zone_for_symbol
from data_handler import get_normalized_market_data
from smc_manager import SMCManager
from delivery_zone_manager import DeliveryZoneManager
from market_event_manager import MarketEventManager
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
        self.smc_manager = SMCManager()
        self.delivery_zone_manager = DeliveryZoneManager()
        self.market_event_manager = MarketEventManager()
        self.drl_super_manager = DRLSuperManager()
        self.drl_super_manager.load_all()
        self.max_prices = {}  # Per posizioni BUY
        self.min_prices = {}  # Per posizioni SELL

    def calculate_dynamic_lot_size(self, account_info, risk_per_trade, stop_loss_pips, symbol):
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info or stop_loss_pips == 0:
            return 0.01  # Lotto minimo di sicurezza
        
        # Calcolo del valore per pip
        contract_size = symbol_info.trade_contract_size
        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value
        pip_value = tick_value / tick_size

        # Calcolo della perdita massima accettabile
        max_loss = account_info.balance * risk_per_trade

        # Calcolo del lotto dinamico
        lot_size = max_loss / (stop_loss_pips * pip_value * contract_size)
        return np.clip(lot_size, 0.01, symbol_info.volume_max)

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
    
    def should_close_based_on_rl(self, pos, action, action_rl, confidence_score):
        """
        Chiude la posizione se l'azione suggerita
        dall'AI è opposta a quella attuale
        e la confidenza è alta.
        """
        if confidence_score >= 0.7 and (
            (action_rl == "sell" and action == "buy") or
            (action_rl == "buy" and action == "sell")
        ):
            logging.info(
                "🤖 RL suggerisce chiusura di %s su %s con alta confidenza (%.2f)",
                action.upper(), pos.symbol, confidence_score
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
                self.update_trailing_stop(pos, trailing_sl, predicted_volatility)
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
                self.update_trailing_stop(pos, trailing_sl, predicted_volatility)
        return False

    def dynamic_stop_management(self, pos, action, current_price, predicted_volatility, profit, predicted_price, success_probability):
        """
        Gestione dinamica di stop loss e take profit su MT5.
        Modifica i parametri in base a condizioni di mercato
        per massimizzare i profitti e ridurre le perdite.
        """
        gain = current_price - pos.price_open if action == "buy" else pos.price_open - current_price

        volatility_factor = np.clip(predicted_volatility / 100, 0.1, 1.0)
        profit_factor = np.clip(profit / 100, 0.5, 2.0)
        dynamic_sl = pos.price_open - (predicted_volatility * volatility_factor) if action == "buy" else pos.price_open + (predicted_volatility * volatility_factor)
        dynamic_tp = pos.price_open + (gain * profit_factor) if action == "buy" else pos.price_open - (gain * profit_factor)
        if (action == "buy" and (pos.tp is None or dynamic_tp > pos.tp)) or \
            (action == "sell" and (pos.tp is None or dynamic_tp < pos.tp)):
            self.update_take_profit(pos, dynamic_tp, predicted_volatility)
             # Sposta il TP più lontano per massimizzare il guadagno
            dynamic_tp = predicted_price
            self.update_take_profit(pos, dynamic_tp, predicted_volatility)

        current_tp = pos.tp if pos.tp else 0
        self.update_trailing_take_profit(pos, action, current_price, gain)

        # Protezione a Break-Even dopo un certo guadagno
        breakeven_trigger = np.clip(predicted_volatility * 1.5, 0.01, 0.05) # Varia da 1% a 5% del prezzo
        if profit > 0 and gain > breakeven_trigger:
            dynamic_sl = pos.price_open  # Break-even attivato in modo intelligente

        # Aggiornamento diretto su MT5 solo se migliora la protezione
        if (pos.sl is None or (action == "buy" and dynamic_sl > pos.sl) or (action == "sell" and dynamic_sl < pos.sl)):
            self.update_trailing_stop(pos, dynamic_sl, predicted_volatility)

        # Se la probabilità di successo è alta, permetti un TP più ambizioso
        if success_probability >= 0.7:
            dynamic_tp *= 1.2  # Allunga l'obiettivo del TP
            self.update_take_profit(pos, dynamic_tp, predicted_volatility)

        if success_probability >= 0.8:
            dynamic_sl = current_price - predicted_volatility * 0.3 if action == "buy" else current_price + predicted_volatility * 0.3
            self.update_trailing_stop(pos, dynamic_sl, predicted_volatility)

    def adjust_sl_tp_based_on_ilq(self, pos, market_data):
        ilq_levels = market_data.filter(market_data["ILQ_Zone"] == 1)["close"].to_numpy()
        if len(ilq_levels) == 0:
            return  # Nessuna zona ILQ rilevata
            
        current_price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask
        if pos.type == 0:  # BUY
            closest_ilq = min(ilq_levels, key=lambda x: abs(x - current_price))
            new_sl = min(ilq_levels)
            new_tp = max(ilq_levels)
        else: # SELL
            closest_ilq = max(ilq_levels, key=lambda x: abs(x - current_price))
            new_sl = max(ilq_levels)
            new_tp = min(ilq_levels)

        self.update_trailing_stop(pos, new_sl)
        self.update_take_profit(pos, new_tp)

    def adjust_sl_tp_smart_money(self, pos, symbol, action):
        ilq_zone = get_ilq_zone_for_symbol(symbol)
        if not ilq_zone:
            return  # Nessuna zona ILQ rilevata
        current_price = mt5.symbol_info_tick(symbol).bid if action == "buy" else mt5.symbol_info_tick(symbol).ask

        if action == "buy":
            sl = ilq_zone['support'] * 0.99  # SL 1% sotto il supporto ILQ
            tp = self.smc_manager.get_delivery_zone(symbol, action)
            if tp is None:
                logging.warning(f"⚠️ Nessuna Delivery Zone trovata per {symbol}. TP non aggiornato.")
                return  # Evita di aggiornare il TP se non è disponibile
        else:
            sl = ilq_zone['resistance'] * 1.01  # SL 1% sopra la resistenza ILQ
            tp = self.smc_manager.get_delivery_zone(symbol, action)
            if tp is None:
                tp = current_price * (1.02 if action == "buy" else 0.98)  # Fallback con un margine fisso
                logging.info(f"⚠️ Delivery Zone non trovata per {symbol}. Imposto fallback TP a {tp:.5f}.")

        if pos.sl is None or (action == "buy" and sl > pos.sl) or (action == "sell" and sl < pos.sl):
            self.update_trailing_stop(pos, sl)
        if pos.tp is None or (action == "buy" and tp < pos.tp) or (action == "sell" and tp > pos.tp):
            self.update_take_profit(pos, tp)

    def set_initial_sl_tp(self, symbol, action, entry_price):
        sl, tp = self.smc_manager.get_stop_loss_tp(symbol, action)
        if sl is not None and tp is not None:
            return sl, tp
        market_data = get_normalized_market_data(symbol)
        if market_data is None or market_data.height == 0:
            return None, None  # Dati mancanti, salta impostazione
        
        # Calcolo Zone ILQ
        ilq_levels = market_data.filter(pl.col("ILQ_Zone") == 1)["close"].to_numpy()
        if len(ilq_levels) == 0:
            return None, None  # Nessuna zona ILQ rilevata
        
        # Calcolo Take Profit e Stop Loss basati su ILQ
        if action == "buy":
            tp = np.max(ilq_levels)  # Delivery in alto
            sl = np.min(ilq_levels)  # Zona di supporto liquida
        else:  # Sell
            tp = np.min(ilq_levels)  # Delivery in basso
            sl = np.max(ilq_levels)  # Zona di resistenza liquida

        return sl, tp

    def update_take_profit(self, pos, new_tp, predicted_volatility):
        """
        Aggiorna il take profit direttamente su MT5
        per garantire protezione anche in caso di crash del bot.
        """
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
        Aggiunge posizioni alla posizione vincente con
        controllo del rischio su drawdown e correlazione.
        """
        if not self.smc_manager.enforce_entry_zone(symbol, current_price, action):
            logging.info(f"🚫 Prezzo fuori dalla ILQ Zone per {symbol}. Nessuna entrata eseguita.")
            return  # Blocca l'ingresso se non siamo nella zona di liquidità
        if not self.market_event_manager.should_trade(symbol, confidence_score, predicted_volatility):
            return  # Skip trade se le condizioni non sono ideali
        positions = mt5.positions_get(symbol=symbol)
        positions_open = len([p for p in positions if p.symbol == symbol])
        total_profit = sum(pos.profit for pos in positions if pos.symbol == symbol)
        max_positions_allowed = 5

        # Controllo sul Drawdown: evita overexposure
        account_info = mt5.account_info()
        if account_info is None or account_info.balance == 0:
            return

        equity_ratio = account_info.equity / account_info.balance
        if equity_ratio < 0.85:  # Se Drawdown > 15%, blocca il pyramiding
            logging.info("🚫 Pyramiding bloccato per Drawdown alto su %s", symbol)
            return

        trend_strength = self.volatility_predictor.predict_volatility(np.array([[current_price]]))[0]
        if positions_open < max_positions_allowed and trend_strength > 0.7:
            risk_per_trade = 0.01  # 1% del bilancio per trade, puoi modularlo
            predicted_volatility = self.volatility_predictor.predict_volatility(np.array([[current_price]]))[0]
            stop_loss_pips = predicted_volatility * 10000
            lot_size = self.calculate_dynamic_lot_size(account_info, risk_per_trade, stop_loss_pips, symbol)
            volume_increment = lot_size
            sl, tp = self.set_initial_sl_tp(symbol, action, current_price)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume_increment,
                "type": mt5.ORDER_BUY if action == "buy" else mt5.ORDER_SELL,
                "price": current_price,
                "sl": sl,
                "tp": tp,
                "deviation": 10,
                "magic": 0,
                "comment": "AI Auto Entry",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info("✅ Ordine aperto con successo: %s | SL: %.5f | TP: %.5f", symbol, sl, tp)
            else:
                logging.warning("⚠️ Errore apertura ordine su %s | Retcode: %d", symbol, result.retcode)

    def evaluate_swap_and_commissions(self, pos, action):
        """
        Valuta i costi di swap e commissioni. Se superano
        il profitto potenziale, chiude la posizione.
        """
        symbol_info = mt5.symbol_info(pos.symbol)
        if symbol_info is None:
            return
        
        swap_cost = symbol_info.swap_long if action == "buy" else symbol_info.swap_short
        # Simuliamo la commissione come costo per lotto (dipende dal broker, qui è un esempio)
        commission_per_lot = 5  # Esempio: 5$ per lotto

        total_swap = swap_cost * pos.volume
        total_commission = commission_per_lot * pos.volume
        total_cost = total_swap + total_commission

        # Se i costi superano il profitto o il margine di guadagno è troppo basso, chiude la posizione
        if pos.profit < total_cost or (pos.profit / total_cost) < 1.2:  # Guadagno minimo 20% sopra i costi
            logging.info(
                "💸 Costi swap/commissioni alti → Chiudo %s su %s | Profit: %.2f | Costi: %.2f",
                action.upper(), pos.symbol, pos.profit, total_cost
            )
            self.close_position(pos)

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
        
        portfolio_risk = self.calculate_portfolio_risk()
        if portfolio_risk > 0.6:  # Se già il 60% del capitale è esposto, blocca nuove operazioni
            logging.info("🛑 Rischio portafoglio alto: %.2f%% → Blocco nuove operazioni", portfolio_risk * 100)
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
            if market_data is not None and market_data.height > 0:
                self.adjust_sl_tp_based_on_ilq(pos, market_data)
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
            predicted_volatility =(
                self.volatility_predictor.predict_volatility(
                    full_state.reshape(1, -1))
            )
            action_rl, confidence_score, algo_used = self.drl_super_manager.get_best_action_and_confidence(full_state)
            if confidence_score < 0.5:
                logging.info(
                    "🤖 Bassa confidenza AI (%.2f) → Evito nuove operazioni su %s",
                    confidence_score, symbol
                )
                continue  # La AI sconsiglia di agire ora
        
            # Previsione del prezzo futuro (ad esempio tra 5 minuti)
            predicted_price = self.price_predictor.predict_price(
                full_state.reshape(1, -1)
            )[0]
            expected_gain = predicted_price - current_price if action == "buy" else current_price - predicted_price

            if last_row.get("trend_change_detected", 0) == 1:
                logging.info("⚠️ Cambio trend rilevato → chiudo %s su %s", action.upper(), pos.symbol)
                self.close_position(pos)
                continue

            if profit > predicted_volatility * 2 and gain > predicted_volatility:
                self.pyramiding_strategy(symbol, action, current_price)

            # Verifica condizioni di chiusura
            if self.should_close_position(
                pos, last_row, action, profit,
                signal_score, predicted_volatility
            ):
                self.close_position(pos)
                continue

            if (action == "buy" and expected_gain < -predicted_volatility) or \
                (action == "sell" and expected_gain < -predicted_volatility):
                # Verifica se lo SL è già abbastanza protettivo
                if (action == "buy" and pos.sl and pos.sl >= current_price - predicted_volatility) or \
                    (action == "sell" and pos.sl and pos.sl <= current_price + predicted_volatility):
                    logging.info(
                            "🔒 SL già protettivo → non chiudo %s su %s",
                            action.upper(), pos.symbol
                    )
                    continue

                logging.info(
                    "📉 Previsione negativa → chiudo %s su %s in anticipo",
                    action.upper(), pos.symbol
                )
                self.close_position(pos)
                continue

            if profit > 0 and self.handle_trailing_stop(
                pos, action, current_price, predicted_volatility, profit
            ):
                continue

            self.evaluate_swap_and_commissions(pos, action)

            if profit > predicted_volatility * 2:
                self.pyramiding_strategy(symbol, action, current_price)

            action_rl, confidence_score, algo_used = (
                self.drl_super_manager.get_best_action_and_confidence(full_state)
            )
            # Valuta la probabilità di successo tramite il DRL
            success_probability = self.calculate_success_probability(confidence_score, signal_score, predicted_volatility, expected_gain)


            if success_probability < 0.5:
                logging.info(
                    "📉 Bassa probabilità di successo (%.2f) → chiudo %s su %s",
                    success_probability, action.upper(), pos.symbol
                )
                self.close_position(pos)
                continue

            if profit > 0:
                self.dynamic_stop_management(
                    pos, action, current_price,
                    predicted_volatility, profit,
                    predicted_price, success_probability
                )


            if self.should_close_based_on_rl(pos, action, action_rl, confidence_score):
                self.close_position(pos)
                continue

        # Valutazione e ribilanciamento del portafoglio
        self.evaluate_portfolio_risk()
        self.rebalance_portfolio()

    def update_trailing_stop(self, pos, new_sl, predicted_volatility):
        """
        Aggiorna il trailing stop direttamente su MT5
        per garantire protezione anche in caso di crash del bot.
        """
        if predicted_volatility > 0.3:
            self.update_trailing_stop(pos, trailing_sl)
        result = mt5.order_modify(
            ticket=pos.ticket,
            price=pos.price_open,
            stoplimit=0,
            sl=new_sl,
            tp=pos.tp,
            deviation=10,
            type_time=mt5.ORDER_TIME_GTC,
            type_filling=mt5.ORDER_FILLING_IOC,
        )
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(
                "🔁 SL aggiornato nativamente su MT5 per %s: %.5f",
                pos.symbol, new_sl
            )
        else:
            logging.warning(
                "⚠️ Errore aggiornamento SL su MT5 per %s | Retcode: %d",
                pos.symbol, result.retcode
            )

    def update_trailing_take_profit(self, pos, action, current_price, gain, predicted_volatility):
        """
        Aggiorna dinamicamente il Take Profit su MT5 in modo intelligente e nativo.
        """
        if action == "buy":
            new_tp = current_price + (gain * 0.5)
            if pos.tp is None or new_tp > pos.tp:
                 self.update_take_profit(pos, new_tp, predicted_volatility)
        else:
            new_tp = current_price - (gain * 0.5)
            if pos.tp is None or new_tp < pos.tp:
                self.update_take_profit(pos, new_tp, predicted_volatility)

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

    def calculate_portfolio_risk(self):
        positions = mt5.positions_get()
        if not positions:
            return 0
        
        total_exposure = sum(pos.volume * mt5.symbol_info_tick(pos.symbol).last for pos in positions)
        account_info = mt5.account_info()
        return total_exposure / account_info.equity if account_info and account_info.equity else 0

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

    def calculate_success_probability(self, signal_score, predicted_volatility, expected_gain):
        """
        Calcola la probabilità di successo con pesi dinamici che si adattano
        alle condizioni di mercato (market sentiment, volatilità e guadagno atteso).
        """
        # Normalizzazioni
        confidence_signal = np.clip(signal_score / 10, 0, 1)
        confidence_volatility = np.clip(1 - predicted_volatility, 0, 1)
        confidence_gain = np.clip(expected_gain / (predicted_volatility + 1e-5), 0, 1)

        # Pesi dinamici in base alla volatilità
        if predicted_volatility < 0.3:
            # Mercato stabile → Dai più peso al gain e ai segnali AI
            weight_signal = 0.5
            weight_volatility = 0.2
            weight_gain = 0.3
        elif predicted_volatility < 0.7:
            # Mercato moderatamente volatile → Equilibrio tra i fattori
            weight_signal = 0.4
            weight_volatility = 0.3
            weight_gain = 0.3
        else:
        # Mercato molto volatile → Dai più peso alla volatilità per evitare rischi
            weight_signal = 0.3
            weight_volatility = 0.5
            weight_gain = 0.2

        success_probability = (
            (weight_signal * confidence_signal) +
            (weight_volatility * confidence_volatility) +
            (weight_gain * confidence_gain)
            )

        return np.clip(success_probability, 0, 1)
