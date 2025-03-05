# risk_management
import logging
import ccxt
import numpy as np
import talib
import data_handler  # Per gestire i dati di mercato (normalizzati)
from datetime import datetime, timedelta
from ai_model import VolatilityPredictor

# 📌 Configurazione avanzata del logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RiskManagement:
    """Gestisce il rischio, l'allocazione del capitale e il trailing stop in modo avanzato per il trading SPOT."""
    
    def __init__(self, max_drawdown=0.20, risk_per_trade=0.02, max_exposure=0.5):
        self.max_drawdown = max_drawdown  # Percentuale massima di perdita prima di fermare il trading
        self.trailing_stop_pct = 0.05  # Valore predefinito del trailing stop
        self.min_balance = float('inf')
        self.highest_balance = 0
        self.kill_switch_activated = False
        self.risk_per_trade = risk_per_trade  # Percentuale del saldo investita per trade
        self.max_exposure = max_exposure  # Percentuale massima del saldo totale allocata a trade aperti
        self.volatility_predictor = VolatilityPredictor()

    def adaptive_stop_loss(self, entry_price, pair):
    """Calcola uno stop-loss e trailing-stop basato su volatilità e trend."""
    ohlcv = self.exchange.fetch_ohlcv(pair, timeframe="1h")
    closes = [candle[4] for candle in ohlcv]
    volatility = np.std(closes) / np.mean(closes)

    stop_loss = entry_price * (1 - (volatility * 1.5))  # Stop-loss adattivo
    trailing_stop = entry_price * (1 - (volatility * 0.8))  # Trailing-stop meno aggressivo

    return stop_loss, trailing_stop

    def adjust_risk(self, market_data):
        """Adatta dinamicamente il trailing stop e il capitale in base alla volatilità del mercato."""
        future_volatility = self.volatility_predictor.predict_volatility(np.array([[market_data['volume'], market_data['price_change'], market_data['rsi'], market_data['bollinger_width']]]))
        atr = future_volatility[0] * 100  # Previsione volatilità futura
        
        if atr > 15:
            self.trailing_stop_pct = 0.15
            self.risk_per_trade = 0.01  # Riduce il rischio in mercati instabili
        elif atr > 10:
            self.trailing_stop_pct = 0.1
            self.risk_per_trade = 0.015
        else:
            self.trailing_stop_pct = 0.05
            self.risk_per_trade = 0.02  # Maggiore rischio in mercati stabili

    def check_drawdown(self, current_balance):
        """Verifica il drawdown e riattiva il trading se il saldo migliora."""
        if current_balance < self.min_balance:
            self.min_balance = current_balance
        
        drawdown = (self.highest_balance - current_balance) / self.highest_balance if self.highest_balance > 0 else 0
        
        if drawdown > self.max_drawdown:
            logging.warning(f"⚠️ Drawdown critico: {drawdown:.2%}. Trading sospeso.")
            self.kill_switch_activated = True
        elif drawdown < (self.max_drawdown / 2):  # Riattivazione del trading
            logging.info("✅ Drawdown recuperato, riattivazione del trading.")
            self.kill_switch_activated = False

    def calculate_position_size(self, balance, market_conditions):
        """Determina la dimensione ottimale della posizione in base al saldo e alle condizioni di mercato."""
        base_position_size = balance * self.risk_per_trade
        adjusted_position_size = base_position_size * (1 + market_conditions['momentum'])  # Aumenta se il trend è positivo
        max_allowed = balance * self.max_exposure
        return min(adjusted_position_size, max_allowed)

class TradingBot:
    """Gestisce il trading SPOT con trailing stop adattivo e gestione avanzata del capitale."""
    
    def __init__(self, account_name, risk_management, balance=100):
        self.account_name = account_name
        self.risk_management = risk_management
        self.entry_price = None
        self.current_position = None
        self.balance = balance

    def manage_trade(self, market_data):
        """Gestisce il trade e applica il trailing stop dinamico."""
        if self.entry_price is None:
            self.entry_price = market_data['entry_price']
        
        self.risk_management.adjust_risk(market_data)
        trailing_stop = self.entry_price * (1 - self.risk_management.trailing_stop_pct)
        logging.info(f"{self.account_name}: Trailing stop impostato a {trailing_stop}.")

    def update_trailing_stop(self, market_price):
        """Aggiorna il trailing stop se il prezzo di mercato aumenta."""
        if self.current_position == 'long' and market_price > self.entry_price:
            new_stop = market_price * (1 - self.risk_management.trailing_stop_pct)
            if new_stop > self.entry_price * (1 - self.risk_management.trailing_stop_pct):
                logging.info(f"🔄 {self.account_name}: Trailing stop aggiornato a {new_stop}.")
                self.entry_price = market_price  # Aggiorna il prezzo d'ingresso
    
    def execute_trade(self, market_data):
        """Esegue un'operazione basata sulla gestione del capitale."""
        position_size = self.risk_management.calculate_position_size(self.balance, market_data)
        logging.info(f"{self.account_name}: Apertura trade con {position_size} unità.")
        self.balance -= position_size  # Simulazione della riduzione del saldo dopo l'apertura del trade

# 📌 Inizializzazione automatica dei bot di trading
risk_manager = RiskManagement()
denny_bot = TradingBot("Denny", risk_manager)
giuseppe_bot = TradingBot("Giuseppe", risk_manager)

# 📌 Simulazione di dati di mercato e test automatico
market_data = {"entry_price": 100, "volatility": 0.12, "predicted_price": 110, "momentum": 0.05, "volume": 500000, "price_change": 2.5, "rsi": 65, "bollinger_width": 0.05}
denny_bot.manage_trade(market_data)
giuseppe_bot.manage_trade(market_data)

denny_bot.execute_trade(market_data)
giuseppe_bot.execute_trade(market_data)

denny_bot.update_trailing_stop(112)
giuseppe_bot.update_trailing_stop(115)
