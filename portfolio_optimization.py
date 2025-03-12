# portfolio_optimization
import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from pypfopt.hierarchical_risk_parity import HRPOpt
from risk_management import RiskManagement

# 📌 Configurazione del logging avanzato
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PortfolioOptimizer:
    """Ottimizzatore del portafoglio con gestione avanzata del rischio, supporto per scalping e auto-adattamento."""

    def __init__(self, market_data, balance, risk_tolerance=0.05, scalping=False):
        self.market_data = market_data
        self.scalping = scalping
        self.risk_management = RiskManagement(max_risk=risk_tolerance)
        self.balance = balance  # Il saldo attuale viene considerato per regolare l'allocazione

    def optimize_portfolio(self):
        """Ottimizza il portafoglio in base al tipo di trading (scalping o dati storici) e gestione del rischio."""
        if self.scalping:
            logging.info("⚡ Ottimizzazione per scalping in corso...")
            return self._optimize_scalping()
        else:
            logging.info("📊 Ottimizzazione per dati storici in corso...")
            return self._optimize_historical()

    def _optimize_historical(self):
        """Ottimizzazione classica basata su dati storici con gestione avanzata del rischio."""
        prices = self.market_data.pivot_table(index="timestamp", columns="symbol", values="close")
        mu = mean_historical_return(prices)
        S = CovarianceShrinkage(prices).ledoit_wolf()
        ef = EfficientFrontier(mu, S)

        # Massimizza Sharpe Ratio con gestione del rischio dinamica
        weights = ef.max_sharpe()
        cleaned_weights = self.risk_management.apply_risk_constraints(ef.clean_weights())
        logging.info(f"✅ Allocazione storica ottimizzata con gestione del rischio: {cleaned_weights}")
        return cleaned_weights

    def _optimize_scalping(self):
        """Ottimizzazione per scalping basata su alta frequenza e liquidità con gestione del rischio."""
        recent_prices = self.market_data.pivot_table(index="timestamp", columns="symbol", values="close").iloc[-20:]
        hrp = HRPOpt(recent_prices)
        hrp_weights = hrp.optimize()

        # Gestione avanzata del rischio per scalping
        optimized_weights = self.risk_management.apply_risk_constraints(hrp_weights)
        logging.info(f"⚡ Allocazione scalping ottimizzata con gestione del rischio: {optimized_weights}")
        return optimized_weights

    def optimize_with_constraints(self):
        """Ottimizza il portafoglio con vincoli avanzati e gestione del rischio basata sul saldo disponibile."""
        max_risk_allowed = self.risk_management.adjust_risk(self.balance)

        # Funzione obiettivo: massimizzare Sharpe Ratio con penalizzazione per rischio elevato
        def objective(weights):
            port_return = np.dot(weights, self.market_data.mean())
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.market_data.cov(), weights)))
            sharpe_ratio = (port_return - 0.01) / port_volatility

            # Penalizza se il rischio supera il massimo consentito
            if port_volatility > max_risk_allowed:
                return np.inf  

            return -sharpe_ratio  # Minimizza il negativo per massimizzare Sharpe Ratio

        constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        bounds = [(0, 1) for _ in range(len(self.market_data.columns))]
        initial_guess = np.ones(len(self.market_data.columns)) / len(self.market_data.columns)
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        optimized_allocation = result.x if result.success else np.zeros_like(initial_guess)
        logging.info(f"🔍 Allocazione finale con vincoli di rischio: {optimized_allocation}")
        return optimized_allocation

def optimize_for_conditions(market_data, balance, market_condition, risk_tolerance=0.05):
    """Seleziona automaticamente l'ottimizzazione migliore in base alle condizioni di mercato e al saldo disponibile."""
    optimizer = PortfolioOptimizer(market_data, balance, risk_tolerance, scalping=(market_condition == "scalping"))
    return optimizer.optimize_with_constraints()

def dynamic_allocation(trading_pairs, capital):
    """Distribuisce il capitale basandosi su volatilità, trend e liquidità."""
    total_score = sum([pair[2] * abs(pair[3] - pair[4]) for pair in trading_pairs])  # Ponderazione
    allocations = {}

    for pair in trading_pairs:
        weight = (pair[2] * abs(pair[3] - pair[4])) / total_score  # Combina volatilità e trend
        allocations[pair[0]] = capital * weight  # Distribuzione intelligente del capitale

    return allocations
