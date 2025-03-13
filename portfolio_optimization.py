# portfolio_optimization.py
import numpy as np
import polars as pl
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from pypfopt.hierarchical_risk_parity import HRPOpt
from risk_management import RiskManagement
from sklearn.preprocessing import MinMaxScaler

# 📌 Configurazione del logging avanzato
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levellevel)s - %(message)s"
)

# ThreadPoolExecutor per calcoli paralleli
executor = ThreadPoolExecutor(max_workers=4)


class PortfolioOptimizer:
    """
    Ottimizzatore del portafoglio con gestione avanzata del rischio,
    supporto per scalping e auto-adattamento basato su Polars.
    """

    def __init__(self, market_data, balance, risk_tolerance=0.05, scalping=False):
        self.market_data = market_data
        self.scalping = scalping
        self.risk_management = RiskManagement(max_risk=risk_tolerance)
        self.balance = balance  # Il saldo attuale per regolare l'allocazione

    async def optimize_portfolio(self):
        """
        Ottimizza il portafoglio in base al tipo di trading 
        (scalping o dati storici) e gestione del rischio.
        """
        if self.scalping:
            logging.info("⚡ Ottimizzazione per scalping in corso...")
            return await asyncio.to_thread(self._optimize_scalping)
        else:
            logging.info("📊 Ottimizzazione per dati storici in corso...")
            return await asyncio.to_thread(self._optimize_historical)

    def _optimize_historical(self):
        """Ottimizzazione basata su dati storici con gestione avanzata del rischio."""
        prices = self._prepare_price_data()
        mu = mean_historical_return(prices)
        S = CovarianceShrinkage(prices).ledoit_wolf()
        ef = EfficientFrontier(mu, S)

        # Massimizza Sharpe Ratio con gestione del rischio dinamica
        weights = ef.max_sharpe()
        cleaned_weights = self.risk_management.apply_risk_constraints(
            ef.clean_weights()
        )
        logging.info(f"✅ Allocazione storica ottimizzata: {cleaned_weights}")
        return cleaned_weights

    def _optimize_scalping(self):
        """Ottimizzazione per scalping basata su alta frequenza e liquidità."""
        recent_prices = self._prepare_price_data().tail(20)
        hrp = HRPOpt(recent_prices)
        hrp_weights = hrp.optimize()

        # Gestione avanzata del rischio per scalping
        optimized_weights = self.risk_management.apply_risk_constraints(
            hrp_weights
        )
        logging.info(f"⚡ Allocazione scalping ottimizzata: {optimized_weights}")
        return optimized_weights

    async def optimize_with_constraints(self):
        """Ottimizza il portafoglio con vincoli avanzati e gestione del rischio."""
        max_risk_allowed = await asyncio.to_thread(
            self.risk_management.adjust_risk, self.balance
        )

        def objective(weights):
            """Funzione obiettivo: massimizzare Sharpe Ratio con penalizzazione rischio."""
            port_return = np.dot(weights, self.market_data.mean(axis=0).to_numpy())
            port_volatility = np.sqrt(
                np.dot(weights.T, np.dot(self.market_data.cov().to_numpy(), weights))
            )
            sharpe_ratio = (port_return - 0.01) / port_volatility

            # Penalizza se il rischio supera il massimo consentito
            return np.inf if port_volatility > max_risk_allowed else -sharpe_ratio

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1)] * len(self.market_data.columns)
        initial_guess = np.ones(len(self.market_data.columns)) / len(
            self.market_data.columns
        )

        result = await asyncio.to_thread(
            minimize,
            objective,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        optimized_allocation = result.x if result.success else np.zeros_like(
            initial_guess
        )
        logging.info(
            f"🔍 Allocazione finale con vincoli di rischio: {optimized_allocation}"
        )
        return optimized_allocation

    def _prepare_price_data(self):
        """
        Prepara i dati dei prezzi per l'ottimizzazione convertendoli in Polars.
        """
        df = pl.DataFrame(self.market_data)
        df = df.select(["timestamp", "symbol", "close"])
        return df.pivot(index="timestamp", columns="symbol", values="close")

    def calculate_performance_metrics(self):
        """
        Calcola metriche di performance del portafoglio utilizzando Polars.
        """
        return self.market_data.groupby('symbol').agg([
            pl.col('return').mean().alias('mean_return'),
            pl.col('return').std().alias('return_volatility'),
            (pl.col('return').mean() / pl.col('return').std()).alias('sharpe_ratio')
        ])

    def normalize_data(self):
        """
        Normalizza e scala i dati di mercato utilizzando Polars.
        """
        numeric_cols = self.market_data.select(pl.col(pl.NUMERIC_DTYPES)).columns
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.market_data.select(numeric_cols).to_numpy())
        self.market_data = self.market_data.with_columns(
            [pl.Series(col, scaled_data[:, idx]) for idx, col in enumerate(numeric_cols)]
        )


async def optimize_for_conditions(market_data, balance, market_condition, risk_tolerance=0.05):
    """
    Seleziona automaticamente l'ottimizzazione migliore in base alle condizioni
    di mercato e al saldo disponibile.
    """
    optimizer = PortfolioOptimizer(
        market_data, balance, risk_tolerance, scalping=(market_condition == "scalping")
    )
    return await optimizer.optimize_with_constraints()


async def dynamic_allocation(trading_pairs, capital):
    """
    Distribuisce il capitale basandosi su volatilità, trend e liquidità.
    """
    total_score = sum(
        [pair[2] * abs(pair[3] - pair[4]) for pair in trading_pairs]
    )  # Ponderazione
    allocations = {}

    for pair in trading_pairs:
        weight = (pair[2] * abs(pair[3] - pair[4])) / total_score
        allocations[pair[0]] = capital * weight  # Distribuzione intelligente

    return allocations

def complex_calculation(df):
    """
    Placeholder function for complex calculations.
    Replace this with the actual implementation.
    """
    # Implement your complex calculation logic here
    # For now, let's just return the input DataFrame
    return df

def parallel_calculations(df):
    """
    Esegue calcoli paralleli su DataFrame di Polars.
    """
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(complex_calculation, df.filter(pl.col('symbol') == symbol)) for symbol in df['symbol'].unique()]
        results = [future.result() for future in futures]
    return pl.concat(results)
