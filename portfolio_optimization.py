"""
portfolio_optimization.py
Modulo per l'ottimizzazione del portafoglio con gestione avanzata del rischio,
supporto per scalping e auto-adattamento basato su Polars.
"""
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import polars as pl
from scipy.optimize import minimize
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from pypfopt.hierarchical_risk_parity import HRPOpt
from risk_management import RiskManagement

# 📌 Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ThreadPoolExecutor per calcoli paralleli
parallel_executor = ThreadPoolExecutor(max_workers=4)


class PortfolioOptimizer:
    """
    Ottimizzatore del portafoglio con gestione avanzata del rischio,
    supporto per scalping e auto-adattamento basato su Polars.
    """

    def __init__(self, market_data, balances, scalping=False):
        """
        balances = {
            "Danny": 1000,
            "Giuseppe": 1500
        }
        """
        self.market_data = market_data
        self.scalping = scalping
        self.balances = balances  # 🔥 Saldo per ogni account
        self.risk_tolerances = self._calculate_dynamic_risk_tolerances()
        self.risk_management = {
            account: RiskManagement(self.risk_tolerances[account])
            for account in self.balances
        }  # 🔥 Creiamo un RiskManagement per ogni account

    async def optimize_portfolio(self):
        """
        Ottimizza il portafoglio in base al tipo di trading
        (scalping o dati storici) e gestione del rischio.
        """
        if self.scalping:
            logging.info("⚡ Ottimizzazione per scalping in corso...")
            return await asyncio.to_thread(self._optimize_scalping)
        logging.info("📊 Ottimizzazione per dati storici in corso...")
        return await asyncio.to_thread(self._optimize_historical)

    def _calculate_dynamic_risk_tolerances(self):
        """
        Calcola automaticamente `risk_tolerance` per ogni account basandosi su:
        - Volatilità del mercato
        - Saldo individuale
        - Drawdown recente
        """
        risk_tolerances = {}

        for account, balance in self.balances.items():
            volatility = self.market_data.select(
                pl.col("volatility").mean()
            ).item()
            drawdown = self.market_data.select(
                pl.col("drawdown").min()
            ).item()

            balance_factor = min(0.05, balance / 10000)
            risk_tolerance = max(
                0.01, min(0.05, balance_factor / (volatility * 10))
            )

            risk_tolerances[account] = risk_tolerance
            logging.info(
                "📊 %s - Risk Tolerance Dinamico: %.4f",
                account, risk_tolerance
            )

        return risk_tolerances

    def _optimize_historical(self):
        """Ottimizzazione basata su dati storici con gestione avanzata."""
        prices = self._prepare_price_data()
        mu = mean_historical_return(prices)
        cov_matrix = CovarianceShrinkage(prices).ledoit_wolf()
        ef = EfficientFrontier(mu, cov_matrix)

        weights = ef.max_sharpe()
        cleaned_weights = {
            acc: self.risk_management[acc].apply_risk_constraints(
                ef.clean_weights()
            ) for acc in self.balances
        }
        logging.info("✅ Allocazione storica ottimizzata: %s", cleaned_weights)
        return cleaned_weights, weights

    def _optimize_scalping(self):
        """Ottimizzazione per scalping basata su alta frequenza e liquidità."""
        recent_prices = self._prepare_price_data().tail(20)
        hrp = HRPOpt(recent_prices)
        hrp_weights = hrp.optimize()

        optimized_weights = {
            acc: self.risk_management[acc].apply_risk_constraints(
                hrp_weights
            ) for acc in self.balances
        }
        logging.info("Allocazione scalping ottimizzata: %s", optimized_weights)
        return optimized_weights, hrp_weights

    def _prepare_price_data(self):
        """
        Prepara i dati dei prezzi per l'ottimizzazione convertendoli in Polars.
        """
        df = pl.DataFrame(self.market_data)
        df = df.select(["timestamp", "symbol", "close"])
        return df.pivot(index="timestamp", columns="symbol", values="close")


async def optimize_for_conditions(market_data, balances, market_condition):
    """
    Seleziona automaticamente l'ottimizzazione migliore in base alle condizioni
    di mercato e al saldo disponibile.
    """
    optimizer = PortfolioOptimizer(
        market_data, balances, scalping=(market_condition == "scalping")
    )
    return await optimizer.optimize_portfolio()


def parallel_calculations(df):
    """
    Esegue calcoli paralleli su DataFrame di Polars.
    """
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(
            complex_calculation, df.filter(pl.col('symbol') == symbol)
        ) for symbol in df['symbol'].unique()]
        results = [future.result() for future in futures]
    return pl.concat(results)


def complex_calculation(df):
    """
    Esegue calcoli avanzati su un dataset:
    - Volatilità annualizzata
    - Maximum Drawdown (per misurare la perdita massima)
    """
    df = df.with_columns(
        pl.col("close").pct_change().alias("returns")
    ).drop_nulls()

    annual_volatility = df.select(
        (pl.col("returns").std() * (252 ** 0.5)).alias("annualized_volatility")
    )

    df = df.with_columns(annual_volatility)

    df = df.with_columns(
        pl.col("close").cummax().alias("rolling_max")
    ).with_columns(
        ((df["close"] - df["rolling_max"]) /
         df["rolling_max"]).alias("drawdown")
    )

    max_drawdown = df.select(pl.min("drawdown").alias("max_drawdown"))

    df = df.with_columns(max_drawdown)

    return df
