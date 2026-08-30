"""
Monte Carlo pricing engine for Stochastic Local Volatility (SLV) models.

Simulates correlated spot/volatility paths under calibrated SLV dynamics
and prices European-style options via discounted payoff averaging.
"""

from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

from .models import OrnsteinUhlenbeckSV, ModelParameters
from .utils import blackscholes_impvol_call


@dataclass
class SimulationResult:
    """Stores simulation paths and terminal statistics."""
    S_paths: np.ndarray  # Shape: (n_steps+1, N)
    Y_paths: np.ndarray  # Shape: (n_steps+1, N)
    terminal_S: np.ndarray  # Shape: (N,)


class MonteCarloEngine:
    """
    Simulates correlated (S, Y) paths under SLV dynamics.
    Uses time-dependent step leverage L(t, S) from Calibrator.
    """

    def __init__(self, model_params: ModelParameters):
        self.params = model_params

    def simulate_paths(
        self,
        leverage_func: Callable[[float, np.ndarray], np.ndarray],
        N: int = 100000,
        dt: float = 0.02,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """
        Simulate N paths under SLV dynamics until maturity T.

        Parameters
        ----------
        leverage_func : callable(t, S) -> np.ndarray
            Function returning local volatility multiplier L(t, S).
        N : int
            Number of Monte Carlo paths.
        dt : float
            Time step size.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        SimulationResult
            Contains path arrays and terminal spot distribution.
        """
        if seed is not None:
            np.random.seed(seed)

        p = self.params
        n_steps = int(np.round(p.T / dt))

        S = np.full((n_steps + 1, N), p.S0, dtype=float)
        Y = np.full((n_steps + 1, N), p.Y0, dtype=float)

        model = OrnsteinUhlenbeckSV(p)

        for k in range(n_steps):
            t_k = k * dt
            l_k = leverage_func(t_k, S[k])

            Z1 = np.random.randn(N)
            Z2 = np.random.randn(N)

            Y[k + 1] = model.exact_OU_step(Y[k], dt, Z2)
            S[k + 1] = model.euler_S_step(S[k], Y[k], l_k, dt, Z1, Z2)

        return SimulationResult(S_paths=S, Y_paths=Y, terminal_S=S[-1])

    def price_european_call(
        self,
        leverage_func: Callable[[float, np.ndarray], np.ndarray],
        K: float,
        N: int = 100000,
        dt: float = 0.02,
        r: float = 0.0,
        seed: Optional[int] = None,
    ) -> float:
        """Price a European call using Monte Carlo under SLV dynamics."""
        p = self.params
        result = self.simulate_paths(leverage_func, N=N, dt=dt, seed=seed)
        ST = result.terminal_S
        payoff = np.maximum(ST - K, 0)
        return float(np.exp(-r * p.T) * np.mean(payoff))

    def implied_vol_smile(
        self,
        leverage_func: Callable[[float, np.ndarray], np.ndarray],
        strikes: np.ndarray,
        N: int = 100000,
        dt: float = 0.02,
        r: float = 0.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the implied volatility smile from SLV-simulated call prices.
        """
        p = self.params
        result = self.simulate_paths(leverage_func, N=N, dt=dt, seed=seed)
        ST = result.terminal_S

        prices = []
        for K in strikes:
            payoff = np.maximum(ST - K, 0)
            price = float(np.exp(-r * p.T) * np.mean(payoff))
            prices.append(price)

        vols = [
            blackscholes_impvol_call(K, p.T, p.S0, C, r=r)
            for K, C in zip(strikes, prices)
        ]
        return np.array(vols)
