"""
Core stochastic volatility model classes.

Defines abstract base class and concrete implementations for:
- Heston-style OU-driven stochastic volatility
- Dupire local volatility surface fitting
- Realistic implied volatility surfaces (SVI parameterization)
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm


@dataclass
class ModelParameters:
    """
    Container for model parameters shared between SV and SLV models.

    Attributes
    ----------
    S0 : float
        Initial spot price.
    T : float
        Time horizon (in years).
    sigma0 : float
        Base volatility level.
    Y0 : float
        Initial value of the OU process (log-variance driver).
    kappa : float
        Mean-reversion speed of the volatility process.
    gamma : float
        Volatility of volatility (vol-of-vol).
    rho : float
        Correlation between spot and volatility Brownian motions (-1 < rho < 1).
    """
    S0: float
    T: float
    sigma0: float
    Y0: float
    kappa: float
    gamma: float
    rho: float


class StochasticVolatilityModel(ABC):
    """
    Abstract base class for stochastic volatility models.

    Subclasses must implement simulate() to advance paths forward.
    """

    def __init__(self, params: ModelParameters):
        self.params = params

    @abstractmethod
    def simulate(self, N_paths: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths over one time step dt.

        Returns
        -------
        (S_new, Y_new) : tuple[np.ndarray, np.ndarray]
            Updated arrays of spot and log-variance processes.
        """


class OrnsteinUhlenbeckSV(StochasticVolatilityModel):
    """
    Heston-style stochastic volatility model where log-volatility Y follows an
    Ornstein-Uhlenbeck process:

        dY_t = -kappa * Y_t dt + gamma dW2_t
        dS_t = a_t * L(t, S_t) * S_t dW1_t
        d<W1, W2> = rho dt

    with a_t = sigma0 * exp(Y_t).
    """

    def __init__(self, params: ModelParameters):
        super().__init__(params)

    def exact_OU_step(self, Y_prev: np.ndarray, dt: float, Z2: np.ndarray) -> np.ndarray:
        """
        Exact simulation of the OU process Y over a time step dt.

        Given Y_{t-dt}, returns Y_t sampled from the known Gaussian transition law.
        Z2 must be standard normal N(0, 1).
        """
        p = self.params
        decay = np.exp(-p.kappa * dt)
        std_dev = np.sqrt((p.gamma**2 / (2 * p.kappa)) * (1 - np.exp(-2 * p.kappa * dt)))
        return Y_prev * decay + std_dev * Z2

    def euler_S_step(
        self,
        S_prev: np.ndarray,
        Y_prev: np.ndarray,
        local_vol_values: np.ndarray,
        dt: float,
        Z1: np.ndarray,
        Z2: np.ndarray,
    ) -> np.ndarray:
        """
        Euler scheme step for spot price S under SLV dynamics.

        Z1 and Z2 must be INDEPENDENT standard normal N(0, 1) arrays.
        Correlated Brownian noise W1 = sqrt(1 - rho_bar^2) * Z1 + rho_bar * Z2
        is constructed internally preserving the sign of rho.
        """
        p = self.params
        a_prev = p.sigma0 * np.exp(Y_prev)
        inst_vol = a_prev * local_vol_values

        rho_bar_sq = self._rho_bar_squared(dt)
        rho_bar = self._rho_bar(dt)

        # Correlated Brownian driver W1 with correlation rho_bar to Z2
        W1 = np.sqrt(max(1.0 - rho_bar_sq, 0.0)) * Z1 + rho_bar * Z2

        drift_term = -0.5 * inst_vol**2 * dt
        diffusion_term = inst_vol * np.sqrt(dt) * W1
        return S_prev * np.exp(drift_term + diffusion_term)

    def _rho_bar_squared(self, dt: float) -> float:
        """Effective squared correlation over time step dt."""
        p = self.params
        exponent = np.exp(-p.kappa * dt)
        numerator = 2 * (1 - exponent)
        denominator = p.kappa * dt * (1 + exponent)
        return p.rho**2 * numerator / denominator

    def _rho_bar(self, dt: float) -> float:
        """Effective correlation used in Euler discretization (preserves sign of rho)."""
        p = self.params
        return np.sign(p.rho) * np.sqrt(self._rho_bar_squared(dt))

    def simulate(self, N_paths: int, dt: float, local_vol_values: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform one time step of the SLV model using independent standard normals.
        """
        p = self.params
        Z1, Z2 = np.random.randn(N_paths), np.random.randn(N_paths)
        Y_prev = np.full(N_paths, p.Y0) if not hasattr(self, '_Y') else self._Y
        S_prev = np.full(N_paths, p.S0) if not hasattr(self, '_S') else self._S

        Y_new = self.exact_OU_step(Y_prev, dt, Z2)

        if local_vol_values is None:
            local_vol_values = np.ones_like(S_prev)

        S_new = self.euler_S_step(S_prev, Y_prev, local_vol_values, dt, Z1, Z2)

        self._Y, self._S = Y_new, S_new
        return S_new, Y_new


@dataclass
class SVIParameters:
    """
    SVI (Stochastic Volatility Inspired) parameterization for implied volatility.

    SVI formula for total variance w = sigma^2 * T:
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

    where k = log(K/F) is log-moneyness.
    """
    a: float      # Overall variance level
    b: float      # Slope/asymmetry parameter
    rho: float    # Skew parameter (-1 < rho < 1)
    m: float      # Horizontal shift (ATM location)
    sigma: float  # Curvature parameter (sigma > 0)
    S0: float     # Forward/spot
    T: float      # Time to maturity


class ImpliedVolatilitySurface:
    """
    Represents a market-implied volatility surface.

    Supports:
    - Flat volatility
    - SVI smile (arbitrage-free, realistic equity skew/smile)
    - Custom strike/vol grid with spline interpolation
    """

    def __init__(
        self,
        sigma_imp: float = None,
        svi_params: SVIParameters = None,
        strikes: np.ndarray = None,
        vols: np.ndarray = None
    ):
        if sum(x is not None for x in [sigma_imp, svi_params]) + (1 if (strikes is not None and vols is not None) else 0) != 1:
            raise ValueError("Exactly one of sigma_imp, svi_params, or (strikes, vols) must be provided")

        self.sigma_imp = sigma_imp
        self.svi_params = svi_params
        self.custom_strikes = strikes
        self.custom_vols = vols

        if svi_params is not None:
            self._build_svi_smile()
        elif strikes is not None and vols is not None:
            self._build_custom_surface()

    def _build_svi_smile(self):
        """Build implied vol smile using SVI parameterization."""
        p = self.svi_params
        self._svi_strikes = np.linspace(0.3 * p.S0, 3.0 * p.S0, 300)
        self._svi_vols = np.array([self._svi_impvol(K, p) for K in self._svi_strikes])
        self._interp = interp1d(self._svi_strikes, self._svi_vols, kind='cubic', fill_value='extrapolate')

    def _svi_impvol(self, K: float, p: SVIParameters) -> float:
        """SVI implied vol formula."""
        if K <= 0:
            return np.nan

        F = p.S0
        k = np.log(K / F)

        w = p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m)**2 + p.sigma**2))
        w = max(w, 1e-8)

        vol = np.sqrt(w / p.T)
        return max(vol, 1e-4)

    def _build_custom_surface(self):
        """Build surface from custom strikes/vols with cubic spline."""
        self._interp = interp1d(self.custom_strikes, self.custom_vols, kind='cubic', fill_value='extrapolate')

    def get_volatility(self, K: np.ndarray, T: float = None) -> np.ndarray:
        """Return implied volatility for given strikes."""
        if self.sigma_imp is not None:
            return np.full_like(K, self.sigma_imp) if isinstance(K, np.ndarray) else self.sigma_imp
        else:
            return self._interp(K)

    def to_dupire_local_vol(self, K_grid: np.ndarray = None, T: float = None, r: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Dupire local volatility surface from implied volatility surface.
        Uses stable log-moneyness formulation to prevent derivative singularities.
        """
        if self.sigma_imp is not None:
            if K_grid is None:
                K_grid = np.linspace(30, 300, 200)
            return K_grid, np.full_like(K_grid, self.sigma_imp)

        if K_grid is None:
            K_grid = np.linspace(0.3 * self.S0, 3.0 * self.S0, 200)

        F = self.S0 * np.exp(r * (T or 1.0))
        T_val = T or (self.svi_params.T if self.svi_params else 1.0)
        T_val = max(T_val, 1e-4)

        k_eval = np.log(K_grid / F)
        k_fine = np.linspace(min(k_eval.min() - 0.5, -2.0), max(k_eval.max() + 0.5, 2.0), 1000)
        dk = k_fine[1] - k_fine[0]

        K_fine = F * np.exp(k_fine)
        sigma_imp = self.get_volatility(K_fine, T_val)
        w = sigma_imp**2 * T_val

        dw_dk = np.gradient(w, dk)
        d2w_dk2 = np.gradient(dw_dk, dk)
        dw_dT = w / T_val

        num = dw_dT
        w_safe = np.maximum(w, 1e-8)
        den = 1.0 - (k_fine / w_safe) * dw_dk + 0.25 * (-0.25 - 1.0/w_safe + (k_fine**2)/(w_safe**2)) * (dw_dk**2) + 0.5 * d2w_dk2

        den = np.maximum(den, 1e-4)
        num = np.maximum(num, 1e-8)

        local_var = num / den
        local_vol = np.sqrt(np.clip(local_var, 1e-4, 4.0))

        local_vol_interp = interp1d(k_fine, local_vol, kind='cubic', bounds_error=False, fill_value=(local_vol[0], local_vol[-1]))
        res_vol = local_vol_interp(k_eval)

        return K_grid, res_vol

    @property
    def S0(self):
        """Spot/forward for the surface."""
        if self.svi_params:
            return self.svi_params.S0
        return 100.0


def black_scholes_call(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes call price (forward notation)."""
    if T <= 0:
        return max(F - K, 0)
    d1 = np.log(F / K) / (sigma * np.sqrt(T)) + 0.5 * sigma * np.sqrt(T)
    d2 = d1 - sigma * np.sqrt(T)
    return F * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes put price (forward notation)."""
    if T <= 0:
        return max(K - F, 0)
    d1 = np.log(F / K) / (sigma * np.sqrt(T)) + 0.5 * sigma * np.sqrt(T)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - F * norm.cdf(-d1)
