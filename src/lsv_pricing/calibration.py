"""
Calibration routines for Stochastic Local Volatility (SLV) models.

Implements the McKean-Vlasov particle calibration scheme (Guyon & Henry-Labordère, 2012):
- Simulate paths forward step-by-step from t_0 to T
- At each step t_k, regress a_t² onto S_t using Nadaraya-Watson kernel regression
- Compute leverage function L(t_k, S) = sigma_Dup(t_k, S) / sqrt(E[a_t² | S_t = S])
- Advance particles to t_{k+1} using the step leverage L(t_k, S)
"""

from dataclasses import dataclass
from typing import Callable, Optional, List
import numpy as np
from scipy.interpolate import interp1d

from .models import OrnsteinUhlenbeckSV, ModelParameters, ImpliedVolatilitySurface
from .utils import nadaraya_watson_regression, quartic_kernel


@dataclass
class CalibrationResult:
    """Stores results of the step-by-step particle calibration procedure."""
    time_grid: np.ndarray
    leverage_functions: List[Callable[[np.ndarray], np.ndarray]]
    knots_per_step: List[np.ndarray]
    leverage_values_per_step: List[np.ndarray]
    terminal_S: np.ndarray
    terminal_Y: np.ndarray

    def leverage_surface(self, t: float, S_arr: np.ndarray) -> np.ndarray:
        """
        Evaluate calibrated leverage function L(t, S) at time t and spot prices S_arr.
        Finds nearest calibrated time-step index.
        """
        if len(self.time_grid) <= 1:
            return self.leverage_functions[0](S_arr)
        dt = self.time_grid[1] - self.time_grid[0] if len(self.time_grid) > 1 else 0.01
        idx = int(np.round(t / dt))
        idx = min(max(idx, 0), len(self.leverage_functions) - 1)
        return self.leverage_functions[idx](S_arr)


class Calibrator:
    """
    Calibrates the local volatility component L(t, S) of a Stochastic Local Volatility model.

    Uses the McKean-Vlasov particle method (Guyon & Henry-Labordère 2012):
    1) Initialize N particles at t = 0.
    2) At step t_k, estimate E[a_k² | S_k = s] on a grid via Nadaraya-Watson regression.
    3) Construct step leverage L(t_k, S) = sigma_Dup(t_k, S) / sqrt(E[a_k² | S_k = S]).
    4) Advance particles to t_{k+1} using L(t_k, S).
    5) Store step-by-step leverage functions for pricing.
    """

    def __init__(
        self,
        model_params: ModelParameters,
        market_vol_surface: ImpliedVolatilitySurface,
        kernel_func: Callable = quartic_kernel,
        bandwidth_scale: float = 0.15,
        quantile_range: tuple[float, float] = (0.002, 0.998),
        n_knots: int = 60,
    ):
        self.model_params = model_params
        self.market_vol_surface = market_vol_surface
        self.kernel_func = kernel_func
        self.bandwidth_scale = bandwidth_scale
        self.quantile_range = quantile_range
        self.n_knots = n_knots

    def calibrate(self, N: int = 50000, dt: float = 0.02, T: Optional[float] = None) -> CalibrationResult:
        """
        Perform step-by-step particle calibration from t = 0 to T.

        Parameters
        ----------
        N : int
            Number of particles for calibration.
        dt : float
            Time step size in years.
        T : float, optional
            Maturity horizon (defaults to model_params.T).

        Returns
        -------
        CalibrationResult
            Contains time grid, step leverage functions, and terminal particles.
        """
        p = self.model_params
        T_target = T if T is not None else p.T
        n_steps = int(np.round(T_target / dt))
        time_grid = np.linspace(0, T_target, n_steps + 1)

        S = np.full(N, p.S0, dtype=float)
        Y = np.full(N, p.Y0, dtype=float)
        model = OrnsteinUhlenbeckSV(p)

        leverage_funcs: List[Callable] = []
        knots_list: List[np.ndarray] = []
        lev_vals_list: List[np.ndarray] = []

        q_low, q_high = self.quantile_range

        for k in range(n_steps):
            t_k = time_grid[k]
            a_k = p.sigma0 * np.exp(Y)
            a_k_sq = a_k**2

            # Define adaptive knots on S distribution quantiles
            s_min = np.quantile(S, q_low)
            s_max = np.quantile(S, q_high)
            knots_S = np.linspace(s_min, s_max, self.n_knots)

            # Adaptive bandwidth for Nadaraya-Watson regression
            sigma_atm = self.market_vol_surface.get_volatility(np.array([p.S0]), t_k)[0] if hasattr(self.market_vol_surface, 'get_volatility') else 0.20
            bw = self.bandwidth_scale * p.S0 * np.sqrt(max(t_k, 0.05)) * N**(-0.2)

            # NW kernel regression: E[a_t² | S_t = s]
            expected_a_sq = nadaraya_watson_regression(S, a_k_sq, knots_S, bw, self.kernel_func)
            expected_a_sq = np.maximum(expected_a_sq, 1e-6)

            # Dupire local vol target at time t_k
            t_dup_eval = max(t_k, 0.01)
            _, sigma_dup = self.market_vol_surface.to_dupire_local_vol(knots_S, t_dup_eval)

            # Leverage ratio L(t_k, S) = sigma_Dup / sqrt(E[a_k² | S_k])
            leverage_vals = sigma_dup / np.sqrt(expected_a_sq)
            leverage_vals = np.clip(leverage_vals, 0.05, 5.0)

            # Linear interpolation with flat extrapolation
            interp_func = interp1d(
                knots_S,
                leverage_vals,
                kind='linear',
                bounds_error=False,
                fill_value=(leverage_vals[0], leverage_vals[-1])
            )

            leverage_funcs.append(interp_func)
            knots_list.append(knots_S)
            lev_vals_list.append(leverage_vals)

            # Advance particles to step k+1 using step leverage L(t_k, S)
            Z1 = np.random.randn(N)
            Z2 = np.random.randn(N)

            l_particles = interp_func(S)
            Y = model.exact_OU_step(Y, dt, Z2)
            S = model.euler_S_step(S, Y, l_particles, dt, Z1, Z2)

        return CalibrationResult(
            time_grid=time_grid,
            leverage_functions=leverage_funcs,
            knots_per_step=knots_list,
            leverage_values_per_step=lev_vals_list,
            terminal_S=S,
            terminal_Y=Y,
        )

    def calibrate_leverage_at_time(self, t: float, N: int = 50000, dt: float = 0.02) -> CalibrationResult:
        """Alias for backward compatibility."""
        return self.calibrate(N=N, dt=dt, T=t)
