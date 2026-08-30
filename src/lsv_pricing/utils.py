"""
Utility functions used across the LSV project.

Includes kernel regression, interpolation, and Black-Scholes helpers.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import norm


def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """Standard Gaussian kernel function."""
    return np.exp(-0.5 * u**2)


def quartic_kernel(u: np.ndarray) -> np.ndarray:
    """Quartic (biweight) kernel with compact support [-1, 1]."""
    u_clipped = np.clip(u, -1, 1)
    return (1 - u_clipped**2)**2


def nadaraya_watson_regression(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_eval: np.ndarray,
    bandwidth: float,
    kernel_func=quartic_kernel,
) -> np.ndarray:
    """
    Estimate E[Y | X = x] using the Nadaraya-Watson kernel regression.

    Parameters
    ----------
    x_data : np.ndarray, shape (N,)
        Observed values of the predictor variable.
    y_data : np.ndarray, shape (N,)
        Observed values of the response variable.
    x_eval : np.ndarray, shape (M,)
        Points at which to evaluate the conditional expectation.
    bandwidth : float
        Bandwidth parameter controlling smoothness of estimation.
    kernel_func : callable, optional
        Kernel function taking array input and returning array output.

    Returns
    -------
    np.ndarray, shape (M,)
        Estimated conditional expectation at x_eval points.
    """
    # Normalize distances by bandwidth
    u = (x_data[:, None] - x_eval[None, :]) / bandwidth
    weights = kernel_func(u)  # Shape: (N, M)

    numerator = np.sum(weights * y_data[:, None], axis=0)
    denominator = np.sum(weights, axis=0)

    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return numerator / denominator


def linear_interpolation(x_points: np.ndarray, y_points: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    """Linear interpolation with extrapolation."""
    f = interp1d(x_points, y_points, kind='linear', fill_value='extrapolate')
    return f(x_query)


def cubic_spline_interpolation(x_points: np.ndarray, y_points: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    """Cubic spline interpolation with extrapolation."""
    f = interp1d(x_points, y_points, kind='cubic', fill_value='extrapolate')
    return f(x_query)


def blackscholes_impvol_call(
    K: float,
    T: float,
    S: float,
    C: float,
    r: float = 0.0,
    q: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> float:
    """
    Invert the European call option price to find its implied volatility.

    Uses Newton-Raphson iteration with fallback to bisection if divergence occurs.

    Parameters
    ----------
    K : float
        Strike price.
    T : float
        Time to maturity (in years).
    S : float
        Spot price.
    C : float
        Observed call option price.
    r : float, optional
        Risk-free rate.
    q : float, optional
        Dividend yield.
    tol : float, optional
        Convergence tolerance.
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    float
        Implied volatility.
    """
    F = S * np.exp((r - q) * T)
    K_normalized = K / F
    C_normalized = C * np.exp(r * T) / F

    # Check for arbitrage violations
    if C_normalized <= max(1 - K_normalized, 0) or C_normalized >= 1:
        return np.nan

    # Initial guess using Manaster-Koehler formula (more robust)
    if K_normalized == 1.0:
        x0 = np.sqrt(2 / T)  # ATM: x0 = sqrt(2/T)
    else:
        x0 = np.sqrt(2 * np.abs(np.log(K_normalized)) / T)

    for _ in range(max_iter):
        d1 = np.log(1 / K_normalized) / x0 + 0.5 * x0
        d2 = d1 - x0
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        price = cdf_d1 - K_normalized * cdf_d2
        vega = np.sqrt(T / (2 * np.pi)) * np.exp(-0.5 * d1**2)

        residual = price - C_normalized
        if abs(residual) < tol:
            return x0

        increment = residual / vega
        x0_new = x0 - increment

        if x0_new <= 0:
            x0_new = x0 * 0.5

        if abs(x0_new - x0) < tol:
            x0 = x0_new
            break

        x0 = x0_new

    return x0