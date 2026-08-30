"""Tests for the LSV pricing package."""
import numpy as np
import pytest

from lsv_pricing import (
    ModelParameters,
    Calibrator,
    ImpliedVolatilitySurface,
    MonteCarloEngine,
)


@pytest.fixture
def model_params():
    return ModelParameters(
        S0=100, T=1, sigma0=0.15, Y0=0, kappa=1, gamma=0.5, rho=-0.5
    )


@pytest.fixture
def market_vol():
    return ImpliedVolatilitySurface(sigma_imp=0.15)


def test_calibrator_returns_callable(model_params, market_vol):
    calibrator = Calibrator(model_params, market_vol)
    result = calibrator.calibrate_leverage_at_time(t=1.0, N=1000)
    assert callable(result.leverage_function)
    assert len(result.S_grid) == calibrator.n_knots


def test_monte_carlo_engine_runs(model_params, market_vol):
    calibrator = Calibrator(model_params, market_vol)
    result = calibrator.calibrate_leverage_at_time(t=1.0, N=1000)

    engine = MonteCarloEngine(model_params)
    vols = engine.implied_vol_smile(
        lambda t, S: result.leverage_function(S),
        strikes=np.array([90, 100, 110]),
        N=1000,
    )
    assert len(vols) == 3
    assert np.all(vols > 0)