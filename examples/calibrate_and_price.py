"""
Tutorial example: calibrate and price under a Stochastic Local Volatility model.

Steps:
1. Define model and market parameters.
2. Calibrate the leverage function l(t,S) from simulated SV paths.
3. Price European call options using the calibrated SLV model.
4. Plot the resulting implied volatility smile vs. market.
"""

import numpy as np
import matplotlib.pyplot as plt

from lsv_pricing import (
    ModelParameters,
    Calibrator,
    ImpliedVolatilitySurface,
    MonteCarloEngine,
)


# -------------------------
# 1. Define parameters
# -------------------------
model_params = ModelParameters(
    S0=100, T=1, sigma0=0.15, Y0=0, kappa=1, gamma=0.5, rho=-0.5
)
market_vol = ImpliedVolatilitySurface(sigma_imp=0.15)

# -------------------------
# 2. Calibrate leverage function
# -------------------------
print("Calibrating leverage function...")
calibrator = Calibrator(model_params, market_vol)
result = calibrator.calibrate_leverage_at_time(t=1.0, N=10000)

plt.figure(figsize=(8, 5))
plt.plot(result.S_grid, result.leverage_values, label="l(t=1, S)")
plt.xlabel("Spot Price S")
plt.ylabel("Leverage Function l(t,S)")
plt.title("Calibrated Leverage Function")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("leverage_function.png", dpi=150)
plt.show()

# -------------------------
# 3. Price options using calibrated SLV model
# -------------------------
print("Pricing options under SLV model...")
engine = MonteCarloEngine(model_params)

strikes = np.linspace(70, 140, 10)
vols = engine.implied_vol_smile(
    lambda t, S: result.leverage_function(S),
    strikes=strikes,
    N=100000,
    seed=42,
)

# -------------------------
# 4. Compare to market
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(strikes, vols, "o-", label="SLV Implied Vol")
plt.axhline(y=market_vol.sigma_imp, color="r", linestyle="--", label="Market σ_imp = 15%")
plt.xlabel("Strike K")
plt.ylabel("Implied Volatility")
plt.title("SLV Implied Volatility Smile vs Market")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("volatility_smile.png", dpi=150)
plt.show()

print("Done!")