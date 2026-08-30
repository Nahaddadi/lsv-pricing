# LSV Pricing: Stochastic Local Volatility Model

Monte Carlo pricing engine for **Stochastic Local Volatility (SLV)** models, combining Heston-style stochastic volatility with Dupire-style local volatility calibration.

## 📖 Overview

This project implements the McKean-Vlasov particle method for calibrating the **leverage function** $l(t,S)$ in the SLV model:

$$
dS_t = a_t \cdot l(t,S_t) \cdot S_t \, dW^{(1)}_t, \quad a_t = \sigma_0 e^{Y_t}
$$

### Key Features

- ✅ Particle-based McKean-Vlasov calibration using **Nadaraya-Watson kernel regression**
- ✅ Euler-Maruyama simulation with correlated Brownian drivers
- ✅ European call pricing under calibrated SLV dynamics
- ✅ Implied volatility smile computation and analysis

## 📚 Documentation

### Core Classes

| Class | Description |
|-------|-------------|
| `ModelParameters` | Container for model parameters |
| `Calibrator` | Calibrates leverage function $l(t,S)$ |
| `MonteCarloEngine` | Simulates paths and prices options |
| `ImpliedVolatilitySurface` | Market volatility surface representation |

## 🧮 Methodology

### Phase 1: Parameter Setup

- Define stochastic volatility parameters ($\kappa, \gamma, \rho$)
- Set market volatility surface $\sigma_{\text{Market}}(K,T)$

### Phase 2: Particle Simulation

1. Simulate $N$ paths of pure SV model using exact OU scheme for $Y_t$ and Euler scheme for $S_t$.

### Phase 3: NW Regression

2. At each time step $t$, regress the conditional variance $\mathbb{E}[a_t^2 \mid S_t = x] \approx \sum_i w_i(x) a_{t,i}^2$ with kernel weights:

$$
w_i(x) = \frac{K\left(\frac{x - S_{t,i}}{h}\right)}{\sum_j K\left(\frac{x - S_{t,j}}{h}\right)}
$$

### Phase 4: Leverage Construction

3. Compute the step leverage function:

$$
l(t,S) = \frac{\sigma_{\text{Market}}(t,S)}{\sqrt{\mathbb{E}[a_t^2 \mid S_t = S]}}
$$

### Phase 5: SLV Pricing

4. Re-simulate under full SLV dynamics incorporating the calibrated time-dependent leverage surface $l(t,S)$.
