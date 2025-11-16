# Biotech Financial Model

This repository packages a biotech/agro valuation engine as an installable Python module named `valuation_codex_package`.

## Features

The package exposes an object-oriented modelling toolkit that covers:

- `ModelConfig` / `ProductConfig`: Declarative configuration dataclasses for portfolio assumptions.
- `Product` / `Portfolio`: Cash-flow construction utilities that produce probability-weighted results.
- `ValuationEngine`: Deterministic discounted cash flow engine with terminal value support.
- `VCInputs` / `VCValuator`: Back-of-the-envelope venture capital method helper.
- `ScenarioEngine`, `Scenario`: Deterministic stress testing via revenue, cost, discount-rate, or probability shocks.
- `MonteCarloEngine`: Randomized revenue/cost shocks to approximate risk via VaR and CVaR.
- `ForecastEngine`: ARIMA, Prophet, and LSTM helpers for forward price/metric projections.
- `ForecastScenarioBridge`: Translates Prophet forecasts into `Scenario` objects for quick stress testing.

## Usage

```python
from valuation_codex_package import (
    ModelConfig, ProductConfig, Product, Portfolio,
    ValuationEngine, Scenario, ScenarioEngine,
    MonteCarloEngine, ForecastEngine, ForecastScenarioBridge,
    VCInputs, VCValuator,
)
```

Combine configuration objects with the modelling helpers to build forecasts, run DCF or VC-style valuations, and perform sensitivity analysis.
