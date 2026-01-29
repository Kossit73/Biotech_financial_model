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

## How the model works

1. **Set global assumptions** with `ModelConfig`, such as the modelling horizon, discount rate, tax rate, and working-capital policy.  Each product inherits these timeline settings so the entire portfolio shares the same calendar.
2. **Describe each asset** with a `ProductConfig`.  Inputs cover launch timing, patent duration, revenue targets (patent vs. post-patent), cost structure, R&D/CAPEX cash needs, and probability of success.  The companion `Product` object converts those assumptions into detailed revenue, expense, and cash-flow schedules.
3. **Aggregate the products** through `Portfolio.consolidated_table()`, which rolls up the probability-weighted financial statements, applies working-capital changes, and produces consolidated free cash flow available to the firm.
4. **Value the portfolio** using `ValuationEngine`.  The engine discounts the consolidated FCFF stream at the configured rate, appends a terminal EV/EBITDA multiple, and reports the resulting rNPV alongside the underlying DCF table and per-product statements.
5. **Layer venture-style analyses** with `VCInputs` and `VCValuator`.  Using the EBITDA in the target exit year (and either a fixed or data-driven multiple), the VC method back-solves the implied pre/post-money valuation for a given investor IRR and ownership target.
6. **Stress test with scenarios** by defining `Scenario` adjustments (revenue/cost/discount/probability shifts) and passing them to `ScenarioEngine.run_scenarios()`.  This revalues the portfolio under each shock and reports rNPV and EBITDA impacts for quick comparison.
7. **Assess risk via Monte Carlo** using `MonteCarloEngine.simulate()`, which perturbs revenue and cost drivers across many draws to approximate the rNPV distribution and compute metrics such as Value at Risk and Conditional VaR.
8. **Incorporate forecasts** with `ForecastEngine` (ARIMA/Prophet/LSTM) and convert those views into scenario inputs through `ForecastScenarioBridge`, enabling Prophet-derived base/pessimistic/optimistic paths to feed directly into the scenario engine.

The accompanying `streamlit_app.py` surfaces these workflows in a browser-based UI featuring dedicated tabs for configuration, financial statements, dashboards, advanced analytics, scenarios, and VC valuation so that non-technical users can operate the model interactively.

## Recommended improvements

If you want to increase realism and analytical depth, consider extending the model with:

- **Stage-gated probability curves** that evolve by development phase (e.g., discovery → preclinical → Phase I/II/III) instead of a single static probability. This can be paired with milestone costs to better reflect attrition over time.
- **Tax loss carryforwards and net operating loss (NOL) utilization**, so early negative EBIT carries forward and reduces future cash taxes rather than assuming taxes only when EBIT is positive.
- **Time-varying discount rates** (or risk-adjusted rates per product), especially to reflect de-risking as assets progress through phases.
- **Correlated Monte Carlo shocks** (e.g., shared market or pricing shocks) to avoid treating each product as independent in portfolio simulations.
- **Explicit dilution and financing rounds** to capture ownership changes, bridge financing, and the impact on VC method outputs.
- **Regional launch curves** (US/EU/ROW) with staggered timing, differing price/reimbursement assumptions, and localized market growth.
- **Working-capital and inventory dynamics** by product segment (especially for therapies with long manufacturing lead times).
- **Scenario libraries** for regulatory, reimbursement, and competitive events (e.g., earlier generic entry, biosimilar penetration, pricing compression).
