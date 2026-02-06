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

## Summary

Replaced the static pipeline stage list with a dropdown selector that lets users choose a stage template and see the selected item in the Model configuration tab.

## Audit & recommendations (20-year biotech financial analyst perspective)

Below is an audit of the current model design plus recommendations to make it enterprise-ready for biotech users.

### What the model currently does well

- **Portfolio-based rNPV**: The core engine builds probability-weighted product cash flows and aggregates them into a portfolio rNPV, which aligns with standard biotech valuation practice.  
- **Clear phase timing and patent life**: Launch timing and patent expiry drive the revenue lifecycle, matching common product lifecycle assumptions.  
- **Scenario + Monte Carlo tools**: Built-in deterministic scenarios and stochastic simulations help frame downside risk and valuation dispersion.  
- **VC method support**: The model provides a venture-style back-solve to align with early-stage financing expectations.

### Key gaps vs. industry-grade biotech models

1. **Phase-specific probability curves**: The current model applies a single success probability per asset, but biotech valuations typically use stage-transition probabilities (e.g., Phase I → II → III → Approval) with time-varying risk adjustments.
2. **Milestone and royalty structures**: Standard biotech deal economics include upfronts, development/regulatory milestones, sales milestones, and tiered royalties. These are not explicitly modeled.
3. **Explicit working-capital drivers**: The working-capital logic is a flat % of sales. Industry models typically use DSO/DIO/DPO to reflect launch ramps and inventory build.
4. **Granular operating expense build**: COGS, SG&A, and R&D are modeled as static percentages. Mature models break expenses into headcount, trial costs, manufacturing scale-up, and post-approval lifecycle management.
5. **Tax treatment and NOLs**: Loss carryforwards and jurisdiction-specific tax rates materially impact early-stage biotech valuations and should be modeled explicitly.
6. **Terminal value methodology**: A single EV/EBITDA multiple is used. Late-stage biotech models often layer alternative terminal methods (perpetuity growth, exit multiple, or patent cliff adjustments).
7. **Regulatory and market-access timing**: The model uses time-to-market, but does not explicitly model approval timelines, payer adoption curves, or pricing pressure post-launch.
8. **Scenario calibration to real benchmarks**: Scenarios are currently user-defined. Industry adoption improves when base/upside/downside assumptions are tied to comparables or historical launch cohorts.

### Recommended enhancements to make the model enterprise-grade

**1) Risk-adjusted pipeline with stage transitions**
- Add per-stage probability-of-success curves (Discovery, Preclinical, Phase I/II/III, Filing, Approval).
- Convert the single `success_prob` into a stage-based matrix so valuation discounts apply over time instead of as a flat factor.

**2) Deal economics module**
- Add configurable upstream/downstream licensing: upfronts, milestones (dev/reg/commercial), and tiered royalties.
- Support co-promotion / co-development splits and profit-sharing structures.

**3) Cash flow mechanics upgrade**
- Replace fixed working-capital % with DSO/DIO/DPO driven cash conversion cycle.
- Introduce NOL carryforwards and tax shields with jurisdiction toggles.

**4) Revenue model fidelity**
- Add patient/market sizing module with treated population, pricing, and penetration curves.
- Allow launch ramp shapes beyond the current fixed factors (S-curve, logarithmic, or user-imported profiles).

**5) Expense model fidelity**
- Create line-item R&D (trial costs by phase), manufacturing scale-up, and post-market study costs.
- Split SG&A into fixed + variable components tied to revenue scale and region footprint.

**6) Risk + sensitivity reporting**
- Output tornado charts and probabilistic distributions for key drivers (price, penetration, success probability, peak year).
- Publish “value of information” insights: which assumptions drive most of the rNPV variance.

**7) Benchmarking & sanity checks**
- Add a benchmarking layer that compares implied peak sales, margins, and R&D spend vs. published biotech peers.
- Flag outliers in assumptions (e.g., Phase II products with 80% PoS, or margins above typical biopharma ranges).

**8) Audit trail & governance**
- Add assumption provenance (date, source, owner) for each input.
- Provide exportable audit reports to meet institutional diligence requirements.

### Governance workflow for stage defaults

To keep stage-to-schedule defaults consistent across the organization while allowing product-specific overrides:

1. **Central finance owns the stage mapping table** (company-wide defaults for durations, success curves, and R&D timing).
2. **Product teams adjust only the product row overrides** (e.g., unique ramp length or R&D spend) when justified.
3. **Quarterly governance review**: toggle the “override existing values” control only during formal refresh cycles to push new company-wide assumptions.
4. **Audit discipline**: record who changed mapping defaults and why (e.g., clinical benchmark updates), and export the mapping table as part of valuation packages.

### How to edit the Stage-to-schedule mapping in the app

1. Open the **Stage-to-schedule mapping** expander in the Streamlit sidebar.
2. Update the **Mapping updated by** field to capture the owner for the audit trail.
3. Edit the table directly (stage durations, success probabilities, R&D amounts, transition weights, and milestone amounts).
4. Use **Auto-apply stage defaults to product assumptions** to push updates into product rows.
5. Only enable **Override existing values** during formal refresh cycles when you want to overwrite any customized product values.
6. Review the **Mapping audit trail** expander to confirm changes were logged.

### Pipeline stage templates (aligned workflow)

Use the standardized biotech development sequence below when defining assets or scenarios to keep the model consistent across teams:

- Discovery → Preclinical → Phase I → Phase II → Phase III → Approval → Commercial

**Stage templates**

- Discovery
- Preclinical
- Phase I
- Phase II
- Phase III
- Approval
- Commercial
