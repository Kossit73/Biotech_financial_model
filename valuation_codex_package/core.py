"""Core valuation models and utilities for the Valuation Codex package.

This module keeps the original public surface while refactoring the internal
architecture to be easier to extend.  Each concept now has a focused class and
there are explicit extension points for valuation, scenario modelling, and
forecasting strategies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, List, MutableMapping, Optional, Protocol

import numpy as np
import pandas as pd


# =========================
# Configuration dataclasses
# =========================


@dataclass(slots=True)
class ModelConfig:
    """Global model assumptions that apply to the full portfolio."""

    first_year: int = 2024
    n_years: int = 25
    currency: str = "USD"
    tax_rate: float = 0.25
    discount_rate: float = 0.10
    ev_ebitda_multiple: float = 8.0
    working_capital_pct_sales: float = 0.08
    sales_ramp_factors: List[float] | None = None

    def __post_init__(self) -> None:
        if self.sales_ramp_factors is None:
            self.sales_ramp_factors = [0.20, 0.60, 1.00, 1.00, 1.00]

    @property
    def years(self) -> np.ndarray:
        return np.arange(self.first_year, self.first_year + self.n_years)


@dataclass(slots=True)
class ProductConfig:
    """Product-specific assumptions."""

    name: str
    stage: str
    success_prob: float
    include_in_consolidation: bool = True

    time_to_market: int = 3
    patent_years: int = 20
    preexisting_market: bool = False

    patent_revenue_target: float = 0.0
    post_patent_revenue_target: float = 0.0
    market_growth_patent: float = 0.005
    market_growth_post: float = 0.0

    cogs_patent: float = 0.30
    cogs_post: float = 0.50
    sales_marketing_pct: float = 0.15
    gna_pct: float = 0.10
    royalty_pct: float = 0.0

    rd_remaining_pre_launch: float = 0.0
    rd_annual_post_launch: float = 0.0

    capex_remaining_pre_launch: float = 0.0
    capex_annual_post_launch: float = 0.0

    rd_capitalization_ratio: float = 0.5
    rd_amort_years: int = 10
    capex_dep_years: int = 10


# ===========================
# Product financials builder
# ===========================


class ProductFinancialsBuilder:
    """Factory for generating deterministic product cash-flow schedules."""

    def _launch_year(self, cfg: ProductConfig, model_cfg: ModelConfig) -> int:
        if cfg.preexisting_market:
            return model_cfg.first_year
        return model_cfg.first_year + max(cfg.time_to_market, 0)

    def _patent_end_year(self, cfg: ProductConfig, model_cfg: ModelConfig) -> int:
        return self._launch_year(cfg, model_cfg) + cfg.patent_years - 1

    @staticmethod
    def _rolling_amortization(additions: pd.Series, life: int) -> pd.Series:
        years = additions.index
        amort = pd.Series(0.0, index=years)
        if life <= 0:
            return amort
        for i in range(len(years)):
            add = additions.iloc[i]
            if add == 0:
                continue
            annual = add / life
            end = min(i + life, len(years))
            amort.iloc[i:end] += annual
        return amort

    def build_revenue_series(self, cfg: ProductConfig, model_cfg: ModelConfig) -> pd.Series:
        years = model_cfg.years
        revenue = pd.Series(0.0, index=years, name=f"{cfg.name}_revenue")
        if not cfg.include_in_consolidation:
            return revenue

        launch_year = self._launch_year(cfg, model_cfg)
        patent_end = self._patent_end_year(cfg, model_cfg)

        for i, year in enumerate(years):
            if year < launch_year:
                continue
            years_since_launch = year - launch_year
            in_patent = year <= patent_end

            if in_patent:
                base_target = cfg.patent_revenue_target
                growth_rate = cfg.market_growth_patent
                growth_start = max(0, years_since_launch - len(model_cfg.sales_ramp_factors))
            else:
                base_target = cfg.post_patent_revenue_target
                growth_rate = cfg.market_growth_post
                growth_start = max(0, year - (patent_end + 1))

            ramp = (
                model_cfg.sales_ramp_factors[years_since_launch]
                if years_since_launch < len(model_cfg.sales_ramp_factors)
                else 1.0
            )
            target_with_growth = base_target * ((1 + growth_rate) ** growth_start)
            revenue.iloc[i] = ramp * target_with_growth

        return revenue

    def build_cashflow_table(self, cfg: ProductConfig, model_cfg: ModelConfig) -> pd.DataFrame:
        years = model_cfg.years
        df = pd.DataFrame(index=years)
        df["revenue"] = self.build_revenue_series(cfg, model_cfg)

        patent_end = self._patent_end_year(cfg, model_cfg)
        cogs = []
        for year, rev in df["revenue"].items():
            if rev == 0:
                cogs.append(0.0)
                continue
            pct = cfg.cogs_patent if year <= patent_end else cfg.cogs_post
            cogs.append(-pct * rev)
        df["cogs"] = cogs

        df["sales_marketing"] = -cfg.sales_marketing_pct * df["revenue"]
        df["gna"] = -cfg.gna_pct * df["revenue"]
        df["royalty"] = -cfg.royalty_pct * df["revenue"]

        rd_cash = pd.Series(0.0, index=years)
        if cfg.rd_remaining_pre_launch > 0 and not cfg.preexisting_market:
            pre_years = max(1, cfg.time_to_market)
            annual_pre = cfg.rd_remaining_pre_launch / pre_years
            rd_cash.iloc[:pre_years] -= annual_pre

        launch_year = self._launch_year(cfg, model_cfg)
        rd_cash.loc[model_cfg.years >= launch_year] -= cfg.rd_annual_post_launch
        df["rd_cash"] = rd_cash

        rd_cap_add = rd_cash * cfg.rd_capitalization_ratio
        rd_expensed_current = rd_cash * (1 - cfg.rd_capitalization_ratio)
        rd_amort = self._rolling_amortization(rd_cap_add, cfg.rd_amort_years)
        df["rd_cap_add"] = rd_cap_add
        df["rd_amort"] = rd_amort
        df["rd_expense_pnl"] = rd_expensed_current + rd_amort

        capex_cash = pd.Series(0.0, index=years)
        if cfg.capex_remaining_pre_launch > 0 and not cfg.preexisting_market:
            pre_years = max(1, cfg.time_to_market)
            annual_pre = cfg.capex_remaining_pre_launch / pre_years
            capex_cash.iloc[:pre_years] -= annual_pre

        capex_cash.loc[model_cfg.years >= launch_year] -= cfg.capex_annual_post_launch
        df["capex_cash"] = capex_cash

        depreciation = self._rolling_amortization(capex_cash, cfg.capex_dep_years)
        df["depreciation"] = depreciation

        df["ebit"] = (
            df["revenue"]
            + df["cogs"]
            + df["sales_marketing"]
            + df["gna"]
            + df["royalty"]
            + df["rd_expense_pnl"]
        )
        df["da"] = -(df["rd_amort"] + df["depreciation"])
        df["ebitda"] = df["ebit"] + df["da"]

        tax_rate = model_cfg.tax_rate
        df["tax"] = 0.0
        positive_ebit = df["ebit"] > 0
        df.loc[positive_ebit, "tax"] = -tax_rate * df.loc[positive_ebit, "ebit"]
        df["nopat"] = df["ebit"] + df["tax"]

        df["fcff"] = df["nopat"] + df["da"] + df["capex_cash"] + df["rd_cap_add"]
        return df


# ==========
# Core model
# ==========


class Product:
    """Represents a single product and delegates computations to the builder."""

    def __init__(
        self,
        config: ProductConfig,
        model_config: ModelConfig,
        *,
        builder: ProductFinancialsBuilder | None = None,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self._builder = builder or ProductFinancialsBuilder()

    def build_revenue_series(self) -> pd.Series:
        return self._builder.build_revenue_series(self.config, self.model_config)

    def build_cashflow_table(self) -> pd.DataFrame:
        return self._builder.build_cashflow_table(self.config, self.model_config)

    def build_probability_weighted_table(self) -> pd.DataFrame:
        df = self.build_cashflow_table().copy()
        p = self.config.success_prob
        for col in ["revenue", "ebit", "ebitda", "nopat", "fcff"]:
            df[col] = df[col] * p
        return df


class WorkingCapitalCalculator:
    """Utility for working-capital adjustments on consolidated cash flows."""

    def __init__(self, pct_sales: float) -> None:
        self.pct_sales = pct_sales

    def apply(self, revenue: pd.Series) -> pd.Series:
        wc = self.pct_sales * revenue
        wc_diff = wc.diff().fillna(wc)
        return -wc_diff


class Portfolio:
    """A collection of products governed by a single model configuration."""

    def __init__(self, products: List[Product], model_config: ModelConfig):
        self.products = products
        self.model_config = model_config

    def consolidated_table(self) -> Dict[str, Dict[str, pd.DataFrame] | pd.DataFrame]:
        years = self.model_config.years
        base_cols = [
            "revenue",
            "cogs",
            "sales_marketing",
            "gna",
            "royalty",
            "rd_cash",
            "rd_cap_add",
            "rd_amort",
            "rd_expense_pnl",
            "capex_cash",
            "depreciation",
            "ebit",
            "da",
            "ebitda",
            "tax",
            "nopat",
            "fcff",
        ]
        cons_df = pd.DataFrame(0.0, index=years, columns=base_cols)

        per_product: Dict[str, pd.DataFrame] = {}
        per_product_prob: Dict[str, pd.DataFrame] = {}

        for prod in self.products:
            cfg = prod.config
            if not cfg.include_in_consolidation:
                continue
            df = prod.build_cashflow_table()
            per_product[cfg.name] = df
            wdf = prod.build_probability_weighted_table()
            per_product_prob[cfg.name] = wdf
            cons_df = cons_df.add(wdf[base_cols], fill_value=0.0)

        wc_calc = WorkingCapitalCalculator(self.model_config.working_capital_pct_sales)
        cons_df["delta_wc"] = wc_calc.apply(cons_df["revenue"])
        cons_df["fcff_after_wc"] = cons_df["fcff"] + cons_df["delta_wc"]

        return {
            "per_product": per_product,
            "per_product_prob": per_product_prob,
            "consolidated": cons_df,
        }


# =================
# Valuation engine
# =================


@dataclass(slots=True)
class ValuationResult:
    portfolio: Portfolio
    rnpv: float
    dcf_table: pd.DataFrame
    consolidated: pd.DataFrame
    per_product: Dict[str, pd.DataFrame]
    per_product_prob: Dict[str, pd.DataFrame]


class ValuationMethod(Protocol):
    """Interface for all valuation strategies."""

    name: str

    def run(self, portfolio: Portfolio) -> ValuationResult:  # pragma: no cover - Protocol
        ...


class DiscountedCashFlowValuation:
    """Classic discounted cash-flow valuation with an EBITDA terminal value."""

    name = "discounted_cash_flow"

    @staticmethod
    def _discounted_cash_flows(fcff: pd.Series, discount_rate: float) -> pd.DataFrame:
        years = fcff.index.values
        t = np.arange(len(years))
        df = pd.DataFrame(index=years)
        df["t"] = t
        df["fcff"] = fcff.values
        df["discount_factor"] = 1.0 / ((1 + discount_rate) ** t)
        df["discounted_fcff"] = df["fcff"] * df["discount_factor"]
        return df

    def _add_terminal_value(
        self, dcf_df: pd.DataFrame, cons_df: pd.DataFrame, model_cfg: ModelConfig
    ) -> float:
        last_year = cons_df.index[-1]
        last_ebitda = cons_df.loc[last_year, "ebitda"]
        terminal_ev = model_cfg.ev_ebitda_multiple * last_ebitda

        t_last = dcf_df.loc[last_year, "t"]
        discount = (1 + model_cfg.discount_rate) ** t_last
        dcf_df.loc[last_year, "terminal_value"] = terminal_ev
        dcf_df.loc[last_year, "discounted_terminal_value"] = terminal_ev / discount
        return float(dcf_df["discounted_fcff"].sum() + dcf_df["discounted_terminal_value"].sum())

    def run(self, portfolio: Portfolio) -> ValuationResult:
        agg = portfolio.consolidated_table()
        cons = agg["consolidated"]
        model_cfg = portfolio.model_config
        dcf_df = self._discounted_cash_flows(cons["fcff_after_wc"], model_cfg.discount_rate)
        rnpv = self._add_terminal_value(dcf_df, cons, model_cfg)
        return ValuationResult(
            portfolio=portfolio,
            rnpv=rnpv,
            dcf_table=dcf_df,
            consolidated=cons,
            per_product=agg["per_product"],
            per_product_prob=agg["per_product_prob"],
        )


class ValuationEngine:
    """Thin orchestrator that delegates to a concrete valuation method."""

    def __init__(
        self,
        portfolio: Portfolio,
        method: ValuationMethod | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.method = method or DiscountedCashFlowValuation()

    def run(self) -> ValuationResult:
        return self.method.run(self.portfolio)


# ==================
# VC-style valuation
# ==================


@dataclass(slots=True)
class VCInputs:
    exit_year: int
    target_irr: float
    investor_ownership_at_exit: float
    new_money: float


class VCValuator:
    """Implements the classic VC method using the DCF results as inputs."""

    def __init__(self, valuation_result: ValuationResult) -> None:
        self.result = valuation_result
        self.model_config = valuation_result.portfolio.model_config

    def compute_exit_ev(self, exit_year: int, multiple: Optional[float] = None) -> float:
        if multiple is None:
            multiple = self.model_config.ev_ebitda_multiple
        cons = self.result.consolidated
        if exit_year not in cons.index:
            raise ValueError("Exit year not in consolidated index")
        exit_ebitda = cons.loc[exit_year, "ebitda"]
        return float(exit_ebitda * multiple)

    def vc_method(self, vc_inputs: VCInputs, exit_multiple: Optional[float] = None) -> Dict[str, float]:
        exit_ev = self.compute_exit_ev(vc_inputs.exit_year, exit_multiple)
        years_to_exit = vc_inputs.exit_year - self.model_config.first_year
        if years_to_exit <= 0:
            raise ValueError("Exit year must be after first model year")

        investor_exit_value = exit_ev * vc_inputs.investor_ownership_at_exit
        investor_pv_required = investor_exit_value / ((1 + vc_inputs.target_irr) ** years_to_exit)
        implied_post_money = investor_pv_required / vc_inputs.investor_ownership_at_exit
        implied_pre_money = implied_post_money - vc_inputs.new_money
        investor_irr_actual = (investor_exit_value / vc_inputs.new_money) ** (1 / years_to_exit) - 1

        return {
            "exit_enterprise_value": exit_ev,
            "investor_exit_value": investor_exit_value,
            "investor_pv_required": investor_pv_required,
            "implied_post_money": implied_post_money,
            "implied_pre_money": implied_pre_money,
            "investor_irr_if_pay_new_money": investor_irr_actual,
        }


# =========================
# Scenario & stress testing
# =========================


class ScenarioAdjustment(Protocol):
    """Extension point that can adjust model-wide or per-product assumptions."""

    name: str

    def adjust_model_config(self, model_config: ModelConfig) -> ModelConfig:  # pragma: no cover - Protocol
        ...

    def adjust_product_config(
        self, product_config: ProductConfig
    ) -> ProductConfig:  # pragma: no cover - Protocol
        ...


@dataclass(slots=True)
class Scenario(ScenarioAdjustment):
    """Multiplicative scenario suitable for simple stress tests."""

    name: str
    revenue_multiplier: float = 1.0
    cost_multiplier: float = 1.0
    discount_rate_shift: float = 0.0
    success_prob_multiplier: float = 1.0

    def adjust_model_config(self, model_config: ModelConfig) -> ModelConfig:
        cfg = replace(model_config)
        cfg.discount_rate += self.discount_rate_shift
        return cfg

    def adjust_product_config(self, product_config: ProductConfig) -> ProductConfig:
        cfg = replace(product_config)
        cfg.patent_revenue_target *= self.revenue_multiplier
        cfg.post_patent_revenue_target *= self.revenue_multiplier
        cfg.cogs_patent *= self.cost_multiplier
        cfg.cogs_post *= self.cost_multiplier
        cfg.success_prob = max(0.0, min(1.0, cfg.success_prob * self.success_prob_multiplier))
        return cfg


class ScenarioEngine:
    """Applies scenario adjustments and re-values the resulting portfolios."""

    def __init__(self, base_portfolio: Portfolio):
        self.base_portfolio = base_portfolio

    def _apply_adjustment(self, adjustment: ScenarioAdjustment) -> Portfolio:
        new_model_cfg = adjustment.adjust_model_config(self.base_portfolio.model_config)
        new_products: List[Product] = []
        for prod in self.base_portfolio.products:
            new_cfg = adjustment.adjust_product_config(prod.config)
            new_products.append(Product(new_cfg, new_model_cfg))
        return Portfolio(new_products, new_model_cfg)

    def run_scenarios(
        self,
        scenarios: Iterable[ScenarioAdjustment],
        ebitda_year_offset: int = 0,
    ) -> pd.DataFrame:
        rows = []

        base_val = ValuationEngine(self.base_portfolio).run()
        base_cons = base_val.consolidated
        base_year = self.base_portfolio.model_config.first_year + ebitda_year_offset
        rows.append(
            {
                "scenario": "Base",
                "discount_rate": self.base_portfolio.model_config.discount_rate,
                "rnpv": base_val.rnpv,
                "ebitda_year": base_year,
                "ebitda_value": float(base_cons.loc[base_year, "ebitda"]),
            }
        )

        for sc in scenarios:
            port_sc = self._apply_adjustment(sc)
            val = ValuationEngine(port_sc).run()
            cons = val.consolidated
            year = port_sc.model_config.first_year + ebitda_year_offset
            rows.append(
                {
                    "scenario": sc.name,
                    "discount_rate": port_sc.model_config.discount_rate,
                    "rnpv": val.rnpv,
                    "ebitda_year": year,
                    "ebitda_value": float(cons.loc[year, "ebitda"]),
                }
            )

        return pd.DataFrame(rows)


# ==========================
# Monte Carlo & risk engine
# ==========================


class MonteCarloEngine:
    """Generates a distribution of rNPVs using simple multiplicative shocks."""

    def __init__(self, base_portfolio: Portfolio):
        self.base_portfolio = base_portfolio

    def simulate(
        self,
        n_sims: int = 1000,
        revenue_sigma: float = 0.1,
        cost_sigma: float = 0.05,
        random_seed: Optional[int] = None,
    ) -> pd.Series:
        rng = np.random.default_rng(random_seed)
        vals = []

        for _ in range(n_sims):
            model_cfg = self.base_portfolio.model_config
            new_products: List[Product] = []
            rev_scale = rng.normal(1.0, revenue_sigma)
            cogs_scale = rng.normal(1.0, cost_sigma)

            for prod in self.base_portfolio.products:
                cfg_dict = asdict(prod.config)
                cfg_dict["patent_revenue_target"] *= rev_scale
                cfg_dict["post_patent_revenue_target"] *= rev_scale
                cfg_dict["cogs_patent"] *= cogs_scale
                cfg_dict["cogs_post"] *= cogs_scale
                new_cfg = ProductConfig(**cfg_dict)
                new_products.append(Product(new_cfg, model_cfg))

            sim_portfolio = Portfolio(new_products, model_cfg)
            val = ValuationEngine(sim_portfolio).run()
            vals.append(val.rnpv)

        return pd.Series(vals, name="rnpv_sim")

    @staticmethod
    def value_at_risk(simulated_rnpv: pd.Series, alpha: float = 0.95) -> float:
        return float(simulated_rnpv.quantile(1 - alpha))

    @staticmethod
    def conditional_value_at_risk(simulated_rnpv: pd.Series, alpha: float = 0.95) -> float:
        var = simulated_rnpv.quantile(1 - alpha)
        tail = simulated_rnpv[simulated_rnpv <= var]
        if len(tail) == 0:
            return float(var)
        return float(tail.mean())


# ==========================
# Forecast engine & bridge
# ==========================


class BaseForecastModel(Protocol):
    """Interface for time-series forecasters."""

    name: str

    def forecast(self, *args, **kwargs):  # pragma: no cover - Protocol
        ...


class ARIMAForecastModel:
    name = "arima"

    def forecast(
        self,
        series: pd.Series,
        *,
        order: tuple[int, int, int] = (1, 1, 1),
        steps: Optional[int] = None,
        model_config: ModelConfig,
    ) -> pd.Series:
        from statsmodels.tsa.arima.model import ARIMA

        steps = steps or model_config.n_years
        model = ARIMA(series, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=steps)
        forecast.name = f"{series.name}_arima_forecast"
        return forecast


class ProphetForecastModel:
    name = "prophet"

    def forecast(
        self,
        df: pd.DataFrame,
        *,
        periods: Optional[int] = None,
        freq: str = "Y",
        model_config: ModelConfig,
    ) -> pd.DataFrame:
        from prophet import Prophet

        periods = periods or model_config.n_years
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=periods, freq=freq)
        forecast = m.predict(future)
        return forecast


class LSTMForecastModel:
    name = "lstm"

    def forecast(
        self,
        series: pd.Series,
        *,
        lookback: int = 12,
        steps_ahead: Optional[int] = None,
        epochs: int = 50,
        batch_size: int = 16,
        model_config: ModelConfig,
    ) -> np.ndarray:
        import numpy as np
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.models import Sequential

        steps_ahead = steps_ahead or model_config.n_years
        values = series.values.astype("float32")
        X, y = [], []
        for i in range(len(values) - lookback):
            X.append(values[i : i + lookback])
            y.append(values[i + lookback])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(32, input_shape=(lookback, 1)))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

        history = values[-lookback:].copy()
        forecasts = []
        for _ in range(steps_ahead):
            x_input = history[-lookback:].reshape((1, lookback, 1))
            yhat = model.predict(x_input, verbose=0)[0, 0]
            forecasts.append(yhat)
            history = np.append(history, yhat)

        return np.array(forecasts)


class ForecastEngine:
    """Registry of forecast models with convenience wrappers for built-ins."""

    def __init__(
        self,
        model_config: ModelConfig,
        models: Optional[MutableMapping[str, BaseForecastModel]] = None,
    ) -> None:
        self.model_config = model_config
        self._models: Dict[str, BaseForecastModel] = models or {
            "arima": ARIMAForecastModel(),
            "prophet": ProphetForecastModel(),
            "lstm": LSTMForecastModel(),
        }

    def register_model(self, model: BaseForecastModel) -> None:
        self._models[model.name] = model

    def forecast(self, model_name: str, *args, **kwargs):
        if model_name not in self._models:
            raise KeyError(f"Unknown forecast model '{model_name}'")
        kwargs.setdefault("model_config", self.model_config)
        return self._models[model_name].forecast(*args, **kwargs)

    # Compatibility helpers -------------------------------------------------
    def forecast_arima(
        self, series: pd.Series, order: tuple[int, int, int] = (1, 1, 1), steps: Optional[int] = None
    ) -> pd.Series:
        return self.forecast("arima", series, order=order, steps=steps)

    def forecast_prophet(self, df: pd.DataFrame, periods: Optional[int] = None, freq: str = "Y") -> pd.DataFrame:
        return self.forecast("prophet", df, periods=periods, freq=freq)

    def forecast_lstm(
        self,
        series: pd.Series,
        lookback: int = 12,
        steps_ahead: Optional[int] = None,
        epochs: int = 50,
        batch_size: int = 16,
    ) -> np.ndarray:
        return self.forecast(
            "lstm",
            series,
            lookback=lookback,
            steps_ahead=steps_ahead,
            epochs=epochs,
            batch_size=batch_size,
        )


class ForecastScenarioBridge:
    """Turns Prophet price forecasts into portfolio scenarios."""

    def __init__(self, base_portfolio: Portfolio, forecast_engine: ForecastEngine):
        self.base_portfolio = base_portfolio
        self.forecast_engine = forecast_engine
        self.model_config = base_portfolio.model_config

    @staticmethod
    def _avg_ratio(series: pd.Series, base_value: float, cap: float = 3.0) -> float:
        if base_value <= 0:
            return 1.0
        ratio = float(series.mean() / base_value)
        return max(0.0, min(ratio, cap))

    def build_price_scenarios_from_prophet(
        self,
        hist_df: pd.DataFrame,
        periods: Optional[int] = None,
        freq: str = "Y",
        pessimistic_discount_uplift: float = 0.02,
        optimistic_discount_reduction: float = 0.01,
        cost_sensitivity: float = 0.05,
        ebitda_year_offset: int = 0,
    ) -> pd.DataFrame:
        forecast = self.forecast_engine.forecast_prophet(hist_df, periods=periods, freq=freq)
        last_hist_date = hist_df["ds"].max()
        future_fc = forecast[forecast["ds"] > last_hist_date].copy()
        future_fc = future_fc.head(self.model_config.n_years)

        base_series = future_fc["yhat"]
        pess_series = future_fc["yhat_lower"]
        opt_series = future_fc["yhat_upper"]

        base_price = float(hist_df["y"].iloc[-1])
        base_ratio = self._avg_ratio(base_series, base_price)
        pess_ratio = self._avg_ratio(pess_series, base_price)
        opt_ratio = self._avg_ratio(opt_series, base_price)

        scenarios: List[ScenarioAdjustment] = [
            Scenario(
                name="Prophet Base",
                revenue_multiplier=base_ratio,
                cost_multiplier=1.0,
                discount_rate_shift=0.0,
                success_prob_multiplier=1.0,
            ),
            Scenario(
                name="Prophet Pessimistic",
                revenue_multiplier=pess_ratio,
                cost_multiplier=1.0 + cost_sensitivity,
                discount_rate_shift=pessimistic_discount_uplift,
                success_prob_multiplier=0.95,
            ),
            Scenario(
                name="Prophet Optimistic",
                revenue_multiplier=opt_ratio,
                cost_multiplier=max(0.0, 1.0 - cost_sensitivity),
                discount_rate_shift=-optimistic_discount_reduction,
                success_prob_multiplier=1.05,
            ),
        ]

        scen_engine = ScenarioEngine(self.base_portfolio)
        return scen_engine.run_scenarios(scenarios=scenarios, ebitda_year_offset=ebitda_year_offset)


__all__ = [
    "ModelConfig",
    "ProductConfig",
    "Product",
    "Portfolio",
    "ValuationEngine",
    "ValuationResult",
    "VCInputs",
    "VCValuator",
    "Scenario",
    "ScenarioAdjustment",
    "ScenarioEngine",
    "MonteCarloEngine",
    "ForecastEngine",
    "ForecastScenarioBridge",
    "DiscountedCashFlowValuation",
]
