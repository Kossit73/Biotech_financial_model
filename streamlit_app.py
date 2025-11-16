"""Streamlit UI for the Valuation Codex biotech financial model."""

from __future__ import annotations

from dataclasses import fields
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from valuation_codex_package import (
    ModelConfig,
    Portfolio,
    Product,
    ProductConfig,
    Scenario,
    ScenarioEngine,
    VCInputs,
    VCValuator,
    ValuationEngine,
    MonteCarloEngine,
)


STAGE_OPTIONS = [
    "Discovery",
    "Preclinical",
    "Phase I",
    "Phase II",
    "Phase III",
    "Commercial",
]


def _default_products() -> pd.DataFrame:
    """Seed table with two representative products."""

    data = [
        {
            "name": "AgSeed-101",
            "stage": "Phase II",
            "success_prob": 0.35,
            "include_in_consolidation": True,
            "time_to_market": 4,
            "patent_years": 15,
            "patent_revenue_target": 120_000_000,
            "post_patent_revenue_target": 60_000_000,
            "market_growth_patent": 0.04,
            "market_growth_post": 0.0,
            "cogs_patent": 0.32,
            "cogs_post": 0.5,
            "sales_marketing_pct": 0.18,
            "gna_pct": 0.12,
            "rd_remaining_pre_launch": 180_000_000,
            "rd_annual_post_launch": 12_000_000,
            "capex_remaining_pre_launch": 55_000_000,
            "capex_annual_post_launch": 6_500_000,
        },
        {
            "name": "BioYield-Plus",
            "stage": "Phase III",
            "success_prob": 0.55,
            "include_in_consolidation": True,
            "time_to_market": 2,
            "patent_years": 17,
            "patent_revenue_target": 200_000_000,
            "post_patent_revenue_target": 95_000_000,
            "market_growth_patent": 0.03,
            "market_growth_post": 0.01,
            "cogs_patent": 0.28,
            "cogs_post": 0.45,
            "sales_marketing_pct": 0.16,
            "gna_pct": 0.1,
            "rd_remaining_pre_launch": 90_000_000,
            "rd_annual_post_launch": 8_000_000,
            "capex_remaining_pre_launch": 35_000_000,
            "capex_annual_post_launch": 4_500_000,
        },
    ]
    return pd.DataFrame(data)


def _default_ramp_schedule() -> pd.DataFrame:
    """Return the seed schedule for global sales ramp factors."""

    default_ramp = [0.2, 0.6, 1.0, 1.0, 1.0]
    data = {
        "Year offset": list(range(len(default_ramp))),
        "Ramp factor": default_ramp,
    }
    return pd.DataFrame(data)


def _render_schedule_editor(title: str, session_key: str) -> pd.DataFrame:
    """Render a reusable schedule editor with manual controls.

    The widget exposes explicit Edit / Add Row / Remove Row controls in addition to a
    "Yearly Increment Helper" that can seed values from a starting point.
    """

    if session_key not in st.session_state:
        st.session_state[session_key] = _default_ramp_schedule().copy()

    schedule_df: pd.DataFrame = st.session_state[session_key]
    st.markdown(f"**{title}**")
    toolbar_cols = st.columns(4)
    edit_mode = toolbar_cols[0].toggle("Edit", value=True, key=f"{session_key}_edit")

    if toolbar_cols[1].button("Add Row", key=f"{session_key}_add"):
        next_year = int(schedule_df["Year offset"].max() + 1) if not schedule_df.empty else 0
        last_value = (
            float(schedule_df["Ramp factor"].iloc[-1]) if not schedule_df.empty else 1.0
        )
        schedule_df.loc[len(schedule_df)] = [next_year, last_value]
        st.session_state[session_key] = schedule_df

    if toolbar_cols[2].button("Remove Row", key=f"{session_key}_remove") and not schedule_df.empty:
        schedule_df = schedule_df.iloc[:-1]
        st.session_state[session_key] = schedule_df

    with toolbar_cols[3]:
        with st.expander("Yearly Increment Helper"):
            start_year = st.number_input(
                "Start year offset", min_value=0, value=0, key=f"{session_key}_start"
            )
            start_value = st.number_input(
                "Starting value", min_value=0.0, value=0.2, step=0.05, key=f"{session_key}_value"
            )
            increment = st.number_input(
                "Increment per year", value=0.2, step=0.05, key=f"{session_key}_increment"
            )
            n_periods = st.number_input(
                "Number of periods", min_value=1, max_value=40, value=5, key=f"{session_key}_periods"
            )
            if st.button("Apply helper", key=f"{session_key}_apply"):
                rows = []
                for i in range(int(n_periods)):
                    rows.append(
                        {
                            "Year offset": int(start_year + i),
                            "Ramp factor": float(start_value + increment * i),
                        }
                    )
                schedule_df = pd.DataFrame(rows)
                st.session_state[session_key] = schedule_df

    edited_df = st.data_editor(
        schedule_df,
        hide_index=True,
        disabled=not edit_mode,
        key=f"{session_key}_editor",
    )
    st.session_state[session_key] = edited_df
    return edited_df


def _sanitize_product_records(df: pd.DataFrame) -> List[Dict]:
    records: List[Dict] = []
    cfg_fields = {f.name for f in fields(ProductConfig)}
    for raw in df.to_dict("records"):
        if not raw.get("name"):
            continue
        cleaned: Dict = {}
        for key, value in raw.items():
            if key not in cfg_fields:
                continue
            if isinstance(value, float) and np.isnan(value):
                continue
            cleaned[key] = value
        cleaned.setdefault("stage", "Unspecified")
        cleaned.setdefault("success_prob", 0.5)
        cleaned.setdefault("include_in_consolidation", True)
        records.append(cleaned)
    return records


def _build_portfolio(product_df: pd.DataFrame, model_cfg: ModelConfig) -> Portfolio | None:
    product_records = _sanitize_product_records(product_df)
    if not product_records:
        return None
    products = [Product(ProductConfig(**record), model_cfg) for record in product_records]
    return Portfolio(products, model_cfg)


def _compute_financial_statements(
    cons: pd.DataFrame, model_cfg: ModelConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = cons.index
    da_positive = -cons["da"]

    perf_df = pd.DataFrame(
        {
            "Revenue": cons["revenue"],
            "COGS": cons["cogs"],
            "Sales & Marketing": cons["sales_marketing"],
            "G&A": cons["gna"],
            "Royalty": cons["royalty"],
            "R&D expense": cons["rd_expense_pnl"],
            "EBITDA": cons["ebitda"],
            "EBIT": cons["ebit"],
            "Tax": cons["tax"],
            "NOPAT": cons["nopat"],
        }
    )

    wc = model_cfg.working_capital_pct_sales * cons["revenue"]
    wc_diff = wc.diff().fillna(wc)

    intangible = []
    ppe = []
    working_capital_asset = []
    retained = []
    paid_in = []

    intangible_val = 0.0
    ppe_val = 0.0
    wc_val = 0.0
    retained_val = 0.0

    for year in years:
        rd_cap_add = cons.loc[year, "rd_cap_add"]
        rd_amort = cons.loc[year, "rd_amort"]
        capex_cash = cons.loc[year, "capex_cash"]
        depreciation = cons.loc[year, "depreciation"]
        nopat = cons.loc[year, "nopat"]

        intangible_val += -rd_cap_add + rd_amort
        ppe_val += -capex_cash + depreciation
        wc_val += wc_diff.loc[year]
        retained_val += nopat

        total_assets = intangible_val + ppe_val + wc_val
        paid_in_val = max(0.0, total_assets - retained_val)

        intangible.append(intangible_val)
        ppe.append(ppe_val)
        working_capital_asset.append(wc_val)
        retained.append(retained_val)
        paid_in.append(paid_in_val)

    position_df = pd.DataFrame(
        {
            "Intangibles": intangible,
            "Property & equipment": ppe,
            "Working capital": working_capital_asset,
            "Total assets": np.array(intangible) + np.array(ppe) + np.array(working_capital_asset),
            "Retained earnings": retained,
            "Paid-in capital": paid_in,
            "Total equity": np.array(retained) + np.array(paid_in),
        },
        index=years,
    )

    cash_from_ops = cons["nopat"] + da_positive - wc_diff
    cash_from_investing = cons["capex_cash"] + cons["rd_cap_add"]
    cash_from_financing = pd.Series(0.0, index=years)
    net_cash = cash_from_ops + cash_from_investing + cash_from_financing

    cash_flow_df = pd.DataFrame(
        {
            "Cash from operations": cash_from_ops,
            "Cash from investing": cash_from_investing,
            "Cash from financing": cash_from_financing,
            "Net change in cash": net_cash,
        }
    )

    return perf_df, position_df, cash_flow_df


def _build_ratio_table(cons: pd.DataFrame) -> pd.DataFrame:
    revenue = cons["revenue"].replace(0, np.nan)
    gross_profit = cons["revenue"] + cons["cogs"]
    ratios = pd.DataFrame(index=cons.index)
    ratios["Gross margin"] = gross_profit / revenue
    ratios["EBITDA margin"] = cons["ebitda"] / revenue
    ratios["NOPAT margin"] = cons["nopat"] / revenue
    ratios["R&D intensity"] = cons["rd_cash"].abs() / revenue
    ratios["Capex intensity"] = (-cons["capex_cash"]) / revenue
    return ratios.fillna(0.0)


def main() -> None:
    st.set_page_config(
        page_title="Biotech Financial Model",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("Biotech / Agro Valuation Sandbox")
    st.write(
        "Configure a portfolio, run discounted cash flow valuations, and explore VC "
        "method estimates or stress scenarios."
    )

    model_cfg: ModelConfig | None = None
    portfolio: Portfolio | None = None
    valuation_result = None

    (
        config_tab,
        financial_tab,
        dashboard_tab,
        analytics_tab,
        scenario_tab,
        vc_tab,
    ) = st.tabs(
        [
            "Model configuration",
            "Financial statements",
            "Dashboard",
            "Advanced analytics",
            "Scenario analysis",
            "VC helper",
        ]
    )

    with config_tab:
        st.subheader("Global assumptions")
        col1, col2, col3 = st.columns(3)
        with col1:
            first_year = st.number_input("First forecast year", value=2024)
            n_years = st.number_input("Number of years", min_value=5, max_value=40, value=25)
            currency = st.text_input("Currency", value="USD")
        with col2:
            discount_rate = st.slider("Discount rate", min_value=0.02, max_value=0.25, value=0.1)
            tax_rate = st.slider("Tax rate", min_value=0.0, max_value=0.35, value=0.25)
        with col3:
            wc_pct = st.slider("Working capital (% sales)", 0.0, 0.3, 0.08)
            ev_multiple = st.slider("Base EV/EBITDA multiple", 2.0, 25.0, 8.0)

        ramp_df = _render_schedule_editor("Sales ramp schedule", "sales_ramp_schedule")
        ramp_df = ramp_df.sort_values("Year offset")
        if ramp_df.empty:
            st.warning("Ramp schedule empty. Reverting to default values.")
            ramp = _default_ramp_schedule()["Ramp factor"].tolist()
        else:
            ramp = ramp_df["Ramp factor"].astype(float).tolist()

        model_cfg = ModelConfig(
            first_year=int(first_year),
            n_years=int(n_years),
            currency=currency,
            discount_rate=float(discount_rate),
            tax_rate=float(tax_rate),
            working_capital_pct_sales=float(wc_pct),
            ev_ebitda_multiple=float(ev_multiple),
            sales_ramp_factors=ramp,
        )

        st.subheader("Product assumptions")
        if "product_table" not in st.session_state:
            st.session_state["product_table"] = _default_products()

        product_df = st.data_editor(
            st.session_state["product_table"],
            num_rows="dynamic",
            hide_index=True,
            key="product_editor",
            column_config={
                "stage": st.column_config.SelectboxColumn("Stage", options=STAGE_OPTIONS),
                "include_in_consolidation": st.column_config.CheckboxColumn("Include", default=True),
                "success_prob": st.column_config.NumberColumn(
                    "Success probability", min_value=0.0, max_value=1.0, step=0.05
                ),
            },
        )
        st.session_state["product_table"] = product_df

        portfolio = _build_portfolio(product_df, model_cfg)
        if portfolio is None:
            st.info("Add at least one product with a name to run valuations.")
        else:
            valuation_result = ValuationEngine(portfolio).run()
            st.success(
                f"Run complete: portfolio rNPV = {valuation_result.rnpv:,.0f} {model_cfg.currency}."
            )

            with st.expander("Consolidated forecast", expanded=True):
                cons = valuation_result.consolidated.copy()
                cons_display = cons[["revenue", "ebitda", "fcff_after_wc"]].copy()
                cons_display.columns = ["Revenue", "EBITDA", "FCFF after WC"]
                st.dataframe(cons_display.style.format("{:.0f}"))
                st.line_chart(cons_display)
    with financial_tab:
        st.subheader("Financial statements")
        if valuation_result is None or model_cfg is None:
            st.info("Run the model configuration tab to populate the statements.")
        else:
            cons = valuation_result.consolidated
            perf_df, position_df, cash_flow_df = _compute_financial_statements(cons, model_cfg)
            st.markdown("**Statement of Financial Performance**")
            st.dataframe(perf_df.style.format("{:.0f}"))
            st.markdown("**Statement of Financial Position**")
            st.dataframe(position_df.style.format("{:.0f}"))
            st.markdown("**Statement of Cash Flows**")
            st.dataframe(cash_flow_df.style.format("{:.0f}"))

    with dashboard_tab:
        st.subheader("Dashboard")
        if valuation_result is None or model_cfg is None:
            st.info("Configure and run the model to see dashboard metrics.")
        else:
            cons = valuation_result.consolidated
            kpi_cols = st.columns(4)
            kpi_cols[0].metric("Portfolio rNPV", f"{valuation_result.rnpv:,.0f} {model_cfg.currency}")
            kpi_cols[1].metric("Peak revenue", f"{cons['revenue'].max():,.0f}")
            avg_margin = cons["ebitda"].sum() / cons["revenue"].sum() if cons["revenue"].sum() else 0.0
            kpi_cols[2].metric("Avg EBITDA margin", f"{avg_margin:.1%}")
            kpi_cols[3].metric("Total FCFF after WC", f"{cons['fcff_after_wc'].sum():,.0f}")

            chart_data = cons[["revenue", "ebitda", "fcff_after_wc"]]
            st.area_chart(chart_data)
            st.bar_chart(cons["fcff_after_wc"], use_container_width=True)

    with analytics_tab:
        st.subheader("Advanced financial analytics")
        if valuation_result is None or model_cfg is None or portfolio is None:
            st.info("Configure the model to unlock analytics.")
        else:
            cons = valuation_result.consolidated
            ratios = _build_ratio_table(cons)
            st.markdown("**Margin & intensity analysis**")
            st.dataframe(ratios.style.format("{:.1%}"))

            st.markdown("**Monte Carlo risk simulation**")
            mc_cols = st.columns(4)
            n_sims = mc_cols[0].number_input("Simulations", min_value=100, max_value=5000, value=1000, step=100)
            rev_sigma = mc_cols[1].number_input("Revenue sigma", min_value=0.01, max_value=0.5, value=0.15, step=0.01)
            cost_sigma = mc_cols[2].number_input("Cost sigma", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
            seed = mc_cols[3].number_input("Random seed", min_value=0, value=42)

            if st.button("Run Monte Carlo simulation"):
                sims = MonteCarloEngine(portfolio).simulate(
                    n_sims=int(n_sims),
                    revenue_sigma=float(rev_sigma),
                    cost_sigma=float(cost_sigma),
                    random_seed=int(seed),
                )
                st.session_state["mc_results"] = sims

            sims = st.session_state.get("mc_results")
            if sims is not None:
                st.line_chart(sims.reset_index(drop=True))
                var = MonteCarloEngine.value_at_risk(sims)
                cvar = MonteCarloEngine.conditional_value_at_risk(sims)
                st.write(
                    f"Mean rNPV: {sims.mean():,.0f} | Std: {sims.std():,.0f} | VaR95: {var:,.0f} | CVaR95: {cvar:,.0f}"
                )

    with scenario_tab:
        st.subheader("Scenario analysis")
        if portfolio is None:
            st.info("Configure the model in the first tab to enable scenarios.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            rev_mult = col1.slider("Revenue multiplier", 0.25, 2.5, 1.0)
            cost_mult = col2.slider("Cost multiplier", 0.5, 2.0, 1.0)
            dr_shift = col3.slider("Discount rate shift", -0.05, 0.1, 0.0)
            prob_mult = col4.slider("Success prob multiplier", 0.5, 1.5, 1.0)
            scenario = Scenario(
                name="Custom scenario",
                revenue_multiplier=float(rev_mult),
                cost_multiplier=float(cost_mult),
                discount_rate_shift=float(dr_shift),
                success_prob_multiplier=float(prob_mult),
            )
            scen_results = ScenarioEngine(portfolio).run_scenarios([scenario])
            st.dataframe(
                scen_results.style.format({"rnpv": "{:.0f}", "ebitda_value": "{:.0f}"})
            )

    with vc_tab:
        st.subheader("VC method helper")
        if valuation_result is None or model_cfg is None:
            st.info("Configure the model and run a valuation before using VC analysis.")
        else:
            vc_col1, vc_col2, vc_col3, vc_col4 = st.columns(4)
            exit_year = vc_col1.number_input("Exit year", value=model_cfg.first_year + 5)
            target_irr = vc_col2.slider("Target IRR", 0.05, 0.6, 0.3)
            ownership = vc_col3.slider("Investor ownership at exit", 0.05, 0.9, 0.25)
            new_money = vc_col4.number_input(
                "New money ($)", min_value=1_000_000, value=50_000_000, step=5_000_000
            )
            exit_multiple = st.slider("Exit EV/EBITDA multiple", 2.0, 25.0, model_cfg.ev_ebitda_multiple)

            vc_inputs = VCInputs(
                exit_year=int(exit_year),
                target_irr=float(target_irr),
                investor_ownership_at_exit=float(ownership),
                new_money=float(new_money),
            )
            vc_valuator = VCValuator(valuation_result)
            vc_output = vc_valuator.vc_method(vc_inputs, exit_multiple=float(exit_multiple))
            vc_df = pd.DataFrame(
                {
                    "Metric": list(vc_output.keys()),
                    "Value": [
                        f"{value:,.0f}" if "irr" not in key else f"{value:.2%}"
                        for key, value in vc_output.items()
                    ],
                }
            )
            st.table(vc_df)

    st.caption(
        "Tip: Upload a Prophet-ready dataframe (ds, y) and plug it into ForecastScenarioBridge for richer scenarios."
    )


if __name__ == "__main__":
    main()
