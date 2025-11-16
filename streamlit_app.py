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
    MultiplesModel,
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


def _parse_sales_ramp(text: str) -> List[float]:
    try:
        return [float(v.strip()) for v in text.split(",") if v.strip()]
    except ValueError:
        st.warning("Sales ramp factors invalid. Using default ramp instead.")
        return [0.2, 0.6, 1.0, 1.0, 1.0]


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


def _train_multiples_model(training_df: pd.DataFrame, target_column: str) -> MultiplesModel:
    features = training_df.drop(columns=[target_column])
    target = training_df[target_column]
    model = MultiplesModel(feature_names=features.columns)
    model.fit(features, target)
    return model


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

    config_tab, scenario_tab, vc_tab = st.tabs(
        ["Model configuration", "Scenario analysis", "VC helper"]
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

        sales_ramp_text = st.text_input(
            "Sales ramp factors (comma separated)", value="0.2,0.6,1.0,1.0,1.0"
        )
        ramp = _parse_sales_ramp(sales_ramp_text)

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

    with scenario_tab:
        st.subheader("Scenario analysis")
        if portfolio is None:
            st.info("Configure the model in the previous tab to enable scenarios.")
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

            ml_model: MultiplesModel | None = st.session_state.get("multiples_model")
            with st.expander("Optional: train an ML multiples model"):
                uploaded = st.file_uploader("Training CSV", type="csv")
                if uploaded is not None:
                    train_df = pd.read_csv(uploaded)
                    st.write("Detected columns:", list(train_df.columns))
                    if len(train_df.columns) >= 2:
                        target_col = st.selectbox(
                            "Target column (EV/EBITDA multiple)",
                            options=train_df.columns,
                            key="target_column_selector",
                        )
                        if st.button("Train multiples model"):
                            model = _train_multiples_model(train_df, target_col)
                            st.session_state["multiples_model"] = model
                            ml_model = model
                            st.success("Multiples model trained and stored in session.")
                    else:
                        st.warning(
                            "Upload a dataset with at least one feature column plus the target."
                        )
                if ml_model is not None:
                    st.info(
                        "A multiples model is available. It will consume consolidated metrics for the exit year."
                    )

            vc_inputs = VCInputs(
                exit_year=int(exit_year),
                target_irr=float(target_irr),
                investor_ownership_at_exit=float(ownership),
                new_money=float(new_money),
            )

            vc_valuator = VCValuator(
                valuation_result,
                default_exit_multiple=None if ml_model else model_cfg.ev_ebitda_multiple,
                multiples_model=ml_model,
            )
            vc_output = vc_valuator.vc_method(vc_inputs)
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
