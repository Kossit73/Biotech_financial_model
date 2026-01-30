"""Streamlit UI for the Valuation Codex biotech financial model."""

from __future__ import annotations

import json
import io
import importlib
import os
from io import BytesIO
from dataclasses import asdict, fields
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency
    go = None

try:  # optional optimisation + ML helpers
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - optional dependency
    minimize = None

try:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - optional dependency
    KMeans = None
    LinearRegression = None
    LogisticRegression = None
    StandardScaler = None

from valuation_codex_package import (
    ModelConfig,
    Portfolio,
    Product,
    ProductConfig,
    Scenario,
    ScenarioEngine,
    ForecastEngine,
    VCInputs,
    VCValuator,
    ValuationEngine,
    ValuationResult,
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

SELECTOR_OPTIONS = [
    "Base case",
    "Upside",
    "Downside",
    "Aggressive expansion",
    "Defensive posture",
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


def _blank_product_row(name: str = "New vaccine") -> Dict:
    """Return a ProductConfig-like dict for initializing new rows."""

    cfg = ProductConfig(
        name=name,
        stage="Discovery",
        success_prob=0.2,
        include_in_consolidation=True,
        patent_revenue_target=50_000_000,
        post_patent_revenue_target=25_000_000,
        cogs_patent=0.35,
        cogs_post=0.5,
        sales_marketing_pct=0.15,
        gna_pct=0.1,
        rd_remaining_pre_launch=25_000_000,
        rd_annual_post_launch=5_000_000,
        capex_remaining_pre_launch=10_000_000,
        capex_annual_post_launch=2_000_000,
    )
    return asdict(cfg)


def _default_vaccine_sales_table(first_year: int = 2024, horizon_years: int = 5) -> pd.DataFrame:
    years = [first_year + i for i in range(max(horizon_years, 1))]
    data = {
        "Year": years,
        "Doses (M)": [5, 7, 10, 12, 12][: len(years)],
        "Price per dose": [25, 26, 27, 27, 28][: len(years)],
        "Comments": [""] * len(years),
    }
    return pd.DataFrame(data)


def _blank_vaccine_sales_row(df: pd.DataFrame, first_year: int) -> Dict:
    next_year = first_year
    if "Year" in df.columns and not df.empty:
        with pd.option_context("mode.use_inf_as_na", True):
            existing_years = pd.to_numeric(df["Year"], errors="coerce").dropna()
        if not existing_years.empty:
            next_year = int(existing_years.max()) + 1
    doses = 5.0
    price = 25.0
    if "Doses (M)" in df.columns and not df.empty:
        last_doses = pd.to_numeric(df["Doses (M)"], errors="coerce").dropna()
        if not last_doses.empty:
            doses = float(last_doses.iloc[-1])
    if "Price per dose" in df.columns and not df.empty:
        last_price = pd.to_numeric(df["Price per dose"], errors="coerce").dropna()
        if not last_price.empty:
            price = float(last_price.iloc[-1])
    return {
        "Year": next_year,
        "Doses (M)": doses,
        "Price per dose": price,
        "Comments": "",
    }


def _default_uses_table() -> pd.DataFrame:
    data = [
        {"Item": "Clinical trials", "Amount": 150_000_000},
        {"Item": "Manufacturing scale-up", "Amount": 90_000_000},
    ]
    return pd.DataFrame(data)


def _blank_use_row(df: pd.DataFrame) -> Dict:
    return {"Item": "New use", "Amount": 0.0}


def _default_sources_table() -> pd.DataFrame:
    data = [
        {"Item": "Existing cash", "Amount": 40_000_000},
        {"Item": "New equity", "Amount": 200_000_000},
    ]
    return pd.DataFrame(data)


def _blank_source_row(df: pd.DataFrame) -> Dict:
    return {"Item": "New source", "Amount": 0.0}


def _default_shareholders_table() -> pd.DataFrame:
    data = [
        {"Shareholder": "Founders", "Ownership %": 0.35, "Investment": 25_000_000},
        {"Shareholder": "Series A fund", "Ownership %": 0.4, "Investment": 80_000_000},
    ]
    return pd.DataFrame(data)


def _blank_shareholder_row(df: pd.DataFrame) -> Dict:
    return {"Shareholder": "New investor", "Ownership %": 0.05, "Investment": 0.0}


def _default_market_sizes_table() -> pd.DataFrame:
    data = [
        {"Segment": "Global vaccine market", "Value": 80_000_000_000},
        {"Segment": "Target indication", "Value": 12_000_000_000},
    ]
    return pd.DataFrame(data)


def _blank_relevant_market_row(df: pd.DataFrame) -> Dict:
    return {"Segment": "New segment", "Value": 1_000_000}


def _default_vaccine_development_table(first_year: int = 2024) -> pd.DataFrame:
    data = [
        {
            "ID_vaccine": "VAC-001",
            "Vaccine name": "AgSeed-101",
            "Stage": "Phase II",
            "Success Probability %": 35.0,
            "Consolidation": True,
            "First year forecast": first_year + 2,
            "Time to market": 4,
            "Market entry year": first_year + 6,
            "Patent duration years": 15,
            "End patent year": first_year + 20,
        },
        {
            "ID_vaccine": "VAC-002",
            "Vaccine name": "BioYield-Plus",
            "Stage": "Phase III",
            "Success Probability %": 55.0,
            "Consolidation": True,
            "First year forecast": first_year + 1,
            "Time to market": 2,
            "Market entry year": first_year + 3,
            "Patent duration years": 17,
            "End patent year": first_year + 19,
        },
    ]
    return pd.DataFrame(data)


def _default_market_size_estimation_table() -> pd.DataFrame:
    data = [
        {
            "ID_vaccine": "VAC-001",
            "Vaccine name": "AgSeed-101",
            "Market size (# customers)": 5_000_000,
            "Average spend (USD/customer)": 120,
            "Serviceable Available Market (% TAM)": 60.0,
            "Serviceable Available Market (% Market size)": 45.0,
            "Serviceable Obtainable Market (%)": 25.0,
        },
        {
            "ID_vaccine": "VAC-002",
            "Vaccine name": "BioYield-Plus",
            "Market size (# customers)": 8_000_000,
            "Average spend (USD/customer)": 150,
            "Serviceable Available Market (% TAM)": 55.0,
            "Serviceable Available Market (% Market size)": 35.0,
            "Serviceable Obtainable Market (%)": 18.0,
        },
    ]
    return pd.DataFrame(data)


def _default_vaccine_revenue_table() -> pd.DataFrame:
    data = [
        {
            "ID_vaccine": "VAC-001",
            "Vaccine name": "AgSeed-101",
            "Patent customers per year": 3_000_000,
            "Patent price (USD/customer)": 50,
            "Post patent customer adj. %": 80.0,
            "Post patent price adj. %": 85.0,
        },
        {
            "ID_vaccine": "VAC-002",
            "Vaccine name": "BioYield-Plus",
            "Patent customers per year": 4_200_000,
            "Patent price (USD/customer)": 65,
            "Post patent customer adj. %": 75.0,
            "Post patent price adj. %": 80.0,
        },
    ]
    return pd.DataFrame(data)


def _default_royalty_table() -> pd.DataFrame:
    data = [
        {
            "ID_vaccine": "VAC-001",
            "Vaccine name": "AgSeed-101",
            "Monetization model": "Product Sale",
            "Royalty rate (%)": 5.0,
        },
        {
            "ID_vaccine": "VAC-002",
            "Vaccine name": "BioYield-Plus",
            "Monetization model": "Licensing",
            "Royalty rate (%)": 6.5,
        },
    ]
    return pd.DataFrame(data)


def _default_market_share_table() -> pd.DataFrame:
    data = [
        {
            "ID_vaccine": "VAC-001",
            "Vaccine name": "AgSeed-101",
            "Relevant market type": "Global row crops",
            "Relevant market size (USD)": 4_500_000_000,
            "Revenue target - patent %": 12.0,
            "Revenue target - post %": 8.0,
            "Market share patent %": 6.0,
            "Market share post %": 4.0,
            "Market growth %": 5.0,
            "Sales growth %": 8.0,
        },
        {
            "ID_vaccine": "VAC-002",
            "Vaccine name": "BioYield-Plus",
            "Relevant market type": "Specialty crops",
            "Relevant market size (USD)": 3_200_000_000,
            "Revenue target - patent %": 15.0,
            "Revenue target - post %": 10.0,
            "Market share patent %": 7.5,
            "Market share post %": 5.0,
            "Market growth %": 4.0,
            "Sales growth %": 6.0,
        },
    ]
    return pd.DataFrame(data)


def _default_vaccine_cost_table() -> pd.DataFrame:
    data = [
        {
            "ID_vaccine": "VAC-001",
            "Vaccine name": "AgSeed-101",
            "COGS patent % of sales": 32.0,
            "COGS post % of sales": 48.0,
            "Marketing annual % of sales": 18.0,
            "Marketing launch cost (USD)": 25_000_000,
            "Indirect staff cost (USD)": 8_500_000,
            "Electricity (USD)": 1_800_000,
            "Depreciation (USD)": 3_200_000,
            "Interest & amortization (USD)": 2_000_000,
            "Royalties cost % of sales": 4.0,
        },
        {
            "ID_vaccine": "VAC-002",
            "Vaccine name": "BioYield-Plus",
            "COGS patent % of sales": 28.0,
            "COGS post % of sales": 45.0,
            "Marketing annual % of sales": 16.0,
            "Marketing launch cost (USD)": 30_000_000,
            "Indirect staff cost (USD)": 6_750_000,
            "Electricity (USD)": 1_400_000,
            "Depreciation (USD)": 2_750_000,
            "Interest & amortization (USD)": 1_500_000,
            "Royalties cost % of sales": 3.5,
        },
    ]
    return pd.DataFrame(data)


def _default_vaccine_rd_table() -> pd.DataFrame:
    data = [
        {
            "ID_vaccine": "VAC-001",
            "Vaccine name": "AgSeed-101",
            "Cost accounting (capitalisation)": "50% capitalised",
            "Pre-GTM spent to date (USD)": 120_000_000,
            "Pre-GTM remaining (USD)": 60_000_000,
            "Post-GTM annual cost (USD/year)": 12_000_000,
        },
        {
            "ID_vaccine": "VAC-002",
            "Vaccine name": "BioYield-Plus",
            "Cost accounting (capitalisation)": "40% capitalised",
            "Pre-GTM spent to date (USD)": 80_000_000,
            "Pre-GTM remaining (USD)": 40_000_000,
            "Post-GTM annual cost (USD/year)": 9_500_000,
        },
    ]
    return pd.DataFrame(data)


def _default_vaccine_capex_table() -> pd.DataFrame:
    data = [
        {
            "ID_vaccine": "VAC-001",
            "Vaccine name": "AgSeed-101",
            "Pre-GTM capex spent (USD)": 55_000_000,
            "Pre-GTM capex remaining (USD)": 25_000_000,
            "Post-GTM yearly capex (USD)": 6_500_000,
        },
        {
            "ID_vaccine": "VAC-002",
            "Vaccine name": "BioYield-Plus",
            "Pre-GTM capex spent (USD)": 35_000_000,
            "Pre-GTM capex remaining (USD)": 15_000_000,
            "Post-GTM yearly capex (USD)": 4_000_000,
        },
    ]
    return pd.DataFrame(data)


def _next_vaccine_id(df: pd.DataFrame) -> str:
    """Return the next sequential vaccine identifier (VAC-XXX)."""

    existing = set()
    if "ID_vaccine" in df.columns:
        existing = {
            str(val)
            for val in df["ID_vaccine"].astype(str).tolist()
            if val and val != "nan"
        }
    idx = 1
    while True:
        candidate = f"VAC-{idx:03d}"
        if candidate not in existing:
            return candidate
        idx += 1


def _blank_vaccine_development_row(df: pd.DataFrame, first_year: int) -> Dict:
    next_id = _next_vaccine_id(df)
    return {
        "ID_vaccine": next_id,
        "Vaccine name": "New vaccine",
        "Stage": "Discovery",
        "Success Probability %": 30.0,
        "Consolidation": True,
        "First year forecast": first_year,
        "Time to market": 3,
        "Market entry year": first_year + 3,
        "Patent duration years": 15,
        "End patent year": first_year + 17,
    }


def _blank_market_size_row(df: pd.DataFrame) -> Dict:
    next_id = _next_vaccine_id(df)
    return {
        "ID_vaccine": next_id,
        "Vaccine name": "New vaccine",
        "Market size (# customers)": 1_000_000,
        "Average spend (USD/customer)": 100.0,
        "Serviceable Available Market (% TAM)": 50.0,
        "Serviceable Available Market (% Market size)": 40.0,
        "Serviceable Obtainable Market (%)": 20.0,
    }


def _blank_vaccine_revenue_row(df: pd.DataFrame) -> Dict:
    next_id = _next_vaccine_id(df)
    return {
        "ID_vaccine": next_id,
        "Vaccine name": "New vaccine",
        "Patent customers per year": 1_000_000,
        "Patent price (USD/customer)": 50.0,
        "Post patent customer adj. %": 80.0,
        "Post patent price adj. %": 85.0,
    }


def _blank_vaccine_cost_row(df: pd.DataFrame) -> Dict:
    next_id = _next_vaccine_id(df)
    return {
        "ID_vaccine": next_id,
        "Vaccine name": "New vaccine",
        "COGS patent % of sales": 30.0,
        "COGS post % of sales": 45.0,
        "Marketing annual % of sales": 15.0,
        "Marketing launch cost (USD)": 10_000_000,
        "Indirect staff cost (USD)": 5_000_000,
        "Electricity (USD)": 1_000_000,
        "Depreciation (USD)": 2_000_000,
        "Interest & amortization (USD)": 1_000_000,
        "Royalties cost % of sales": 3.0,
    }


def _blank_vaccine_rd_row(df: pd.DataFrame) -> Dict:
    next_id = _next_vaccine_id(df)
    return {
        "ID_vaccine": next_id,
        "Vaccine name": "New vaccine",
        "Cost accounting (capitalisation)": "50% capitalised",
        "Pre-GTM spent to date (USD)": 20_000_000,
        "Pre-GTM remaining (USD)": 10_000_000,
        "Post-GTM annual cost (USD/year)": 5_000_000,
    }


def _blank_vaccine_capex_row(df: pd.DataFrame) -> Dict:
    next_id = _next_vaccine_id(df)
    return {
        "ID_vaccine": next_id,
        "Vaccine name": "New vaccine",
        "Pre-GTM capex spent (USD)": 10_000_000,
        "Pre-GTM capex remaining (USD)": 5_000_000,
        "Post-GTM yearly capex (USD)": 2_000_000,
    }


def _blank_vaccine_royalty_row(df: pd.DataFrame) -> Dict:
    next_id = _next_vaccine_id(df)
    return {
        "ID_vaccine": next_id,
        "Vaccine name": "New vaccine",
        "Monetization model": "Licensing",
        "Royalty rate (%)": 5.0,
    }


def _blank_vaccine_market_share_row(df: pd.DataFrame) -> Dict:
    next_id = _next_vaccine_id(df)
    return {
        "ID_vaccine": next_id,
        "Vaccine name": "New vaccine",
        "Relevant market type": "New segment",
        "Relevant market size (USD)": 1_000_000_000,
        "Revenue target - patent %": 10.0,
        "Revenue target - post %": 5.0,
        "Market share patent %": 5.0,
        "Market share post %": 3.0,
        "Market growth %": 5.0,
        "Sales growth %": 8.0,
    }


def _ensure_table_state(key: str, default_factory: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    if key not in st.session_state or st.session_state[key] is None:
        st.session_state[key] = default_factory()
    return st.session_state[key]


def _format_row_label(
    df: pd.DataFrame,
    idx,
    id_column: Optional[str],
    name_column: Optional[str],
) -> str:
    parts: List[str] = []
    if id_column and id_column in df.columns:
        val = df.at[idx, id_column]
        if pd.notna(val):
            parts.append(str(val))
    if name_column and name_column in df.columns:
        val = df.at[idx, name_column]
        if pd.notna(val):
            parts.append(str(val))
    if not parts:
        pos = df.index.get_loc(idx) if idx in df.index else 0
        parts.append(f"Row {pos + 1}")
    return " - ".join(parts)


def _pending_selection_key(select_key: str) -> str:
    return f"{select_key}_pending"


def _set_pending_selection(select_key: str, value: Optional[object]) -> None:
    st.session_state[_pending_selection_key(select_key)] = value


def _consume_pending_selection(select_key: str) -> Optional[object]:
    pending_key = _pending_selection_key(select_key)
    if pending_key in st.session_state:
        value = st.session_state.pop(pending_key)
        st.session_state[select_key] = value
        return value
    return None


def _row_identifier(df: pd.DataFrame, idx: int, id_column: Optional[str]) -> object:
    if id_column and id_column in df.columns:
        value = df.at[idx, id_column]
        if pd.isna(value):
            return idx
        return value
    return idx


def _resolve_selected_index(
    df: pd.DataFrame,
    select_key: str,
    id_column: Optional[str],
) -> Optional[int]:
    if df.empty:
        return None

    selected_id = st.session_state.get(select_key)
    if id_column and id_column in df.columns and selected_id is not None:
        matches = df.index[df[id_column] == selected_id]
        if len(matches):
            return matches[0]

    if selected_id in df.index:
        return selected_id
    return df.index[0]


def _resolve_selected_index_from_value(
    df: pd.DataFrame,
    selected_id: Optional[object],
    id_column: Optional[str],
) -> Optional[int]:
    if df.empty:
        return None
    if id_column and id_column in df.columns and selected_id is not None:
        matches = df.index[df[id_column] == selected_id]
        if len(matches):
            return matches[0]
    if selected_id in df.index:
        return selected_id
    return df.index[0]


def _validate_selection(
    df: pd.DataFrame,
    select_key: str,
    id_column: Optional[str],
) -> None:
    if df.empty:
        _set_pending_selection(select_key, None)
        return

    selected_idx = _resolve_selected_index(df, select_key, id_column)
    if selected_idx is None:
        _set_pending_selection(select_key, _row_identifier(df, df.index[0], id_column))
        return

    selected_id = _row_identifier(df, selected_idx, id_column)
    if selected_id != st.session_state.get(select_key):
        _set_pending_selection(select_key, selected_id)


def _render_row_selector(
    df: pd.DataFrame,
    select_key: str,
    id_column: Optional[str],
    name_column: Optional[str],
) -> Optional[int]:
    pending = _consume_pending_selection(select_key)

    if df.empty:
        st.caption("No rows available yet.")
        st.session_state.pop(select_key, None)
        st.session_state.pop(_pending_selection_key(select_key), None)
        return None

    options = list(df.index)
    selected_id = pending if pending is not None else st.session_state.get(select_key)
    default_idx = _resolve_selected_index_from_value(df, selected_id, id_column)
    if default_idx is None or default_idx not in options:
        default_idx = options[0]

    def _format(idx):
        return _format_row_label(df, idx, id_column, name_column)

    selected = st.selectbox(
        "Select row",
        options=options,
        format_func=_format,
        index=options.index(default_idx),
        key=select_key,
    )
    return selected


def _apply_yearly_increment(
    section_key: str,
    df: pd.DataFrame,
    selected_idx: Optional[int],
) -> pd.DataFrame:
    st.markdown("**Yearly Increment Helper**")
    if df.empty or selected_idx is None:
        st.caption("Select a row to apply increments.")
        return df
    if selected_idx not in df.index:
        st.caption("Selected row is no longer available.")
        return df

    numeric_cols = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_cols:
        st.caption("No numeric columns available.")
        return df

    target_col = st.selectbox(
        "Column",
        options=numeric_cols,
        key=f"{section_key}_inc_col",
    )
    increment = st.number_input(
        "Increment per year",
        value=1.0,
        step=0.1,
        key=f"{section_key}_inc_value",
    )
    years = st.number_input(
        "Years to apply",
        min_value=1,
        max_value=50,
        value=1,
        key=f"{section_key}_inc_years",
    )

    base_value = df.at[selected_idx, target_col]
    if pd.isna(base_value):
        base_value = 0.0
    st.caption(f"Current value: {base_value:,.2f}")

    if st.button("Apply increment", key=f"{section_key}_inc_apply", use_container_width=True):
        df.at[selected_idx, target_col] = float(base_value) + increment * years
        st.session_state[section_key] = df
        st.success("Increment applied")
    return st.session_state.get(section_key, df)


def _widget_value(label: str, value, key: str):
    """Render an input widget based on the inferred data type of ``value``."""

    label_lower = label.lower()
    if label == "Stage":
        current = value if value in STAGE_OPTIONS else STAGE_OPTIONS[0]
        return st.selectbox(label, options=STAGE_OPTIONS, index=STAGE_OPTIONS.index(current), key=key)

    bool_like = isinstance(value, (bool, np.bool_)) or label_lower in {
        "include_in_consolidation",
        "consolidation",
    }
    if bool_like:
        return st.checkbox(label, value=bool(value), key=key)

    # Treat missing numeric values as zero for editing convenience.
    numeric_like = isinstance(value, (int, float, np.number)) or (
        isinstance(value, str) and value.strip().replace(".", "", 1).isdigit()
    )
    if numeric_like:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = 0.0
        min_value: Optional[float] = None
        max_value: Optional[float] = None
        step = 0.1 if abs(numeric_value) < 1 else 1.0
        if "%" in label or "prob" in label_lower or "growth" in label_lower or "share" in label_lower:
            min_value = 0.0
        if "%" in label or "prob" in label_lower or "probability" in label_lower:
            max_value = 100.0
        kwargs = {"value": float(numeric_value), "step": step, "key": key}
        if min_value is not None:
            kwargs["min_value"] = float(min_value)
        if max_value is not None:
            kwargs["max_value"] = float(max_value)
        return st.number_input(label, **kwargs)

    safe_value = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    return st.text_input(label, value=safe_value, key=key)


def _render_row_form(
    *,
    section_key: str,
    form_key: str,
    title: str,
    columns: List[str],
    initial_values: Dict,
    submit_label: str,
) -> Optional[Dict]:
    """Generic helper that renders a form for editing/adding a row."""

    with st.form(f"{section_key}_{form_key}"):
        st.caption(title)
        new_values: Dict = {}
        for col in columns:
            val = initial_values.get(col, "")
            widget_key = f"{section_key}_{form_key}_{col}"
            new_values[col] = _widget_value(col, val, widget_key)
        submitted = st.form_submit_button(submit_label, use_container_width=True)
    if submitted:
        return new_values
    return None


def _edit_selected_row(
    section_key: str,
    df: pd.DataFrame,
    selected_idx: Optional[int],
) -> pd.DataFrame:
    """Allow inline editing of the currently selected row."""

    if df.empty or selected_idx is None:
        st.caption("Select a row to edit.")
        return df

    columns = list(df.columns)
    initial_values = df.loc[selected_idx].to_dict()
    edited_values = _render_row_form(
        section_key=section_key,
        form_key="edit",
        title="Edit selected row",
        columns=columns,
        initial_values=initial_values,
        submit_label="Save changes",
    )
    if edited_values is not None:
        for col, val in edited_values.items():
            df.at[selected_idx, col] = val
        st.session_state[section_key] = df
        st.success("Row updated")
    return st.session_state.get(section_key, df)


def _add_row_via_form(
    section_key: str,
    df: pd.DataFrame,
    blank_row_factory: Callable[[pd.DataFrame], Dict],
    select_key: str,
    id_column: Optional[str],
) -> pd.DataFrame:
    """Render an add-row form so users can insert new entries with custom values."""

    template_row = blank_row_factory(df.copy())
    columns = list(df.columns) if not df.empty else list(template_row.keys())
    initial_values = {col: template_row.get(col, "") for col in columns}
    new_row = _render_row_form(
        section_key=section_key,
        form_key="add",
        title="Add a new row",
        columns=columns,
        initial_values=initial_values,
        submit_label="Add row",
    )
    if new_row is not None:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state[section_key] = df
        _set_pending_selection(select_key, _row_identifier(df, df.index[-1], id_column))
        st.success("Row added")
    return st.session_state.get(section_key, df)


def _remove_selected_row(
    section_key: str,
    df: pd.DataFrame,
    selected_idx: Optional[int],
    select_key: str,
    id_column: Optional[str],
) -> pd.DataFrame:
    """Delete the selected row when the user confirms the removal."""

    disabled = df.empty or selected_idx is None or selected_idx not in df.index
    if st.button(
        "Remove row",
        key=f"{section_key}_remove",
        use_container_width=True,
        disabled=disabled,
    ):
        if selected_idx is not None and selected_idx in df.index:
            df = df.drop(index=selected_idx).reset_index(drop=True)
            st.session_state[section_key] = df
            if not df.empty:
                _set_pending_selection(select_key, _row_identifier(df, df.index[-1], id_column))
            else:
                _set_pending_selection(select_key, None)
            st.success("Row removed")
    return st.session_state.get(section_key, df)


def _render_product_assumption_table(
    *,
    session_key: str,
    default_factory: Callable[[], pd.DataFrame],
    blank_row_factory: Callable[[pd.DataFrame], Dict],
    column_config: Optional[Dict] = None,
    id_column: Optional[str] = "ID_vaccine",
    name_column: Optional[str] = "Vaccine name",
) -> pd.DataFrame:
    df = _ensure_table_state(session_key, default_factory).copy()
    select_key = f"{session_key}_row_select"
    selected_idx = _render_row_selector(df, select_key, id_column, name_column)

    action_cols = st.columns(4)
    with action_cols[0]:
        df = _edit_selected_row(session_key, df, selected_idx)
    with action_cols[1]:
        df = _add_row_via_form(session_key, df, blank_row_factory, select_key, id_column)
    with action_cols[2]:
        df = _remove_selected_row(session_key, df, selected_idx, select_key, id_column)
    with action_cols[3]:
        df = _apply_yearly_increment(session_key, df, selected_idx)

    df = st.session_state.get(session_key, df)
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        hide_index=True,
        key=f"{session_key}_editor",
        column_config=column_config,
    )
    st.session_state[session_key] = edited_df
    _validate_selection(edited_df, select_key, id_column)
    return edited_df


def _default_ramp_schedule() -> pd.DataFrame:
    """Return the seed schedule for global sales ramp factors."""

    default_ramp = [0.2, 0.6, 1.0, 1.0, 1.0]
    data = {
        "Year offset": list(range(len(default_ramp))),
        "Ramp factor": default_ramp,
    }
    return pd.DataFrame(data)


def _coerce_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


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


def _validate_product_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clamp probability/percentage fields to avoid invalid model assumptions."""

    validated = df.copy()
    if "success_prob" in validated.columns:
        validated["success_prob"] = (
            validated["success_prob"].fillna(0.0).clip(0.0, 1.0)
        )

    percent_cols = [
        "cogs_patent",
        "cogs_post",
        "sales_marketing_pct",
        "gna_pct",
        "royalty_pct",
        "rd_capitalization_ratio",
    ]
    for col in percent_cols:
        if col in validated.columns:
            upper = 1.0 if col != "royalty_pct" else None
            series = validated[col].fillna(0.0)
            if upper is None:
                validated[col] = series.clip(lower=0.0)
            else:
                validated[col] = series.clip(0.0, upper)

    if "include_in_consolidation" in validated.columns:
        validated["include_in_consolidation"] = validated[
            "include_in_consolidation"
        ].fillna(True)

    return validated


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


def _evaluate_portfolio_shock(
    portfolio: Portfolio,
    *,
    revenue_multiplier: float = 1.0,
    cost_multiplier: float = 1.0,
    discount_shift: float = 0.0,
    success_prob_multiplier: float = 1.0,
) -> Optional[ValuationResult]:
    """Run a valuation after applying a Scenario-style shock."""

    if portfolio is None:
        return None
    scenario = Scenario(
        name="analytics_scenario",
        revenue_multiplier=revenue_multiplier,
        cost_multiplier=cost_multiplier,
        discount_rate_shift=discount_shift,
        success_prob_multiplier=success_prob_multiplier,
    )
    scen_engine = ScenarioEngine(portfolio)
    shocked_portfolio = scen_engine._apply_scenario(scenario)
    return ValuationEngine(shocked_portfolio).run()


def _run_sensitivity_matrix(
    portfolio: Portfolio,
    driver_settings: Dict[str, Tuple[float, str]],
) -> pd.DataFrame:
    """Evaluate +/- shocks for each driver and return the resulting rNPVs."""

    rows: List[Dict[str, float]] = []
    if portfolio is None:
        return pd.DataFrame()

    for driver, (delta, driver_type) in driver_settings.items():
        for direction in (-(delta), 0.0, delta):
            rev_mult = 1.0
            cost_mult = 1.0
            if driver_type == "revenue":
                rev_mult = 1.0 + direction
            elif driver_type == "cost":
                cost_mult = 1.0 + direction
            elif driver_type == "productivity":
                rev_mult = 1.0 + direction
                cost_mult = max(0.1, 1.0 - direction / 2)

            result = _evaluate_portfolio_shock(
                portfolio,
                revenue_multiplier=rev_mult,
                cost_multiplier=cost_mult,
            )
            if result is None:
                continue
            rows.append(
                {
                    "Driver": driver,
                    "Change": f"{direction:+.0%}",
                    "rNPV": result.rnpv,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Delta vs base"] = df.groupby("Driver")["rNPV"].transform(lambda x: x - x.iloc[1])
    return df


def _compute_decomposition(cons: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Run a simple trend/seasonality decomposition on revenue if enough history exists."""

    if len(cons) < 6:
        return None
    series = cons["revenue"].copy()
    idx = pd.PeriodIndex(cons.index, freq="Y").to_timestamp()
    ts = pd.Series(series.values, index=idx)
    period = max(2, min(6, len(ts) // 2))
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except Exception:
        return None

    result = seasonal_decompose(ts, model="additive", period=period, extrapolate_trend="freq")
    return pd.DataFrame(
        {
            "observed": result.observed,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
        }
    )


def _build_segmentation_table(val_result) -> pd.DataFrame:
    rows = []
    if val_result is None:
        return pd.DataFrame()
    per_product = val_result.per_product_prob
    total_rev = sum(df["revenue"].sum() for df in per_product.values()) or 1.0
    for name, df in per_product.items():
        revenue = df["revenue"].sum()
        ebitda = df["ebitda"].sum()
        fcff = df["fcff"].sum()
        margin = ebitda / revenue if revenue else 0.0
        rows.append(
            {
                "Product": name,
                "Revenue share": revenue / total_rev,
                "EBITDA margin": margin,
                "FCFF (PV proxy)": fcff,
            }
        )
    return pd.DataFrame(rows)


def _goal_seek_revenue_multiplier(
    portfolio: Portfolio, target_rnpv: float, tolerance: float = 1e-3, max_iter: int = 20
) -> Tuple[float, Optional[float]]:
    """Binary-search the revenue multiplier needed to hit a target rNPV."""

    if portfolio is None:
        return 1.0, None
    low, high = 0.25, 3.0
    solution = None
    for _ in range(max_iter):
        mid = (low + high) / 2
        result = _evaluate_portfolio_shock(portfolio, revenue_multiplier=mid)
        if result is None:
            break
        diff = result.rnpv - target_rnpv
        if abs(diff) <= tolerance * max(1.0, target_rnpv):
            solution = result.rnpv
            return mid, solution
        if diff < 0:
            low = mid
        else:
            high = mid
    if solution is None:
        result = _evaluate_portfolio_shock(portfolio, revenue_multiplier=high)
        solution = result.rnpv if result else None
    return high, solution


def _tornado_dataframe(portfolio: Portfolio, base_rnpv: float) -> pd.DataFrame:
    """Compute +/- shocks for tornado and spider charts."""

    drivers = [
        ("Revenue", "revenue_multiplier"),
        ("COGS", "cost_multiplier"),
        ("Discount rate", "discount_rate"),
        ("Success probability", "success"),
    ]
    records = []
    for label, driver_type in drivers:
        for change in (-0.2, 0.2):
            kwargs = {
                "revenue_multiplier": 1.0,
                "cost_multiplier": 1.0,
                "discount_shift": 0.0,
                "success_prob_multiplier": 1.0,
            }
            if driver_type == "revenue_multiplier":
                kwargs["revenue_multiplier"] += change
            elif driver_type == "cost_multiplier":
                kwargs["cost_multiplier"] += change
            elif driver_type == "discount_rate":
                kwargs["discount_shift"] = change * 0.5
            else:
                kwargs["success_prob_multiplier"] += change
            result = _evaluate_portfolio_shock(portfolio, **kwargs)
            if result is None:
                continue
            records.append(
                {
                    "Driver": label,
                    "Change": f"{change:+.0%}",
                    "rNPV": result.rnpv,
                    "Delta": result.rnpv - base_rnpv,
                }
            )
    return pd.DataFrame(records)


def _run_linear_regressions(cons: pd.DataFrame) -> Optional[pd.DataFrame]:
    if LinearRegression is None or cons.empty:
        return None
    x = cons[["revenue"]].values
    rows = []
    for target in ["ebitda", "nopat", "fcff_after_wc"]:
        y = cons[target].values
        model = LinearRegression()
        try:
            model.fit(x, y)
        except Exception:
            return None
        rows.append(
            {
                "Target": target.upper(),
                "Intercept": model.intercept_,
                "Revenue beta": model.coef_[0],
                "R^2": model.score(x, y),
            }
        )
    return pd.DataFrame(rows)


def _run_classification_model(seg_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if LogisticRegression is None or seg_df.empty:
        return None
    df = seg_df.copy()
    df["High margin"] = (df["EBITDA margin"] > 0.3).astype(int)
    X = df[["Revenue share", "EBITDA margin"]].values
    y = df["High margin"].values
    model = LogisticRegression()
    try:
        model.fit(X, y)
    except Exception:
        return None
    probs = model.predict_proba(X)[:, 1]
    df["High-margin probability"] = probs
    return df[["Product", "Revenue share", "EBITDA margin", "High-margin probability"]]


def _optimize_operations(cons: pd.DataFrame) -> Optional[pd.DataFrame]:
    if minimize is None or cons.empty:
        return None

    avg_rev = cons["revenue"].mean()
    avg_cost = (-cons["cogs"].mean()) if not cons["cogs"].empty else 0.0

    def objective(x: np.ndarray) -> float:
        volume, efficiency = x
        revenue = avg_rev * volume * efficiency
        cost = avg_cost * volume * (2 - efficiency)
        return -(revenue - cost)

    cons_list = (
        {"type": "ineq", "fun": lambda x: x[0] - 0.5},
        {"type": "ineq", "fun": lambda x: x[1] - 0.5},
        {"type": "ineq", "fun": lambda x: 2.0 - x[0]},
        {"type": "ineq", "fun": lambda x: 1.5 - x[1]},
    )
    res = minimize(objective, x0=np.array([1.0, 1.0]), constraints=cons_list)
    if not res.success:
        return None
    volume, efficiency = res.x
    opt_profit = -res.fun
    return pd.DataFrame(
        {
            "Metric": ["Optimal volume scale", "Optimal efficiency", "Profit"],
            "Value": [volume, efficiency, opt_profit],
        }
    )


def _mean_variance_portfolio(val_result) -> Optional[pd.DataFrame]:
    if val_result is None:
        return None
    per_product = val_result.per_product_prob
    rows = []
    for name, df in per_product.items():
        returns = df["fcff"].values
        if len(returns) < 2:
            continue
        rows.append(
            {
                "Product": name,
                "Mean": np.mean(returns),
                "Std": np.std(returns),
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    inv_var = 1.0 / df["Std"].replace(0, np.nan)
    inv_var = inv_var.fillna(0.0)
    if inv_var.sum() > 0:
        df["Suggested weight"] = inv_var / inv_var.sum()
    else:
        df["Suggested weight"] = 1.0 / len(df)
    return df


def _real_options_value(val_result, volatility: float = 0.35, years: int = 3) -> Optional[float]:
    if val_result is None:
        return None
    underlying = max(val_result.rnpv, 0.0)
    strike = val_result.consolidated["rd_cash"].abs().sum() / years if years else 1.0
    if strike <= 0:
        return None
    # Black-Scholes call option approximation on project deferral
    from math import log, sqrt
    try:
        from scipy.stats import norm
    except Exception:
        return None

    r = 0.05
    T = max(1e-6, years)
    d1 = (log(underlying / strike) + (r + 0.5 * volatility**2) * T) / (volatility * sqrt(T))
    d2 = d1 - volatility * sqrt(T)
    option_value = underlying * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
    return option_value


def _copula_simulation(cons: pd.DataFrame, rho: float = 0.4, draws: int = 2000) -> Optional[pd.DataFrame]:
    if cons.empty:
        return None
    mean_vec = np.array([cons["revenue"].mean(), cons["ebitda"].mean()])
    std_vec = np.array([cons["revenue"].std(), cons["ebitda"].std()])
    cov = np.array([[1.0, rho], [rho, 1.0]])
    samples = np.random.multivariate_normal([0, 0], cov, size=draws)
    revenue_sim = mean_vec[0] + std_vec[0] * samples[:, 0]
    ebitda_sim = mean_vec[1] + std_vec[1] * samples[:, 1]
    return pd.DataFrame({"Revenue": revenue_sim, "EBITDA": ebitda_sim})


def _cluster_products(val_result) -> Optional[pd.DataFrame]:
    if val_result is None or KMeans is None:
        return None
    per_product = val_result.per_product_prob
    rows = []
    for name, df in per_product.items():
        revenue = df["revenue"].sum()
        ebitda = df["ebitda"].sum()
        growth = df["revenue"].pct_change().mean()
        rows.append([name, revenue, ebitda, growth if pd.notna(growth) else 0.0])
    if not rows:
        return None
    names, data = zip(*[(r[0], r[1:]) for r in rows])
    scaler = StandardScaler() if StandardScaler else None
    matrix = np.array(data, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is not None:
        matrix = scaler.fit_transform(matrix)
    n_clusters = min(3, len(matrix))
    if n_clusters < 1:
        return None
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(matrix)
    return pd.DataFrame({"Product": names, "Cluster": labels})


def _machine_learning_multiple(cons: pd.DataFrame) -> Optional[pd.DataFrame]:
    if LinearRegression is None or cons.empty:
        return None
    growth = cons["revenue"].pct_change().fillna(0.0)
    features = pd.DataFrame(
        {
            "Revenue": cons["revenue"],
            "EBITDA": cons["ebitda"],
            "Growth": growth,
        }
    )
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    multiples = (cons["ebitda"].rolling(3).mean().fillna(method="bfill") + 1) / 1_000_000
    model = LinearRegression()
    model.fit(features.values, multiples.values)
    pred = model.predict(features.values)
    return pd.DataFrame({"Year": cons.index, "Predicted multiple": pred})


def _compute_irr(cashflows: List[float]) -> Optional[float]:
    if not cashflows or all(cf >= 0 for cf in cashflows) or all(cf <= 0 for cf in cashflows):
        return None

    def npv(rate: float) -> float:
        return sum(cf / ((1 + rate) ** idx) for idx, cf in enumerate(cashflows))

    low, high = -0.9, 1.0
    npv_low, npv_high = npv(low), npv(high)
    attempts = 0
    while npv_low * npv_high > 0 and attempts < 10:
        high += 1.0
        npv_high = npv(high)
        attempts += 1
    if npv_low * npv_high > 0:
        return None

    for _ in range(60):
        mid = (low + high) / 2
        npv_mid = npv(mid)
        if abs(npv_mid) < 1e-6:
            return mid
        if npv_low * npv_mid <= 0:
            high = mid
            npv_high = npv_mid
        else:
            low = mid
            npv_low = npv_mid
    return (low + high) / 2


def _compute_payback_years(years: List[int], cashflows: List[float]) -> Optional[float]:
    if not years or not cashflows or len(years) != len(cashflows):
        return None
    cumulative = 0.0
    for idx, (year, cf) in enumerate(zip(years, cashflows)):
        prev_cumulative = cumulative
        cumulative += cf
        if cumulative >= 0 and idx > 0:
            prev_year = years[idx - 1]
            if cf == 0:
                return float(year - years[0])
            fraction = (0 - prev_cumulative) / cf
            return (prev_year + fraction * (year - prev_year)) - years[0]
    return None


def _build_snapshot_from_result(
    model_cfg: ModelConfig,
    valuation_result: ValuationResult,
    scenarios: Optional[List[dict]] = None,
    sensitivities: Optional[List[dict]] = None,
) -> dict:
    cons = valuation_result.consolidated
    dcf = valuation_result.dcf_table
    cashflows = dcf["fcff"].tolist()
    if "terminal_value" in dcf.columns:
        cashflows[-1] += float(dcf["terminal_value"].fillna(0.0).iloc[-1])
    irr = _compute_irr(cashflows)
    payback = _compute_payback_years(cons.index.tolist(), cashflows)
    capex_total = -float(cons["capex_cash"].sum()) if "capex_cash" in cons.columns else None
    opex_components = [
        "sales_marketing",
        "gna",
        "royalty",
        "rd_cash",
    ]
    opex_available = [col for col in opex_components if col in cons.columns]
    opex_annual = None
    if opex_available:
        opex_annual = -float(cons[opex_available].sum(axis=1).mean())
    revenue_annual = float(cons["revenue"].mean()) if "revenue" in cons.columns else None
    snapshot = {
        "currency": model_cfg.currency,
        "npv": valuation_result.rnpv,
        "irr": irr,
        "dscr_min": None,
        "payback_years": payback,
        "capex_total": capex_total,
        "opex_annual": opex_annual,
        "revenue_annual": revenue_annual,
        "scenarios": scenarios or [],
        "sensitivities": sensitivities or [],
        "assumptions": {
            "discount_rate": model_cfg.discount_rate,
            "tax_rate": model_cfg.tax_rate,
            "working_capital_pct": model_cfg.working_capital_pct_sales,
            "inflation_rate": getattr(model_cfg, "inflation_rate", None),
        },
    }
    return snapshot


def _default_scenario_pack(portfolio: Optional[Portfolio]) -> List[dict]:
    if portfolio is None:
        return []
    base = ValuationEngine(portfolio).run()
    upside = _evaluate_portfolio_shock(
        portfolio,
        revenue_multiplier=1.15,
        cost_multiplier=0.95,
        discount_shift=-0.01,
        success_prob_multiplier=1.1,
    )
    downside = _evaluate_portfolio_shock(
        portfolio,
        revenue_multiplier=0.85,
        cost_multiplier=1.05,
        discount_shift=0.02,
        success_prob_multiplier=0.9,
    )
    scenarios = [
        {"name": "Base", "npv": base.rnpv, "irr": None},
    ]
    if upside is not None:
        scenarios.append({"name": "Upside", "npv": upside.rnpv, "irr": None})
    if downside is not None:
        scenarios.append({"name": "Downside", "npv": downside.rnpv, "irr": None})
    return scenarios


def _build_financial_excel(
    cons: pd.DataFrame,
    perf_df: pd.DataFrame,
    position_df: pd.DataFrame,
    cash_flow_df: pd.DataFrame,
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        cons.to_excel(writer, sheet_name="Consolidated forecast")
        perf_df.to_excel(writer, sheet_name="Financial performance")
        position_df.to_excel(writer, sheet_name="Financial position")
        cash_flow_df.to_excel(writer, sheet_name="Cash flows")
    return output.getvalue()


def _rag_section_outline() -> List[str]:
    return [
        "Executive Summary",
        "Project Description & Scope",
        "Market & Demand Analysis",
        "Technical & Operations",
        "Legal, Permitting & Environmental",
        "Implementation Plan",
        "Financial Analysis",
        "Risk Assessment & Mitigations",
        "Conclusion & Recommendation",
        "Appendices",
    ]


def _rag_blueprint_markdown() -> str:
    return (
        "# RAG Feasibility Study Generator\n"
        "\n"
        "A production-ready blueprint (plus reference code) for a Retrieval-Augmented Generation (RAG) "
        "system that ingests up to 1 GB of project materials and automatically drafts a comprehensive "
        "feasibility study grounded in your financial model outputs and accompanying documents.\n"
        "\n"
        "## 0) RAC: Model-Integrated Design (RAG inside the Financial Model)\n"
        "- **What changed**: The Excel workbook is the system of record and orchestrator. The "
        "Retrieval–Aggregation–Composer (RAC) service is triggered from the model to collect results "
        "directly from defined cells/ranges and to ingest up to 1 GB of external evidence.\n"
        "- **Why this pattern**: Single source of truth, fewer manual steps, and repeatable runs tied to "
        "workbook hash + timestamp.\n"
        "\n"
        "## 1) High-level Architecture\n"
        "1. Upload & Ingest: stream large files to disk, parse text, chunk, embed, store in FAISS.\n"
        "2. Financial Model Extraction: load Excel and extract standardized metrics/tables.\n"
        "3. Retrieval: dense + reranker, optional hybrid.\n"
        "4. Planning & Generation: section-by-section prompts grounded by snapshot + retrieved passages.\n"
        "5. Audit: attach provenance and snapshot metadata for reproducibility.\n"
        "\n"
        "## 2) Data Model & Financial Schema\n"
        "Store project artifacts under `projects/<project_id>/` with uploads, parsed text, index, and a "
        "financial snapshot JSON. Snapshot keys include NPV/IRR/DSCR, capex/opex, scenarios, and "
        "sensitivities.\n"
        "\n"
        "## 3) Prompt Strategy & Section Templates\n"
        "Use a strict system prompt that forbids unsupported claims and enforces inline citations. "
        "Each section receives the financial snapshot and top-k contextual passages.\n"
        "\n"
        "## 4) Reference Implementation (FastAPI + FAISS + Sentence-Transformers)\n"
        "The API exposes `/collect`, `/ingest`, and `/generate` endpoints. `/ingest` streams large "
        "uploads, `/collect` stores a validated snapshot, and `/generate` composes the feasibility "
        "study.\n"
        "\n"
        "## 5) Quality, Auditing & Reproducibility\n"
        "- Enforce citations and reject unsupported claims.\n"
        "- Record workbook hash + timestamp.\n"
        "- Run numeric sanity checks (IRR bounds, DSCR thresholds).\n"
        "\n"
        "## 6) Deployment Notes (1 GB uploads)\n"
        "- Stream uploads to disk; avoid in-memory buffers.\n"
        "- Use Nginx `client_max_body_size 1024m` and disable proxy buffering.\n"
        "- Run uvicorn with multiple workers and fast local storage.\n"
        "\n"
        "## 7) Section-specific Retrieval Queries\n"
        "- Executive Summary: decision drivers, showstoppers\n"
        "- Market: market size, demand forecast, price assumptions\n"
        "- Technical: process design, throughput, yield\n"
        "- Legal/Env: permits, EIA/ESIA, land rights\n"
        "- Implementation: schedule, capex phasing\n"
        "- Financial: NPV, IRR, DSCR, sensitivities\n"
        "- Risk/ESG: risk register, mitigations\n"
        "\n"
        "## 8) Appendices & Outputs\n"
        "Include the financial snapshot, sensitivity matrices, scenarios, and an audit trail mapping "
        "sources to citations.\n"
    )


def _build_export_payload(bundle_payload: Dict[str, Any]) -> Dict[str, Any]:
    snapshot_summary = bundle_payload["snapshot"]["financial_snapshot"]
    scenarios = snapshot_summary.get("scenarios") or []
    sensitivities = snapshot_summary.get("sensitivities") or []
    last_report = bundle_payload.get("last_report") or {}
    summary_rows = [
        {"Metric": "Project ID", "Value": bundle_payload["snapshot"]["project_id"]},
        {"Metric": "Currency", "Value": snapshot_summary.get("currency")},
        {"Metric": "NPV", "Value": snapshot_summary.get("npv")},
        {"Metric": "IRR", "Value": snapshot_summary.get("irr")},
        {"Metric": "Min DSCR", "Value": snapshot_summary.get("dscr_min")},
        {"Metric": "Payback (years)", "Value": snapshot_summary.get("payback_years")},
        {"Metric": "Total Capex", "Value": snapshot_summary.get("capex_total")},
        {"Metric": "Annual Opex", "Value": snapshot_summary.get("opex_annual")},
        {"Metric": "Annual Revenue", "Value": snapshot_summary.get("revenue_annual")},
    ]
    return {
        "summary_rows": summary_rows,
        "scenarios": scenarios,
        "sensitivities": sensitivities,
        "last_report": last_report,
        "ai_config": bundle_payload["ai_config"],
    }


def _build_chart_tables(
    valuation_result: Optional[ValuationResult],
    model_cfg: Optional[ModelConfig],
    portfolio: Optional[Portfolio],
) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    if valuation_result is None or model_cfg is None:
        return tables

    cons = valuation_result.consolidated.copy()
    cons_display = cons[["revenue", "ebitda", "fcff_after_wc"]].copy()
    cons_display.columns = ["Revenue", "EBITDA", "FCFF after WC"]
    tables["financial_statements_chart"] = cons_display
    tables["dashboard_chart"] = cons[["revenue", "ebitda", "fcff_after_wc"]]
    tables["dashboard_fcff_bar"] = cons[["fcff_after_wc"]]

    decomp_df = _compute_decomposition(cons)
    if decomp_df is not None:
        tables["analytics_decomposition"] = decomp_df

    seg_df = _build_segmentation_table(valuation_result)
    if not seg_df.empty:
        tables["analytics_segmentation"] = seg_df

    if portfolio is not None:
        base_rnpv = valuation_result.rnpv
        tornado_df = _tornado_dataframe(portfolio, base_rnpv)
        if not tornado_df.empty:
            tables["analytics_tornado"] = tornado_df

        scenarios = [
            Scenario(
                name="Base case",
                revenue_multiplier=1.0,
                cost_multiplier=1.0,
                discount_rate_shift=0.0,
                success_prob_multiplier=1.0,
            ),
            Scenario(
                name="Upside",
                revenue_multiplier=1.2,
                cost_multiplier=0.9,
                discount_rate_shift=-0.01,
                success_prob_multiplier=1.1,
            ),
            Scenario(
                name="Downside",
                revenue_multiplier=0.8,
                cost_multiplier=1.1,
                discount_rate_shift=0.01,
                success_prob_multiplier=0.9,
            ),
        ]
        scen_results = ScenarioEngine(portfolio).run_scenarios(scenarios)
        tables["scenario_results"] = scen_results

    return tables


def _build_chart_images(chart_tables: Dict[str, pd.DataFrame]) -> Dict[str, BytesIO]:
    images: Dict[str, BytesIO] = {}
    if importlib.util.find_spec("matplotlib") is None:
        return images

    import matplotlib.pyplot as plt

    def _save_fig(fig, key: str) -> None:
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        images[key] = buffer
        plt.close(fig)

    if "financial_statements_chart" in chart_tables:
        fig, ax = plt.subplots()
        chart_tables["financial_statements_chart"].plot(ax=ax)
        ax.set_title("Financial Statements Overview")
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        _save_fig(fig, "financial_statements_chart")

    if "dashboard_chart" in chart_tables:
        fig, ax = plt.subplots()
        chart_tables["dashboard_chart"].plot(ax=ax)
        ax.set_title("Dashboard Trends")
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        _save_fig(fig, "dashboard_chart")

    if "dashboard_fcff_bar" in chart_tables:
        fig, ax = plt.subplots()
        chart_tables["dashboard_fcff_bar"].plot(kind="bar", ax=ax)
        ax.set_title("FCFF After WC")
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        _save_fig(fig, "dashboard_fcff_bar")

    if "analytics_decomposition" in chart_tables:
        fig, ax = plt.subplots()
        chart_tables["analytics_decomposition"].plot(ax=ax)
        ax.set_title("Trend & Seasonality")
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        _save_fig(fig, "analytics_decomposition")

    if "analytics_segmentation" in chart_tables:
        fig, ax = plt.subplots()
        seg_df = chart_tables["analytics_segmentation"]
        seg_df.set_index("Product")["Revenue share"].plot(kind="bar", ax=ax)
        ax.set_title("Revenue Share by Product")
        ax.set_xlabel("Product")
        ax.set_ylabel("Revenue Share")
        _save_fig(fig, "analytics_segmentation")

    if "analytics_tornado" in chart_tables:
        fig, ax = plt.subplots()
        tornado_df = chart_tables["analytics_tornado"].sort_values("Delta")
        ax.barh(tornado_df["Driver"], tornado_df["Delta"])
        ax.set_title("Tornado Impact")
        ax.set_xlabel("Delta")
        _save_fig(fig, "analytics_tornado")

    if "scenario_results" in chart_tables:
        fig, ax = plt.subplots()
        scen_df = chart_tables["scenario_results"]
        ax.bar(scen_df["scenario"], scen_df["rnpv"])
        ax.set_title("Scenario rNPV Comparison")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("rNPV")
        _save_fig(fig, "scenario_results")

    return images


def _build_excel_export(payload: Dict[str, Any]) -> io.BytesIO:
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        pd.DataFrame(payload["summary_rows"]).to_excel(writer, index=False, sheet_name="Summary")
        if payload["scenarios"]:
            pd.DataFrame(payload["scenarios"]).to_excel(writer, index=False, sheet_name="Scenarios")
        if payload["sensitivities"]:
            pd.DataFrame(payload["sensitivities"]).to_excel(writer, index=False, sheet_name="Sensitivities")
        if payload["last_report"]:
            pd.DataFrame(
                [{"Section": key, "Content": value} for key, value in payload["last_report"].items()]
            ).to_excel(writer, index=False, sheet_name="Last Report")
        chart_tables = payload.get("chart_tables", {})
        for sheet_name, table in chart_tables.items():
            if not table.empty:
                safe_name = sheet_name[:31]
                table.to_excel(writer, index=True, sheet_name=safe_name)
    excel_buffer.seek(0)
    return excel_buffer


def _build_word_export(payload: Dict[str, Any]) -> io.BytesIO:
    Document = importlib.import_module("docx").Document
    docx_buffer = io.BytesIO()
    document = Document()
    document.add_heading("Business Plan Bundle", level=1)
    document.add_paragraph(
        "This bundle summarizes the financial snapshot and the AI configuration used for the "
        "RAG Assistant report generation."
    )
    document.add_heading("Financial Snapshot", level=2)
    for row in payload["summary_rows"]:
        document.add_paragraph(f"{row['Metric']}: {row['Value']}")
    if payload["scenarios"]:
        document.add_heading("Scenarios", level=2)
        for scenario in payload["scenarios"]:
            document.add_paragraph(json.dumps(scenario, ensure_ascii=False))
    if payload["sensitivities"]:
        document.add_heading("Sensitivities", level=2)
        for sensitivity in payload["sensitivities"]:
            document.add_paragraph(json.dumps(sensitivity, ensure_ascii=False))
    document.add_heading("AI Configuration", level=2)
    for key, value in payload["ai_config"].items():
        document.add_paragraph(f"{key}: {value}")
    if payload["last_report"]:
        document.add_heading("Last Report", level=2)
        for key, value in payload["last_report"].items():
            document.add_paragraph(f"{key}: {value}")
    if payload.get("chart_images"):
        document.add_heading("Financial Statements Charts", level=2)
        if payload["chart_images"].get("financial_statements_chart"):
            document.add_picture(payload["chart_images"]["financial_statements_chart"])
        document.add_heading("Dashboard Charts", level=2)
        if payload["chart_images"].get("dashboard_chart"):
            document.add_picture(payload["chart_images"]["dashboard_chart"])
        if payload["chart_images"].get("dashboard_fcff_bar"):
            document.add_picture(payload["chart_images"]["dashboard_fcff_bar"])
        document.add_heading("Advanced Analytics Charts", level=2)
        if payload["chart_images"].get("analytics_decomposition"):
            document.add_picture(payload["chart_images"]["analytics_decomposition"])
        if payload["chart_images"].get("analytics_segmentation"):
            document.add_picture(payload["chart_images"]["analytics_segmentation"])
        if payload["chart_images"].get("analytics_tornado"):
            document.add_picture(payload["chart_images"]["analytics_tornado"])
        document.add_heading("Scenario Analysis Charts", level=2)
        if payload["chart_images"].get("scenario_results"):
            document.add_picture(payload["chart_images"]["scenario_results"])
    document.save(docx_buffer)
    docx_buffer.seek(0)
    return docx_buffer


def _build_pdf_export(payload: Dict[str, Any]) -> io.BytesIO:
    canvas = importlib.import_module("reportlab.pdfgen.canvas")
    image_reader = importlib.import_module("reportlab.lib.utils").ImageReader
    pdf_buffer = io.BytesIO()
    pdf_canvas = canvas.Canvas(pdf_buffer)
    pdf_canvas.setFont("Helvetica-Bold", 14)
    pdf_canvas.drawString(72, 770, "Business Plan Bundle")
    pdf_canvas.setFont("Helvetica", 11)
    y_position = 740
    pdf_canvas.drawString(72, y_position, "Financial Snapshot")
    y_position -= 18
    for row in payload["summary_rows"]:
        pdf_canvas.drawString(72, y_position, f"{row['Metric']}: {row['Value']}")
        y_position -= 16
        if y_position <= 72:
            pdf_canvas.showPage()
            pdf_canvas.setFont("Helvetica", 11)
            y_position = 770
    if payload["scenarios"]:
        y_position -= 6
        if y_position <= 72:
            pdf_canvas.showPage()
            pdf_canvas.setFont("Helvetica", 11)
            y_position = 770
        pdf_canvas.drawString(72, y_position, "Scenarios")
        y_position -= 18
        for scenario in payload["scenarios"]:
            pdf_canvas.drawString(72, y_position, json.dumps(scenario, ensure_ascii=False))
            y_position -= 16
            if y_position <= 72:
                pdf_canvas.showPage()
                pdf_canvas.setFont("Helvetica", 11)
                y_position = 770
    if payload["sensitivities"]:
        y_position -= 6
        if y_position <= 72:
            pdf_canvas.showPage()
            pdf_canvas.setFont("Helvetica", 11)
            y_position = 770
        pdf_canvas.drawString(72, y_position, "Sensitivities")
        y_position -= 18
        for sensitivity in payload["sensitivities"]:
            pdf_canvas.drawString(72, y_position, json.dumps(sensitivity, ensure_ascii=False))
            y_position -= 16
            if y_position <= 72:
                pdf_canvas.showPage()
                pdf_canvas.setFont("Helvetica", 11)
                y_position = 770
    y_position -= 6
    if y_position <= 72:
        pdf_canvas.showPage()
        pdf_canvas.setFont("Helvetica", 11)
        y_position = 770
    pdf_canvas.drawString(72, y_position, "AI Configuration")
    y_position -= 18
    for key, value in payload["ai_config"].items():
        pdf_canvas.drawString(72, y_position, f"{key}: {value}")
        y_position -= 16
        if y_position <= 72:
            pdf_canvas.showPage()
            pdf_canvas.setFont("Helvetica", 11)
            y_position = 770
    if payload["last_report"]:
        y_position -= 6
        if y_position <= 72:
            pdf_canvas.showPage()
            pdf_canvas.setFont("Helvetica", 11)
            y_position = 770
        pdf_canvas.drawString(72, y_position, "Last Report")
        y_position -= 18
        for key, value in payload["last_report"].items():
            pdf_canvas.drawString(72, y_position, f"{key}: {value}")
            y_position -= 16
            if y_position <= 72:
                pdf_canvas.showPage()
                pdf_canvas.setFont("Helvetica", 11)
                y_position = 770
    chart_images = payload.get("chart_images", {})
    if chart_images:
        pdf_canvas.showPage()
        pdf_canvas.setFont("Helvetica-Bold", 14)
        pdf_canvas.drawString(72, 770, "Charts & Graphs")
        y_position = 740
        pdf_canvas.setFont("Helvetica", 11)

        def _draw_image(image_key: str, title: str) -> None:
            nonlocal y_position
            image = chart_images.get(image_key)
            if not image:
                return
            if y_position <= 180:
                pdf_canvas.showPage()
                pdf_canvas.setFont("Helvetica-Bold", 14)
                pdf_canvas.drawString(72, 770, "Charts & Graphs (cont.)")
                pdf_canvas.setFont("Helvetica", 11)
                y_position = 740
            pdf_canvas.drawString(72, y_position, title)
            y_position -= 14
            pdf_canvas.drawImage(image_reader(image), 72, y_position - 120, width=450, height=120)
            y_position -= 140

        _draw_image("financial_statements_chart", "Financial Statements")
        _draw_image("dashboard_chart", "Dashboard Trends")
        _draw_image("dashboard_fcff_bar", "Dashboard FCFF")
        _draw_image("analytics_decomposition", "Analytics Decomposition")
        _draw_image("analytics_segmentation", "Analytics Segmentation")
        _draw_image("analytics_tornado", "Analytics Tornado")
        _draw_image("scenario_results", "Scenario Analysis")
    pdf_canvas.save()
    pdf_buffer.seek(0)
    return pdf_buffer


def _build_export_buffers(payload: Dict[str, Any]) -> Tuple[Dict[str, io.BytesIO], List[str]]:
    buffers: Dict[str, io.BytesIO] = {}
    warnings: List[str] = []
    if importlib.util.find_spec("openpyxl") is not None:
        buffers["excel"] = _build_excel_export(payload)
    else:
        warnings.append("Excel export unavailable: install openpyxl.")

    if importlib.util.find_spec("docx") is not None:
        buffers["docx"] = _build_word_export(payload)
    else:
        warnings.append("Word export unavailable: install python-docx.")

    if importlib.util.find_spec("reportlab") is not None:
        buffers["pdf"] = _build_pdf_export(payload)
    else:
        warnings.append("PDF export unavailable: install reportlab.")

    if payload.get("chart_tables") and importlib.util.find_spec("matplotlib") is None:
        warnings.append("Charts export unavailable: install matplotlib to embed plots.")

    return buffers, warnings


def _render_export_downloads(
    buffers: Dict[str, io.BytesIO],
    *,
    project_id: str,
    rag_key_prefix: str,
) -> None:
    if "excel" in buffers:
        st.download_button(
            "Download business plan (Excel)",
            data=buffers["excel"],
            file_name=f"{project_id}_business_plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"{rag_key_prefix}_bundle_download_excel",
        )
    if "docx" in buffers:
        st.download_button(
            "Download business plan (Word)",
            data=buffers["docx"],
            file_name=f"{project_id}_business_plan.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            key=f"{rag_key_prefix}_bundle_download_docx",
        )
    if "pdf" in buffers:
        st.download_button(
            "Download business plan (PDF)",
            data=buffers["pdf"],
            file_name=f"{project_id}_business_plan.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"{rag_key_prefix}_bundle_download_pdf",
        )


def _render_rag_assistant_page() -> None:
    st.subheader("RAG Assistant")

    rag_key_prefix = "rag_assistant"
    st.markdown("## Upload reference documents")
    project_id = st.session_state.get(
        f"{rag_key_prefix}_project_id",
        os.environ.get("RAG_PROJECT_ID", "default-project"),
    )
    rag_host = st.session_state.get(
        f"{rag_key_prefix}_rag_host",
        os.environ.get("RAG_HOST", "http://localhost:8000"),
    )
    st.caption("Set RAG_HOST and RAG_PROJECT_ID env vars to override defaults.")

    uploads = st.file_uploader(
        "Upload reference documents",
        accept_multiple_files=True,
        key=f"{rag_key_prefix}_uploads",
    )
    if uploads:
        st.caption(f"{len(uploads)} document(s) ready for indexing.")

    st.markdown("## AI & Machine Learning Configuration")
    enable_ai = st.checkbox("Enable AI enhancements", value=True, key=f"{rag_key_prefix}_enable_ai")
    provider_options_key = f"{rag_key_prefix}_provider_options"
    if provider_options_key not in st.session_state:
        st.session_state[provider_options_key] = [
            "OpenAI",
            "Azure OpenAI",
            "Anthropic",
            "Vertex",
            "Custom",
        ]
    provider_options = st.session_state[provider_options_key]
    provider = st.selectbox(
        "Provider",
        provider_options,
        key=f"{rag_key_prefix}_provider",
    )
    custom_provider = ""
    if provider == "Custom":
        custom_provider = st.text_input(
            "Custom provider name",
            placeholder="Enter a provider name (e.g., Cohere, Mistral)",
            key=f"{rag_key_prefix}_custom_provider",
        )
        add_provider = st.button("Add provider", key=f"{rag_key_prefix}_add_provider")
        if add_provider and custom_provider:
            updated_providers = [*provider_options]
            if custom_provider not in updated_providers:
                updated_providers.insert(-1, custom_provider)
                st.session_state[provider_options_key] = updated_providers
                st.success(f"Added provider: {custom_provider}")
            else:
                st.info("That provider is already available.")
    model_name = st.text_input(
        "Model",
        value="gpt-4o-mini",
        key=f"{rag_key_prefix}_model",
    )
    forecast_horizon = st.number_input(
        "Forecast horizon (years)",
        min_value=1,
        max_value=50,
        value=10,
        key=f"{rag_key_prefix}_forecast_horizon",
    )
    ml_methods = st.multiselect(
        "Machine learning method",
        ["Linear regression", "Compound annual growth", "ARIMA", "Prophet", "LSTM"],
        default=["Linear regression"],
        key=f"{rag_key_prefix}_ml_methods",
    )
    generative_features = st.multiselect(
        "Generative features",
        ["Executive summary", "Risk review", "Cash flow highlights", "ESG review", "Market overview"],
        default=["Executive summary", "Risk review", "Cash flow highlights"],
        key=f"{rag_key_prefix}_gen_features",
    )
    api_key = st.text_input(
        "API key",
        type="password",
        key=f"{rag_key_prefix}_api_key",
    )
    if st.button("Save AI configuration", key=f"{rag_key_prefix}_save_config"):
        st.session_state["rag_ai_config"] = {
            "enable_ai": enable_ai,
            "provider": custom_provider or provider,
            "model": model_name,
            "forecast_horizon": forecast_horizon,
            "ml_methods": ml_methods,
            "generative_features": generative_features,
            "api_key_set": bool(api_key),
        }
        st.success("AI configuration saved.")

    st.markdown("## AI Insights")
    model_cfg = st.session_state.get("model_config")
    valuation_result = st.session_state.get("valuation_result")
    portfolio = st.session_state.get("portfolio")

    if "rag_snapshot" not in st.session_state:
        if model_cfg is not None and valuation_result is not None:
            default_scenarios = _default_scenario_pack(portfolio)
            st.session_state["rag_snapshot"] = _build_snapshot_from_result(
                model_cfg,
                valuation_result,
                scenarios=default_scenarios,
            )
        else:
            st.session_state["rag_snapshot"] = {
                "currency": "USD",
                "npv": None,
                "irr": None,
                "dscr_min": None,
                "payback_years": None,
                "capex_total": None,
                "opex_annual": None,
                "revenue_annual": None,
                "scenarios": [],
                "sensitivities": [],
                "assumptions": {},
            }

    if st.button("Refresh snapshot from latest model", key=f"{rag_key_prefix}_refresh_snapshot"):
        if model_cfg is None or valuation_result is None:
            st.warning("Run the model workspace to generate a snapshot.")
        else:
            st.session_state["rag_snapshot"] = _build_snapshot_from_result(
                model_cfg,
                valuation_result,
                scenarios=_default_scenario_pack(portfolio),
            )

    snapshot_state = st.session_state["rag_snapshot"]
    with st.expander("Snapshot inputs", expanded=False):
        snap_cols = st.columns(3)
        snapshot_state["currency"] = snap_cols[0].text_input(
            "Currency",
            value=snapshot_state.get("currency") or "USD",
            key=f"{rag_key_prefix}_currency",
        )
        snapshot_state["npv"] = snap_cols[1].number_input(
            "NPV",
            value=float(snapshot_state["npv"]) if snapshot_state.get("npv") is not None else 0.0,
            step=1000000.0,
            key=f"{rag_key_prefix}_npv",
        )
        snapshot_state["irr"] = snap_cols[2].number_input(
            "IRR",
            value=float(snapshot_state["irr"]) if snapshot_state.get("irr") is not None else 0.0,
            step=0.01,
            format="%.4f",
            key=f"{rag_key_prefix}_irr",
        )

        snap_cols2 = st.columns(3)
        snapshot_state["dscr_min"] = snap_cols2[0].number_input(
            "Minimum DSCR",
            value=float(snapshot_state.get("dscr_min") or 0.0),
            step=0.1,
            format="%.2f",
            key=f"{rag_key_prefix}_dscr_min",
        )
        snapshot_state["payback_years"] = snap_cols2[1].number_input(
            "Payback (years)",
            value=float(snapshot_state.get("payback_years") or 0.0),
            step=0.1,
            format="%.2f",
            key=f"{rag_key_prefix}_payback_years",
        )
        snapshot_state["capex_total"] = snap_cols2[2].number_input(
            "Total capex",
            value=float(snapshot_state.get("capex_total") or 0.0),
            step=1000000.0,
            key=f"{rag_key_prefix}_capex_total",
        )

        snap_cols3 = st.columns(2)
        snapshot_state["opex_annual"] = snap_cols3[0].number_input(
            "Annual opex",
            value=float(snapshot_state.get("opex_annual") or 0.0),
            step=100000.0,
            key=f"{rag_key_prefix}_opex_annual",
        )
        snapshot_state["revenue_annual"] = snap_cols3[1].number_input(
            "Annual revenue",
            value=float(snapshot_state.get("revenue_annual") or 0.0),
            step=100000.0,
            key=f"{rag_key_prefix}_revenue_annual",
        )

        scenarios_df = pd.DataFrame(snapshot_state.get("scenarios") or [])
        scenarios_df = st.data_editor(
            scenarios_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Scenario"),
                "npv": st.column_config.NumberColumn("NPV"),
                "irr": st.column_config.NumberColumn("IRR"),
            },
            key=f"{rag_key_prefix}_scenarios_editor",
        )
        snapshot_state["scenarios"] = scenarios_df.to_dict(orient="records")

    snapshot_payload = {
        "project_id": project_id,
        "financial_snapshot": snapshot_state,
        "workbook_hash": snapshot_state.get("workbook_hash"),
    }

    has_uploads = bool(uploads)
    has_indexed = bool(st.session_state.get("rag_last_ingest"))
    insight_cols = st.columns(3)
    if insight_cols[0].button(
        "Index documents",
        key=f"{rag_key_prefix}_index_docs",
        disabled=not has_uploads,
    ):
        files = [("files", (u.name, u.getvalue(), u.type or "application/octet-stream")) for u in uploads]
        try:
            response = requests.post(
                f"{rag_host.rstrip('/')}/ingest",
                params={"project_id": project_id},
                files=files,
                timeout=120,
            )
            response.raise_for_status()
            st.session_state["rag_last_ingest"] = response.json()
            st.success(response.json())
        except requests.RequestException as exc:
            if isinstance(exc, requests.ConnectionError):
                st.warning(
                    "RAG service unreachable. Start the service or update RAG_HOST to a reachable URL."
                )
            else:
                st.error(f"Failed to ingest files: {exc}")
    if not has_uploads:
        st.caption("Upload reference documents to enable indexing.")

    if insight_cols[1].button("Clear indexed documents", key=f"{rag_key_prefix}_clear_index"):
        st.session_state.pop("rag_last_ingest", None)
        st.session_state.pop("rag_last_report", None)
        st.info("Local index metadata cleared. Clear the backend index from the service if needed.")

    if insight_cols[2].button(
        "Run AI insights",
        key=f"{rag_key_prefix}_run_ai",
        disabled=not has_indexed,
    ):
        outline = _rag_section_outline()
        try:
            response = requests.post(
                f"{rag_host.rstrip('/')}/generate",
                json={"project_id": project_id, "section_outline": outline},
                timeout=180,
            )
            response.raise_for_status()
            st.session_state["rag_last_report"] = response.json()
            st.success(response.json())
        except requests.RequestException as exc:
            if isinstance(exc, requests.ConnectionError):
                st.warning(
                    "RAG service unreachable. Start the service or update RAG_HOST to a reachable URL."
                )
            else:
                st.error(f"Failed to run AI insights: {exc}")
    if not has_indexed:
        st.caption("Index documents before running AI insights.")

    question = st.text_input(
        "Ask a question",
        key=f"{rag_key_prefix}_question",
    )
    if st.button("Search", key=f"{rag_key_prefix}_search"):
        if not question:
            st.warning("Enter a question to search.")
        else:
            st.info("Search requires a backend endpoint (e.g. /search). Configure it to enable results.")

    st.markdown("## Business Plan Downloads")
    st.caption(
        "Generate a consolidated business plan bundle that includes the full financial report and snapshot."
    )
    if st.button("Prepare business plan bundle", key=f"{rag_key_prefix}_bundle"):
        st.session_state["rag_bundle_ready"] = True
        st.success("Bundle ready. Download below.")

    if st.session_state.get("rag_bundle_ready"):
        bundle_payload = {
            "snapshot": snapshot_payload,
            "ai_config": st.session_state.get("rag_ai_config", {}),
            "last_report": st.session_state.get("rag_last_report", {}),
        }
        export_payload = _build_export_payload(bundle_payload)
        chart_tables = _build_chart_tables(
            st.session_state.get("valuation_result"),
            st.session_state.get("model_config"),
            st.session_state.get("portfolio"),
        )
        export_payload["chart_tables"] = chart_tables
        export_payload["chart_images"] = _build_chart_images(chart_tables)
        export_buffers, export_warnings = _build_export_buffers(export_payload)

        for warning in export_warnings:
            st.warning(warning)

        _render_export_downloads(
            export_buffers,
            project_id=project_id,
            rag_key_prefix=rag_key_prefix,
        )

def main() -> None:
    st.set_page_config(
        page_title="Biotech Financial Model",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("Biotech")
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
        rag_tab,
    ) = st.tabs(
        [
            "Model configuration",
            "Financial statements",
            "Dashboard",
            "Advanced analytics",
            "Scenario analysis",
            "VC helper",
            "RAG Assistant",
        ]
    )

    with config_tab:
        st.subheader("Model assumptions")

        with st.expander("General assumptions", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                first_year = st.number_input("First forecast year", value=2024)
                n_years = st.number_input("Number of years", min_value=5, max_value=40, value=25)
                currency = st.text_input("Currency", value="USD")
            with col2:
                tax_rate = st.slider("Tax rate", min_value=0.0, max_value=0.35, value=0.25)
                wc_pct = st.slider("Working capital (% sales)", 0.0, 0.3, 0.08)
            with col3:
                inflation = st.number_input("Inflation assumption", value=0.02, min_value=0.0, max_value=0.25, step=0.005)
                base_fx = st.text_input("Reporting FX pair", value="USD/EUR")
            st.caption("Set the macro baseline for the consolidated forecast and disclosures.")

        with st.expander("Forecast assumptions", expanded=True):
            ramp_df = _render_schedule_editor("Sales ramp schedule", "sales_ramp_schedule")
            ramp_df = ramp_df.sort_values("Year offset")
            if ramp_df.empty:
                st.warning("Ramp schedule empty. Reverting to default values.")
                ramp = _default_ramp_schedule()["Ramp factor"].tolist()
            else:
                ramp = ramp_df["Ramp factor"].astype(float).tolist()
            st.caption("Ramp factors feed revenue build-ups across every product.")

        with st.expander("Vaccine sales"):
            vaccine_df = _render_product_assumption_table(
                session_key="vaccine_sales_table",
                default_factory=lambda: _default_vaccine_sales_table(int(first_year), int(n_years)),
                blank_row_factory=lambda df: _blank_vaccine_sales_row(df, int(first_year)),
                id_column=None,
                name_column="Year",
                column_config={
                    "Year": st.column_config.NumberColumn("Year", step=1),
                    "Doses (M)": st.column_config.NumberColumn("Doses (M)", min_value=0.0, step=0.5),
                    "Price per dose": st.column_config.NumberColumn(
                        "Price per dose", min_value=0.0, step=1.0
                    ),
                },
            )
            doses = pd.to_numeric(vaccine_df.get("Doses (M)", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            price = pd.to_numeric(vaccine_df.get("Price per dose", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            vaccine_df["Implied revenue"] = doses * 1e6 * price
            st.session_state["vaccine_sales_table"] = vaccine_df
            st.metric(f"{int(n_years)}-year vaccine sales", f"{vaccine_df['Implied revenue'].sum():,.0f}")

        with st.expander("Uses and sources of funds"):
            uses_col, sources_col = st.columns(2)
            with uses_col:
                st.markdown("**Uses**")
                uses_df = _render_product_assumption_table(
                    session_key="uses_table",
                    default_factory=_default_uses_table,
                    blank_row_factory=_blank_use_row,
                    id_column=None,
                    name_column="Item",
                    column_config={
                        "Amount": st.column_config.NumberColumn("Amount", step=1_000_000.0),
                    },
                )
                uses_total = float(uses_df.get("Amount", pd.Series(dtype=float)).sum())
                st.metric("Total uses", f"{uses_total:,.0f}")
            with sources_col:
                st.markdown("**Sources**")
                sources_df = _render_product_assumption_table(
                    session_key="sources_table",
                    default_factory=_default_sources_table,
                    blank_row_factory=_blank_source_row,
                    id_column=None,
                    name_column="Item",
                    column_config={
                        "Amount": st.column_config.NumberColumn("Amount", step=1_000_000.0),
                    },
                )
                sources_total = float(sources_df.get("Amount", pd.Series(dtype=float)).sum())
                st.metric("Total sources", f"{sources_total:,.0f}")
            delta = sources_total - uses_total
            st.info(f"Funding gap (sources - uses): {delta:,.0f}")

        with st.expander("Risk-adjusted DCF valuation method - assumptions"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                discount_rate = st.slider("Discount rate", min_value=0.02, max_value=0.30, value=0.10)
            with col_b:
                ev_multiple = st.slider("Terminal EV/EBITDA multiple", 2.0, 30.0, 8.0)
            with col_c:
                risk_buffer = st.number_input(
                    "Additional risk premium", min_value=0.0, max_value=0.20, value=0.0, step=0.01
                )
            st.caption("Discount rate + premium governs the rNPV and terminal value." )

        with st.expander("Funding required"):
            funding_required = st.number_input(
                "Total funding required", value=250_000_000.0, step=5_000_000.0, format="%0.0f"
            )

        with st.expander("Shareholders / Investors"):
            shareholders_df = _render_product_assumption_table(
                session_key="shareholders_table",
                default_factory=_default_shareholders_table,
                blank_row_factory=_blank_shareholder_row,
                id_column=None,
                name_column="Shareholder",
                column_config={
                    "Ownership %": st.column_config.NumberColumn(
                        "Ownership %", min_value=0.0, max_value=1.0, step=0.01
                    ),
                    "Investment": st.column_config.NumberColumn("Investment", step=1_000_000.0),
                },
            )
            st.metric("Total ownership reported", f"{shareholders_df['Ownership %'].sum():.0%}")

        with st.expander("Relevant market sizes"):
            market_df = _render_product_assumption_table(
                session_key="market_sizes_table",
                default_factory=_default_market_sizes_table,
                blank_row_factory=_blank_relevant_market_row,
                id_column=None,
                name_column="Segment",
                column_config={
                    "Value": st.column_config.NumberColumn("Value", step=1_000_000.0),
                },
            )

        with st.expander("New equity issued"):
            new_equity = st.number_input(
                "Planned new equity", value=200_000_000.0, step=5_000_000.0, format="%0.0f"
            )

        with st.expander("Selectors"):
            selector_choices = st.multiselect(
                "Tag this run with selectors", options=SELECTOR_OPTIONS, default=["Base case"]
            )
            st.write("Active selectors:", ", ".join(selector_choices) or "None")

        effective_discount_rate = float(min(0.40, discount_rate + risk_buffer))
        model_cfg = ModelConfig(
            first_year=int(first_year),
            n_years=int(n_years),
            currency=currency,
            discount_rate=effective_discount_rate,
            tax_rate=float(tax_rate),
            working_capital_pct_sales=float(wc_pct),
            ev_ebitda_multiple=float(ev_multiple),
            sales_ramp_factors=ramp,
        )

        st.subheader("Product assumptions")

        dev_df = _render_product_assumption_table(
            session_key="vaccine_development_table",
            default_factory=lambda: _default_vaccine_development_table(int(first_year)),
            blank_row_factory=lambda df: _blank_vaccine_development_row(df, int(first_year)),
            column_config={
                "Stage": st.column_config.SelectboxColumn("Stage", options=STAGE_OPTIONS),
                "Consolidation": st.column_config.CheckboxColumn("Consolidate", default=True),
                "Success Probability %": st.column_config.NumberColumn(
                    "Success Probability %", min_value=0.0, max_value=100.0, step=1.0
                ),
            },
        )
        entry_calc = _coerce_numeric(dev_df.get("First year forecast", pd.Series(dtype=float))) + _coerce_numeric(
            dev_df.get("Time to market", pd.Series(dtype=float))
        )
        if "Market entry year" not in dev_df.columns:
            dev_df["Market entry year"] = entry_calc
        else:
            missing_entry = dev_df["Market entry year"].isna()
            dev_df.loc[missing_entry, "Market entry year"] = entry_calc[missing_entry]
        if "End patent year" not in dev_df.columns:
            dev_df["End patent year"] = dev_df["Market entry year"] + _coerce_numeric(
                dev_df.get("Patent duration years", pd.Series(dtype=float)), default=0
            ) - 1
        else:
            mask_patent = dev_df["End patent year"].isna()
            dev_df.loc[mask_patent, "End patent year"] = (
                dev_df.loc[mask_patent, "Market entry year"]
                + _coerce_numeric(
                    dev_df.loc[mask_patent, "Patent duration years"],
                    default=0,
                )
                - 1
            )
        st.session_state["vaccine_development_table"] = dev_df
        st.caption("Track each vaccine's readiness, probability of success, and patent end year.")

        with st.expander("Vaccine market size estimation", expanded=True):
            market_size_df = _render_product_assumption_table(
                session_key="market_size_estimation",
                default_factory=_default_market_size_estimation_table,
                blank_row_factory=_blank_market_size_row,
            )
            market_size = _coerce_numeric(market_size_df.get("Market size (# customers)", pd.Series(dtype=float)))
            avg_spend = _coerce_numeric(
                market_size_df.get("Average spend (USD/customer)", pd.Series(dtype=float))
            )
            tam = market_size * avg_spend
            market_size_df["Total Addressable Market Size (USD)"] = tam
            sam_pct = _coerce_numeric(
                market_size_df.get("Serviceable Available Market (% TAM)", pd.Series(dtype=float))
            )
            market_size_df["Serviceable Available Market (USD)"] = tam * sam_pct.div(100)
            som_pct = _coerce_numeric(
                market_size_df.get("Serviceable Obtainable Market (%)", pd.Series(dtype=float))
            )
            market_size_df["Serviceable Obtainable Market (USD)"] = tam * som_pct.div(100)
            st.session_state["market_size_estimation"] = market_size_df
            market_size_display = market_size_df[[
                "ID_vaccine",
                "Vaccine name",
                "Total Addressable Market Size (USD)",
                "Serviceable Available Market (USD)",
                "Serviceable Obtainable Market (USD)",
            ]]
            st.dataframe(
                market_size_display.style.format(
                    {
                        "Total Addressable Market Size (USD)": "{:.0f}",
                        "Serviceable Available Market (USD)": "{:.0f}",
                        "Serviceable Obtainable Market (USD)": "{:.0f}",
                    }
                )
            )

        with st.expander("Vaccines revenue estimation", expanded=True):
            revenue_df = _render_product_assumption_table(
                session_key="vaccine_revenue_table",
                default_factory=_default_vaccine_revenue_table,
                blank_row_factory=_blank_vaccine_revenue_row,
            )
            patent_customers = _coerce_numeric(
                revenue_df.get("Patent customers per year", pd.Series(dtype=float))
            )
            patent_price = _coerce_numeric(
                revenue_df.get("Patent price (USD/customer)", pd.Series(dtype=float))
            )
            revenue_df["Patent revenue target (USD)"] = patent_customers * patent_price
            cust_adj = _coerce_numeric(
                revenue_df.get("Post patent customer adj. %", pd.Series(dtype=float))
            ).div(100).replace(0, np.nan)
            price_adj = _coerce_numeric(
                revenue_df.get("Post patent price adj. %", pd.Series(dtype=float))
            ).div(100).replace(0, np.nan)
            if "Post patent customers per year" not in revenue_df.columns:
                revenue_df["Post patent customers per year"] = patent_customers * cust_adj.fillna(1.0)
            else:
                mask_missing = revenue_df["Post patent customers per year"].isna()
                revenue_df.loc[mask_missing, "Post patent customers per year"] = (
                    patent_customers[mask_missing] * cust_adj.fillna(1.0)[mask_missing]
                )
            if "Post patent price (USD/customer)" not in revenue_df.columns:
                revenue_df["Post patent price (USD/customer)"] = patent_price * price_adj.fillna(1.0)
            else:
                mask_price = revenue_df["Post patent price (USD/customer)"].isna()
                revenue_df.loc[mask_price, "Post patent price (USD/customer)"] = (
                    patent_price[mask_price] * price_adj.fillna(1.0)[mask_price]
                )
            revenue_df["Post patent revenue target (USD)"] = (
                _coerce_numeric(revenue_df["Post patent customers per year"], 0)
                * _coerce_numeric(revenue_df["Post patent price (USD/customer)"], 0)
            )
            st.session_state["vaccine_revenue_table"] = revenue_df
            revenue_display = revenue_df[[
                "ID_vaccine",
                "Vaccine name",
                "Patent revenue target (USD)",
                "Post patent revenue target (USD)",
            ]]
            st.dataframe(
                revenue_display.style.format(
                    {
                        "Patent revenue target (USD)": "{:.0f}",
                        "Post patent revenue target (USD)": "{:.0f}",
                    }
                )
            )

        with st.expander("Vaccine cost assumptions", expanded=True):
            cost_df = _render_product_assumption_table(
                session_key="vaccine_cost_table",
                default_factory=_default_vaccine_cost_table,
                blank_row_factory=_blank_vaccine_cost_row,
            )
            cogs_patent = _coerce_numeric(cost_df.get("COGS patent % of sales", pd.Series(dtype=float)))
            cogs_post = _coerce_numeric(cost_df.get("COGS post % of sales", pd.Series(dtype=float)))
            marketing_pct = _coerce_numeric(cost_df.get("Marketing annual % of sales", pd.Series(dtype=float)))
            royalty_pct = _coerce_numeric(cost_df.get("Royalties cost % of sales", pd.Series(dtype=float)))
            gna_cols = [
                "Indirect staff cost (USD)",
                "Electricity (USD)",
                "Depreciation (USD)",
                "Interest & amortization (USD)",
            ]
            cost_df["G&A total (USD)"] = cost_df[gna_cols].sum(axis=1)
            cost_df["Patent operating cost %"] = cogs_patent + marketing_pct + royalty_pct
            cost_df["Post operating cost %"] = cogs_post + marketing_pct + royalty_pct
            st.session_state["vaccine_cost_table"] = cost_df
            cost_display = cost_df[
                [
                    "ID_vaccine",
                    "Vaccine name",
                    "COGS patent % of sales",
                    "COGS post % of sales",
                    "Marketing annual % of sales",
                    "Marketing launch cost (USD)",
                    "Royalties cost % of sales",
                    "G&A total (USD)",
                    "Patent operating cost %",
                    "Post operating cost %",
                ]
            ]
            percent_cols = [
                "COGS patent % of sales",
                "COGS post % of sales",
                "Marketing annual % of sales",
                "Royalties cost % of sales",
                "Patent operating cost %",
                "Post operating cost %",
            ]
            percent_fmt = {col: "{:.1f}%" for col in percent_cols if col in cost_display.columns}
            currency_fmt = {
                col: "{:.0f}"
                for col in ["Marketing launch cost (USD)", "G&A total (USD)"]
                if col in cost_display.columns
            }
            st.dataframe(cost_display.style.format({**percent_fmt, **currency_fmt}))

        with st.expander("Vaccines research & development (R&D)", expanded=True):
            rd_df = _render_product_assumption_table(
                session_key="vaccine_rd_table",
                default_factory=_default_vaccine_rd_table,
                blank_row_factory=_blank_vaccine_rd_row,
            )
            rd_df["Pre-GTM total (USD)"] = _coerce_numeric(
                rd_df.get("Pre-GTM spent to date (USD)", pd.Series(dtype=float))
            ) + _coerce_numeric(rd_df.get("Pre-GTM remaining (USD)", pd.Series(dtype=float)))
            st.session_state["vaccine_rd_table"] = rd_df
            rd_display = rd_df[
                [
                    "ID_vaccine",
                    "Vaccine name",
                    "Cost accounting (capitalisation)",
                    "Pre-GTM spent to date (USD)",
                    "Pre-GTM remaining (USD)",
                    "Pre-GTM total (USD)",
                    "Post-GTM annual cost (USD/year)",
                ]
            ]
            rd_fmt = {
                col: "{:.0f}"
                for col in rd_display.columns
                if col not in ["ID_vaccine", "Vaccine name", "Cost accounting (capitalisation)"]
            }
            st.dataframe(rd_display.style.format(rd_fmt))

        with st.expander("Vaccine CAPEX assumptions", expanded=True):
            capex_df = _render_product_assumption_table(
                session_key="vaccine_capex_table",
                default_factory=_default_vaccine_capex_table,
                blank_row_factory=_blank_vaccine_capex_row,
            )
            capex_df["Total Pre-GTM capex (USD)"] = _coerce_numeric(
                capex_df.get("Pre-GTM capex spent (USD)", pd.Series(dtype=float))
            ) + _coerce_numeric(capex_df.get("Pre-GTM capex remaining (USD)", pd.Series(dtype=float)))
            st.session_state["vaccine_capex_table"] = capex_df
            capex_display = capex_df[
                [
                    "ID_vaccine",
                    "Vaccine name",
                    "Pre-GTM capex spent (USD)",
                    "Pre-GTM capex remaining (USD)",
                    "Total Pre-GTM capex (USD)",
                    "Post-GTM yearly capex (USD)",
                ]
            ]
            capex_fmt = {
                col: "{:.0f}"
                for col in capex_display.columns
                if col not in ["ID_vaccine", "Vaccine name"]
            }
            st.dataframe(capex_display.style.format(capex_fmt))

        with st.expander("Vaccines royalty revenues", expanded=True):
            royalty_df = _render_product_assumption_table(
                session_key="vaccine_royalty_table",
                default_factory=_default_royalty_table,
                blank_row_factory=_blank_vaccine_royalty_row,
                column_config={
                    "Monetization model": st.column_config.SelectboxColumn(
                        "Monetization model", options=["Product Sale", "Licensing"]
                    )
                },
            )
            revenue_lookup = st.session_state.get("vaccine_revenue_table", pd.DataFrame())
            patent_lookup = revenue_lookup.set_index("ID_vaccine").get(
                "Patent revenue target (USD)", pd.Series(dtype=float)
            )
            post_lookup = revenue_lookup.set_index("ID_vaccine").get(
                "Post patent revenue target (USD)", pd.Series(dtype=float)
            )
            royalty_rate = _coerce_numeric(royalty_df.get("Royalty rate (%)", pd.Series(dtype=float))).div(100)
            royalty_df["Patent revenue (USD)"] = royalty_df["ID_vaccine"].map(patent_lookup)
            royalty_df["Post patent revenue (USD)"] = royalty_df["ID_vaccine"].map(post_lookup)
            royalty_df["Royalty income (USD)"] = royalty_df["Patent revenue (USD)"] * royalty_rate
            st.session_state["vaccine_royalty_table"] = royalty_df
            st.dataframe(
                royalty_df[[
                    "ID_vaccine",
                    "Vaccine name",
                    "Royalty rate (%)",
                    "Royalty income (USD)",
                    "Patent revenue (USD)",
                    "Post patent revenue (USD)",
                ]].style.format({
                    "Royalty rate (%)": "{:.1f}",
                    "Royalty income (USD)": "{:.0f}",
                    "Patent revenue (USD)": "{:.0f}",
                    "Post patent revenue (USD)": "{:.0f}",
                })
            )

        with st.expander("Vaccines market share", expanded=True):
            market_share_df = _render_product_assumption_table(
                session_key="vaccine_market_share_table",
                default_factory=_default_market_share_table,
                blank_row_factory=_blank_vaccine_market_share_row,
            )
            relevant_market = _coerce_numeric(
                market_share_df.get("Relevant market size (USD)", pd.Series(dtype=float))
            )
            patent_target_pct = _coerce_numeric(
                market_share_df.get("Revenue target - patent %", pd.Series(dtype=float))
            ).div(100)
            post_target_pct = _coerce_numeric(
                market_share_df.get("Revenue target - post %", pd.Series(dtype=float))
            ).div(100)
            market_share_df["Revenue target patent (USD)"] = relevant_market * patent_target_pct
            market_share_df["Revenue target post (USD)"] = relevant_market * post_target_pct
            st.session_state["vaccine_market_share_table"] = market_share_df
            st.dataframe(
                market_share_df[[
                    "ID_vaccine",
                    "Vaccine name",
                    "Relevant market type",
                    "Relevant market size (USD)",
                    "Revenue target patent (USD)",
                    "Revenue target post (USD)",
                    "Market share patent %",
                    "Market share post %",
                    "Market growth %",
                    "Sales growth %",
                ]].style.format({
                    "Relevant market size (USD)": "{:.0f}",
                    "Revenue target patent (USD)": "{:.0f}",
                    "Revenue target post (USD)": "{:.0f}",
                    "Market share patent %": "{:.1f}",
                    "Market share post %": "{:.1f}",
                    "Market growth %": "{:.1f}",
                    "Sales growth %": "{:.1f}",
                })
            )

        product_df = _render_product_assumption_table(
            session_key="product_table",
            default_factory=_default_products,
            blank_row_factory=lambda df: _blank_product_row(f"Product {len(df) + 1}"),
            column_config={
                "stage": st.column_config.SelectboxColumn("Stage", options=STAGE_OPTIONS),
                "include_in_consolidation": st.column_config.CheckboxColumn("Include", default=True),
                "success_prob": st.column_config.NumberColumn(
                    "Success probability", min_value=0.0, max_value=1.0, step=0.05
                ),
            },
            id_column=None,
            name_column="name",
        )
        product_df = _validate_product_df(product_df)
        st.session_state["product_table"] = product_df

        portfolio = _build_portfolio(product_df, model_cfg)
        if portfolio is None:
            st.info("Add at least one product with a name to run valuations.")
        else:
            valuation_result = ValuationEngine(portfolio).run()
            st.session_state["model_config"] = model_cfg
            st.session_state["portfolio"] = portfolio
            st.session_state["valuation_result"] = valuation_result
            st.success(
                f"Run complete: portfolio rNPV = {valuation_result.rnpv:,.0f} {model_cfg.currency}."
            )

    with financial_tab:
        st.subheader("Financial statements")
        if valuation_result is None or model_cfg is None:
            st.info("Run the model configuration tab to populate the statements.")
        else:
            cons = valuation_result.consolidated
            with st.expander("Consolidated forecast", expanded=True):
                cons_display = cons[["revenue", "ebitda", "fcff_after_wc"]].copy()
                cons_display.columns = ["Revenue", "EBITDA", "FCFF after WC"]
                st.dataframe(
                    cons_display.style.format(
                        {
                            "Revenue": "{:.0f}",
                            "EBITDA": "{:.0f}",
                            "FCFF after WC": "{:.0f}",
                        }
                    )
                )
                st.line_chart(cons_display)
            perf_df, position_df, cash_flow_df = _compute_financial_statements(cons, model_cfg)
            st.markdown("**Statement of Financial Performance**")
            st.dataframe(
                perf_df.style.format({col: "{:.0f}" for col in perf_df.columns})
            )
            st.markdown("**Statement of Financial Position**")
            st.dataframe(
                position_df.style.format({col: "{:.0f}" for col in position_df.columns})
            )
            st.markdown("**Statement of Cash Flows**")
            st.dataframe(
                cash_flow_df.style.format({col: "{:.0f}" for col in cash_flow_df.columns})
            )
            st.markdown("**Excel Model Download**")
            excel_bytes = st.session_state.get("financial_excel_bytes")
            download_container = st.container()
            with download_container:
                if not excel_bytes:
                    if st.button("Prepare Excel Model", key="prepare_financial_excel"):
                        with st.spinner("Preparing Excel workbook..."):
                            excel_bytes = _build_financial_excel(
                                cons,
                                perf_df,
                                position_df,
                                cash_flow_df,
                            )
                        st.session_state["financial_excel_bytes"] = excel_bytes
                if excel_bytes:
                    st.download_button(
                        "Download Excel Model",
                        data=excel_bytes,
                        file_name="Financial_Report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_financial_excel",
                    )
                    if st.button("Clear Prepared Excel", key="clear_financial_excel"):
                        st.session_state.pop("financial_excel_bytes", None)
                        excel_bytes = None
                if not excel_bytes:
                    st.info("Click 'Prepare Excel Model' to generate the workbook for download.")

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
            base_rnpv = valuation_result.rnpv
            ratios = _build_ratio_table(cons)
            st.markdown("**Margin & intensity analysis**")
            st.dataframe(ratios.style.format("{:.1%}"))

            with st.expander("Sensitivity & stress testing", expanded=True):
                sens_cols = st.columns(3)
                pricing_delta = sens_cols[0].slider(
                    "Pricing pressure swing",
                    0.0,
                    0.5,
                    0.15,
                    help="Revenue-linked driver",
                )
                manufacturing_delta = sens_cols[1].slider("Manufacturing cost swing", 0.0, 0.5, 0.2)
                clinical_delta = sens_cols[2].slider("Clinical success swing", 0.0, 0.5, 0.1)
                drivers = {
                    "Pricing pressure": (pricing_delta, "revenue"),
                    "Manufacturing costs": (manufacturing_delta, "cost"),
                    "Clinical success": (clinical_delta, "productivity"),
                }
                sens_df = _run_sensitivity_matrix(portfolio, drivers)
                if sens_df.empty:
                    st.info("Not enough data to compute sensitivities.")
                else:
                    st.dataframe(sens_df.style.format({"rNPV": "{:.0f}", "Delta vs base": "{:+,.0f}"}))

                st.markdown("**Scenario stress testing**")
                severe_cases = [
                    ("Regulatory delay", 0.7, 1.2, 0.03, 0.9),
                    ("Trial failure", 0.6, 1.3, 0.04, 0.75),
                    ("Pricing squeeze", 0.5, 1.05, 0.02, 0.95),
                ]
                stress_rows = []
                for name, rev_mult, cost_mult, dr_shift, prob_mult in severe_cases:
                    result = _evaluate_portfolio_shock(
                        portfolio,
                        revenue_multiplier=rev_mult,
                        cost_multiplier=cost_mult,
                        discount_shift=dr_shift,
                        success_prob_multiplier=prob_mult,
                    )
                    if result is None:
                        continue
                    stress_rows.append(
                        {
                            "Scenario": name,
                            "rNPV": result.rnpv,
                            "EBITDA impact": result.consolidated["ebitda"].sum(),
                        }
                    )
                if stress_rows:
                    stress_df = pd.DataFrame(stress_rows)
                    numeric_cols = stress_df.select_dtypes(include="number").columns
                    formatter = {col: "{:+,.0f}" if "impact" in col.lower() else "{:,}" for col in numeric_cols}
                    st.dataframe(stress_df.style.format(formatter))

            with st.expander("Trend, seasonality & segmentation", expanded=False):
                decomp_df = _compute_decomposition(cons)
                if decomp_df is not None:
                    st.line_chart(decomp_df)
                else:
                    st.info("Need more history to decompose trend/seasonality.")

                seg_df = _build_segmentation_table(valuation_result)
                if not seg_df.empty:
                    st.dataframe(
                        seg_df.style.format({
                            "Revenue share": "{:.1%}",
                            "EBITDA margin": "{:.1%}",
                            "FCFF (PV proxy)": "{:.0f}",
                        })
                    )
                    st.bar_chart(seg_df.set_index("Product")["Revenue share"])
                else:
                    st.info("Add probability-weighted products to see segmentation insights.")

            with st.expander("Monte Carlo & probabilistic valuation", expanded=False):
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
                    hist = np.histogram(sims, bins=20)
                    st.bar_chart(pd.DataFrame({"rNPV": hist[0]}, index=hist[1][:-1]))
                    var = MonteCarloEngine.value_at_risk(sims)
                    cvar = MonteCarloEngine.conditional_value_at_risk(sims)
                    st.write(
                        f"Mean rNPV: {sims.mean():,.0f} | Std: {sims.std():,.0f} | VaR95: {var:,.0f} | CVaR95: {cvar:,.0f}"
                    )
                    st.write(
                        "Probabilistic valuation percentiles:",
                        sims.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
                    )

                else:
                    st.info("Run the simulation to unlock probabilistic metrics.")

            with st.expander("What-if analysis & goal seek", expanded=False):
                what_cols = st.columns(3)
                what_rev = what_cols[0].slider("Revenue multiplier", 0.4, 2.0, 1.0)
                what_cost = what_cols[1].slider("Cost multiplier", 0.5, 2.5, 1.0)
                what_dr = what_cols[2].slider("Discount shift", -0.05, 0.1, 0.0)
                if st.button("Evaluate what-if case"):
                    result = _evaluate_portfolio_shock(
                        portfolio,
                        revenue_multiplier=float(what_rev),
                        cost_multiplier=float(what_cost),
                        discount_shift=float(what_dr),
                    )
                    if result is not None:
                        st.success(f"What-if rNPV: {result.rnpv:,.0f}")

                target_rnpv = st.number_input("Target rNPV for goal seek", value=base_rnpv)
                if st.button("Solve revenue multiplier"):
                    multiplier, achieved = _goal_seek_revenue_multiplier(portfolio, float(target_rnpv))
                    if achieved is not None:
                        st.write(
                            f"Revenue multiplier {multiplier:.2f} approximates the goal (achieved rNPV {achieved:,.0f})."
                        )
                    else:
                        st.warning("Goal seek failed—try adjusting the target or assumptions.")

            with st.expander("Tornado & spider diagnostics", expanded=False):
                tornado_df = _tornado_dataframe(portfolio, base_rnpv)
                if tornado_df.empty:
                    st.info("Unable to compute tornado deltas.")
                else:
                    st.dataframe(tornado_df.style.format({"rNPV": "{:.0f}", "Delta": "{:+,.0f}"}))
                    if go is not None:
                        tornado_fig = go.Figure()
                        pos = tornado_df[tornado_df["Delta"] >= 0]
                        neg = tornado_df[tornado_df["Delta"] < 0]
                        tornado_fig.add_trace(
                            go.Bar(
                                y=pos["Driver"],
                                x=pos["Delta"],
                                orientation="h",
                                name="Positive",
                            )
                        )
                        tornado_fig.add_trace(
                            go.Bar(
                                y=neg["Driver"],
                                x=neg["Delta"],
                                orientation="h",
                                name="Negative",
                            )
                        )
                        tornado_fig.update_layout(barmode="relative", title="Tornado impact")
                        st.plotly_chart(tornado_fig, use_container_width=True)

                        spider_fig = go.Figure()
                        pivot = tornado_df.pivot(index="Driver", columns="Change", values="rNPV").fillna(base_rnpv)
                        spider_fig.add_trace(
                            go.Scatterpolar(r=pivot.get("+20%", [base_rnpv]), theta=pivot.index, name="Upside")
                        )
                        spider_fig.add_trace(
                            go.Scatterpolar(r=pivot.get("-20%", [base_rnpv]), theta=pivot.index, name="Downside")
                        )
                        st.plotly_chart(spider_fig, use_container_width=True)

            with st.expander("Regression & classification models", expanded=False):
                reg_df = _run_linear_regressions(cons)
                if reg_df is not None:
                    st.table(reg_df.style.format({"Intercept": "{:.0f}", "Revenue beta": "{:.2f}", "R^2": "{:.2f}"}))
                else:
                    st.info("Install scikit-learn to unlock regression diagnostics.")

                seg_df = _build_segmentation_table(valuation_result)
                class_df = _run_classification_model(seg_df)
                if class_df is not None:
                    st.dataframe(class_df.style.format({
                        "Revenue share": "{:.1%}",
                        "EBITDA margin": "{:.1%}",
                        "High-margin probability": "{:.1%}",
                    }))
                else:
                    st.caption("Classification output requires scikit-learn and at least one product.")

            with st.expander("Time-series & ML forecasting", expanded=False):
                ts_metric = st.selectbox("Series to forecast", ["revenue", "ebitda"], key="forecast_metric")
                method = st.selectbox("Forecast model", ["ARIMA", "Prophet", "LSTM"], key="forecast_method")
                horizon_max = max(5, int(model_cfg.n_years))
                horizon_default = min(10, horizon_max)
                horizon = st.slider("Forecast steps", 5, horizon_max, horizon_default)
                if st.button("Run time-series model"):
                    fe = ForecastEngine(model_cfg)
                    period_index = pd.period_range(str(model_cfg.first_year), periods=len(cons), freq="Y")
                    series = pd.Series(cons[ts_metric].values, index=period_index)
                    series.index = series.index.to_timestamp()
                    try:
                        if method == "ARIMA":
                            forecast = fe.forecast_arima(series, steps=horizon)
                            st.line_chart(forecast)
                        elif method == "Prophet":
                            hist_df = pd.DataFrame({"ds": series.index, "y": series.values})
                            forecast = fe.forecast_prophet(hist_df, periods=horizon)
                            st.line_chart(forecast.set_index("ds")["yhat"])
                        else:
                            forecast = fe.forecast_lstm(series, steps_ahead=horizon)
                            st.line_chart(pd.Series(forecast))
                    except Exception as exc:
                        st.warning(f"Forecast failed: {exc}")

            with st.expander("Optimisation, portfolio design & real options", expanded=False):
                opt_df = _optimize_operations(cons)
                if opt_df is not None:
                    st.table(opt_df.style.format({"Value": "{:.2f}"}))
                else:
                    st.caption("Install SciPy to enable nonlinear optimisation.")

                mv_df = _mean_variance_portfolio(valuation_result)
                if mv_df is not None:
                    st.dataframe(
                        mv_df.style.format({"Mean": "{:.0f}", "Std": "{:.0f}", "Suggested weight": "{:.1%}"})
                    )

                option_val = _real_options_value(valuation_result)
                if option_val is not None:
                    st.write(f"Real option (deferral) value estimate: {option_val:,.0f}")
                else:
                    st.caption("Provide R&D cash flows and install SciPy to compute real options.")

            with st.expander("Risk, copulas, macro & ESG linkages", expanded=False):
                copula_df = _copula_simulation(cons)
                if copula_df is not None:
                    st.scatter_chart(copula_df)

                macro_cols = st.columns(4)
                inflation = macro_cols[0].slider("Inflation", 0.0, 0.15, 0.03)
                gdp = macro_cols[1].slider("GDP growth", -0.05, 0.1, 0.02)
                fx = macro_cols[2].slider("FX depreciation", -0.1, 0.2, 0.0)
                sentiment = macro_cols[3].slider("Market sentiment", -0.3, 0.3, 0.0)
                macro_revenue = cons["revenue"] * (1 + inflation + gdp + sentiment - fx)
                st.line_chart(pd.DataFrame({"Original": cons["revenue"], "Macro-adjusted": macro_revenue}))

                esg_cols = st.columns(3)
                carbon_price = esg_cols[0].slider("Carbon price ($/t)", 0, 200, 75)
                emissions = esg_cols[1].slider("Emissions (kt)", 0, 500, 120)
                renewable_share = esg_cols[2].slider("Renewable share", 0.0, 1.0, 0.35)
                esg_cost = carbon_price * emissions * (1 - renewable_share)
                st.write(f"ESG-adjusted annual carbon cost: {esg_cost:,.0f}")

                intel_score = st.slider("Market intelligence sentiment", -1.0, 1.0, 0.1)
                st.write(
                    f"Sentiment-adjusted revenue uplift: {(intel_score * 5):+.1f}% applied to TAM during scenario planning."
                )

            with st.expander("Comparative & ML-based valuation", expanded=False):
                cluster_df = _cluster_products(valuation_result)
                if cluster_df is not None:
                    st.dataframe(cluster_df)
                else:
                    st.caption("Need scikit-learn and multiple products for clustering.")

                ml_mult_df = _machine_learning_multiple(cons)
                if ml_mult_df is not None:
                    st.line_chart(ml_mult_df.set_index("Year"))
                else:
                    st.caption("Install scikit-learn to run ML-driven multiple predictions.")

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

    with rag_tab:
        _render_rag_assistant_page()

    st.caption(
        "Tip: Upload a Prophet-ready dataframe (ds, y) and plug it into ForecastScenarioBridge for richer scenarios."
    )


if __name__ == "__main__":
    main()
