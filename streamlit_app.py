"""Streamlit UI for the Valuation Codex biotech financial model."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from typing import Callable, Dict, List, Optional, Tuple
import io
import zipfile

import numpy as np
import pandas as pd
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
    validate_portfolio,
)


STAGE_OPTIONS = [
    "Discovery",
    "Preclinical",
    "Phase I",
    "Phase II",
    "Phase III",
    "Approval",
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


def _stage_template(name: str, stage: str, success_prob: float, time_to_market: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": name,
                "stage": stage,
                "success_prob": success_prob,
                "include_in_consolidation": True,
                "time_to_market": time_to_market,
                "patent_years": 15,
                "patent_revenue_target": 120_000_000,
                "post_patent_revenue_target": 60_000_000,
                "market_growth_patent": 0.03,
                "market_growth_post": 0.0,
                "cogs_patent": 0.32,
                "cogs_post": 0.5,
                "sales_marketing_pct": 0.18,
                "gna_pct": 0.12,
                "rd_remaining_pre_launch": 150_000_000,
                "rd_annual_post_launch": 10_000_000,
                "capex_remaining_pre_launch": 50_000_000,
                "capex_annual_post_launch": 6_000_000,
            }
        ]
    )


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


def _default_phase_probabilities_table() -> pd.DataFrame:
    data = [
        {
            "name": "AgSeed-101",
            "stage": "Phase II",
            "Discovery→Preclinical": 0.7,
            "Preclinical→Phase I": 0.65,
            "Phase I→Phase II": 0.55,
            "Phase II→Phase III": 0.4,
            "Phase III→Approval": 0.6,
        },
        {
            "name": "BioYield-Plus",
            "stage": "Phase III",
            "Discovery→Preclinical": 0.75,
            "Preclinical→Phase I": 0.7,
            "Phase I→Phase II": 0.6,
            "Phase II→Phase III": 0.45,
            "Phase III→Approval": 0.65,
        },
    ]
    return pd.DataFrame(data)


def _default_erosion_table() -> pd.DataFrame:
    data = [
        {
            "name": "AgSeed-101",
            "Year 1": 1.0,
            "Year 2": 0.8,
            "Year 3": 0.6,
            "Year 4": 0.4,
            "Year 5": 0.3,
        },
        {
            "name": "BioYield-Plus",
            "Year 1": 1.0,
            "Year 2": 0.85,
            "Year 3": 0.7,
            "Year 4": 0.55,
            "Year 5": 0.4,
        },
    ]
    return pd.DataFrame(data)


def _default_milestones_table() -> pd.DataFrame:
    data = [
        {"name": "AgSeed-101", "Label": "Phase III start", "Year offset": 2, "Amount": 15_000_000, "Probability": 0.4},
        {"name": "AgSeed-101", "Label": "Approval", "Year offset": 5, "Amount": 35_000_000, "Probability": 0.6},
        {"name": "BioYield-Plus", "Label": "Approval", "Year offset": 3, "Amount": 40_000_000, "Probability": 0.65},
    ]
    return pd.DataFrame(data)


def _default_comps_table() -> pd.DataFrame:
    data = [
        {"Company": "PeerCo A", "EV/EBITDA": 9.5, "EV/Sales": 4.2, "Notes": "Mid-cap biologics"},
        {"Company": "PeerCo B", "EV/EBITDA": 11.2, "EV/Sales": 5.0, "Notes": "Late-stage vaccines"},
        {"Company": "PeerCo C", "EV/EBITDA": 8.0, "EV/Sales": 3.6, "Notes": "Specialty pharma"},
    ]
    return pd.DataFrame(data)


def _vaccine_sales_year_columns(first_year: int, years: int) -> List[int]:
    return [first_year + i for i in range(years)]


def _default_vaccine_sales_table(first_year: int = 2024, years: int = 5) -> pd.DataFrame:
    years = _vaccine_sales_year_columns(first_year, years)
    rows = []
    for name, base_doses, base_price in [
        ("AgSeed-101", 5, 25),
        ("BioYield-Plus", 7, 30),
    ]:
        row = {
            "ID_vaccine": f"VAC-{len(rows) + 1:03d}",
            "Vaccine name": name,
        }
        for idx, year in enumerate(years, start=1):
            row[f"{year} Doses (M)"] = base_doses + idx
            row[f"{year} Price per dose"] = base_price + idx
        rows.append(row)
    return pd.DataFrame(rows)


def _blank_vaccine_sales_row(df: pd.DataFrame, first_year: int, years: int) -> Dict:
    next_id = _next_vaccine_id(df)
    years = _vaccine_sales_year_columns(first_year, years)
    row = {"ID_vaccine": next_id, "Vaccine name": "New vaccine"}
    for year in years:
        row[f"{year} Doses (M)"] = 5.0
        row[f"{year} Price per dose"] = 25.0
    return row


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


def _build_assumption_audit(model_cfg: ModelConfig, product_df: pd.DataFrame) -> pd.DataFrame:
    model_defaults = ModelConfig()
    audit_rows = []
    for field_name, value in asdict(model_cfg).items():
        default_value = getattr(model_defaults, field_name)
        if value != default_value:
            audit_rows.append(
                {
                    "Scope": "Model",
                    "Item": field_name,
                    "Current": value,
                    "Default": default_value,
                }
            )
    product_defaults = ProductConfig(
        name="Default",
        stage="Discovery",
        success_prob=0.2,
    )
    for _, row in product_df.iterrows():
        name = row.get("name", "Unnamed")
        for field_name in [f.name for f in fields(ProductConfig) if f.name in row]:
            current_value = row.get(field_name)
            default_value = getattr(product_defaults, field_name, None)
            if pd.isna(current_value):
                continue
            if current_value != default_value:
                audit_rows.append(
                    {
                        "Scope": f"Product: {name}",
                        "Item": field_name,
                        "Current": current_value,
                        "Default": default_value,
                    }
                )
    return pd.DataFrame(audit_rows)


def _build_dashboard_summary(cons: pd.DataFrame) -> Dict[str, float | int]:
    peak_sales = float(cons["revenue"].max())
    peak_year = int(cons["revenue"].idxmax())
    ebitda_margin = float(cons["ebitda"].sum() / cons["revenue"].sum()) if cons["revenue"].sum() != 0 else 0.0
    cumulative_fcff = cons["fcff_after_wc"].cumsum()
    break_even_year = int(cumulative_fcff[cumulative_fcff > 0].index.min()) if (cumulative_fcff > 0).any() else -1
    return {
        "peak_sales": peak_sales,
        "peak_year": peak_year,
        "ebitda_margin": ebitda_margin,
        "break_even_year": break_even_year,
    }


def _build_sensitivity_table(portfolio: Portfolio, base_rnpv: float) -> pd.DataFrame:
    scenarios = [
        ("Discount rate +1%", {"discount_rate_shift": 0.01}),
        ("Discount rate -1%", {"discount_rate_shift": -0.01}),
        ("Revenue +10%", {"revenue_multiplier": 1.1}),
        ("Revenue -10%", {"revenue_multiplier": 0.9}),
        ("COGS +5%", {"cost_multiplier": 1.05}),
        ("COGS -5%", {"cost_multiplier": 0.95}),
        ("Success prob +10%", {"success_prob_multiplier": 1.1}),
        ("Success prob -10%", {"success_prob_multiplier": 0.9}),
        ("Delay launch +1y", {"time_to_market_shift": 1}),
        ("Accelerate launch -1y", {"time_to_market_shift": -1}),
    ]
    rows = []
    scen_engine = ScenarioEngine(portfolio)
    for label, kwargs in scenarios:
        scenario = Scenario(name=label, **kwargs)
        port = scen_engine._apply_scenario(scenario)
        val = ValuationEngine(port).run(validate=False)
        rows.append(
            {
                "Driver": label,
                "rNPV": val.rnpv,
                "Delta vs base": val.rnpv - base_rnpv,
            }
        )
    return pd.DataFrame(rows).sort_values("Delta vs base")


def _assemble_investor_packet(
    model_cfg: ModelConfig,
    valuation_result: ValuationResult,
    audit_df: pd.DataFrame,
    scenario_df: pd.DataFrame | None,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", json.dumps(asdict(model_cfg), indent=2))
        zf.writestr("rnpv.json", json.dumps({"rnpv": valuation_result.rnpv}, indent=2))
        zf.writestr("consolidated.csv", valuation_result.consolidated.to_csv())
        if scenario_df is not None:
            zf.writestr("scenarios.csv", scenario_df.to_csv(index=False))
        if not audit_df.empty:
            zf.writestr("assumption_audit.csv", audit_df.to_csv(index=False))
        for name, df in valuation_result.per_product.items():
            zf.writestr(f"product_{name}.csv", df.to_csv())
    buffer.seek(0)
    return buffer.read()


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

    for col in ["time_to_market", "patent_years"]:
        if col in validated.columns:
            validated[col] = validated[col].fillna(0).clip(lower=0)

    return validated


def _build_stage_probability_map(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stage_map: Dict[str, Dict[str, float]] = {}
    if df.empty:
        return stage_map
    for _, row in df.iterrows():
        name = row.get("name")
        if not name:
            continue
        stage_map[name] = {
            "Discovery": float(row.get("Discovery→Preclinical", 0.0)),
            "Preclinical": float(row.get("Preclinical→Phase I", 0.0)),
            "Phase I": float(row.get("Phase I→Phase II", 0.0)),
            "Phase II": float(row.get("Phase II→Phase III", 0.0)),
            "Phase III": float(row.get("Phase III→Approval", 0.0)),
        }
    return stage_map


def _build_erosion_map(df: pd.DataFrame) -> Dict[str, List[float]]:
    erosion_map: Dict[str, List[float]] = {}
    if df.empty:
        return erosion_map
    for _, row in df.iterrows():
        name = row.get("name")
        if not name:
            continue
        factors = [float(row.get(f"Year {i}", 1.0)) for i in range(1, 6)]
        erosion_map[name] = factors
    return erosion_map


def _build_milestone_map(df: pd.DataFrame) -> Dict[str, List[Dict[str, float | int | str]]]:
    milestone_map: Dict[str, List[Dict[str, float | int | str]]] = {}
    if df.empty:
        return milestone_map
    for _, row in df.iterrows():
        name = row.get("name")
        if not name:
            continue
        milestone = {
            "label": row.get("Label", "Milestone"),
            "year_offset": int(row.get("Year offset", 0)),
            "amount": float(row.get("Amount", 0.0)),
            "probability": None if pd.isna(row.get("Probability")) else float(row.get("Probability")),
        }
        milestone_map.setdefault(name, []).append(milestone)
    return milestone_map


def _sanitize_product_records(
    df: pd.DataFrame,
    stage_map: Dict[str, Dict[str, float]] | None = None,
    erosion_map: Dict[str, List[float]] | None = None,
    milestone_map: Dict[str, List[Dict[str, float | int | str]]] | None = None,
) -> List[Dict]:
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
        name = cleaned.get("name")
        if stage_map and name in stage_map:
            cleaned["stage_probabilities"] = stage_map[name]
        if erosion_map and name in erosion_map:
            cleaned["post_patent_erosion"] = erosion_map[name]
        if milestone_map and name in milestone_map:
            cleaned["milestone_cashflows"] = milestone_map[name]
        records.append(cleaned)
    return records


def _build_portfolio(
    product_df: pd.DataFrame,
    model_cfg: ModelConfig,
    stage_df: pd.DataFrame | None = None,
    erosion_df: pd.DataFrame | None = None,
    milestone_df: pd.DataFrame | None = None,
) -> Portfolio | None:
    stage_map = _build_stage_probability_map(stage_df or pd.DataFrame())
    erosion_map = _build_erosion_map(erosion_df or pd.DataFrame())
    milestone_map = _build_milestone_map(milestone_df or pd.DataFrame())
    product_records = _sanitize_product_records(product_df, stage_map, erosion_map, milestone_map)
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
    multiples = (cons["ebitda"].rolling(3).mean().fillna(method="bfill") + 1) / 1_000_000
    model = LinearRegression()
    model.fit(features.values, multiples.values)
    pred = model.predict(features.values)
    return pd.DataFrame({"Year": cons.index, "Predicted multiple": pred})


def _render_rag_assistant_page() -> None:
    st.subheader("RAG Assistant")
    st.write(
        "Turn your valuation workbook into an evidence-backed investment memo. "
        "The RAG Assistant gathers model outputs, ingests external research, and drafts "
        "a report that highlights risks, catalysts, and valuation proof points."
    )

    with st.container(border=True):
        hero_cols = st.columns([2, 1])
        with hero_cols[0]:
            st.markdown("### What you can do")
            st.markdown(
                "- **Capture a model snapshot** with key KPIs and scenario outputs.\n"
                "- **Ingest evidence packs** (clinical readouts, market research, diligence).\n"
                "- **Generate a feasibility report** with citations and risk callouts."
            )
            st.markdown("**Best for:** investor memos, internal IC reviews, and diligence briefs.")
        with hero_cols[1]:
            st.markdown("### Readiness checklist")
            st.metric("Model snapshot", "Ready")
            st.metric("Evidence library", "Awaiting upload")
            st.metric("Report draft", "Not generated")

    st.markdown("### Launch plan")
    step_cols = st.columns(3)
    step_cols[0].markdown("**1. Collect**\n\nSend the financial snapshot to the `/collect` endpoint.")
    step_cols[1].markdown("**2. Ingest**\n\nUpload supporting documents to `/ingest`.")
    step_cols[2].markdown("**3. Generate**\n\nTrigger `/generate` to build `report.md`.")
    st.info(
        "Once the report is generated, the assistant can summarize risks, highlight catalysts, "
        "and trace each claim back to a supporting document."
    )

    with st.expander("View sample snapshot payload", expanded=False):
        snapshot_payload = {
            "project_id": "example-project",
            "financial_snapshot": {
                "currency": "USD",
                "npv": 54000000,
                "irr": 0.19,
                "dscr_min": 1.35,
                "payback_years": 5.6,
                "capex_total": 120000000,
                "opex_annual": 8500000,
                "revenue_annual": 23500000,
                "scenarios": [
                    {"name": "Base", "npv": 54000000, "irr": 0.19},
                    {"name": "Downside", "npv": 30000000, "irr": 0.14},
                    {"name": "Upside", "npv": 78000000, "irr": 0.24},
                ],
            },
            "cell_map": {
                "npv": "Assumptions!B12",
                "irr": "Assumptions!B13",
                "dscr_min": "Debt!F22",
            },
            "workbook_hash": "sha256-hash-here",
        }
        st.code(snapshot_payload, language="json")
        st.download_button(
            "Download sample JSON",
            data=json.dumps(snapshot_payload, indent=2),
            file_name="rag_snapshot_sample.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("### Service endpoints")
    st.code(
        "\n".join(
            [
                "POST http://<rag-host>/collect",
                "POST http://<rag-host>/ingest",
                "POST http://<rag-host>/generate",
            ]
        ),
        language="bash",
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

    page_choice = st.sidebar.radio(
        "Navigate",
        ["Model workspace", "RAG Assistant"],
        index=0,
    )

    if page_choice == "RAG Assistant":
        _render_rag_assistant_page()
        st.caption(
            "Tip: Upload a Prophet-ready dataframe (ds, y) and plug it into ForecastScenarioBridge for richer scenarios."
        )
        return

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
        st.subheader("Model assumptions")

        with st.expander("Start here: guided setup", expanded=False):
            st.markdown(
                "1) **Define the portfolio** with stages, timelines, and probability profiles.\n"
                "2) **Calibrate revenue and cost assumptions** (market size, pricing, COGS).\n"
                "3) **Review scenario and sensitivity outputs** for investor-facing insights.\n"
                "4) **Export the investor packet** for sharing."
            )
            st.info(
                "Tip: start with a template from the library below, then refine probabilities, milestones, "
                "and erosion curves to match your asset."
            )

        with st.expander("Template library", expanded=False):
            template_choice = st.selectbox(
                "Choose a template",
                [
                    "Select template",
                    "Discovery",
                    "Preclinical",
                    "Phase I",
                    "Phase II",
                    "Phase III",
                    "Approval",
                    "Commercial",
                ],
            )
            if st.button("Load template") and template_choice != "Select template":
                if template_choice == "Discovery":
                    template_df = _stage_template("Discovery asset", "Discovery", 0.1, 8)
                elif template_choice == "Preclinical":
                    template_df = _stage_template("Preclinical asset", "Preclinical", 0.2, 6)
                elif template_choice == "Phase I":
                    template_df = _stage_template("Phase I asset", "Phase I", 0.3, 5)
                elif template_choice == "Phase II":
                    template_df = _stage_template("Phase II asset", "Phase II", 0.45, 4)
                elif template_choice == "Phase III":
                    template_df = _stage_template("Phase III asset", "Phase III", 0.6, 3)
                elif template_choice == "Approval":
                    template_df = _stage_template("Approval-stage asset", "Approval", 0.8, 1)
                else:
                    template_df = _stage_template("Commercial asset", "Commercial", 0.95, 0)
                st.session_state["product_table"] = template_df
                st.success("Template loaded. Review assumptions below.")

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
            sales_years = _vaccine_sales_year_columns(int(first_year), int(n_years))
            display_years = sales_years[:1]
            sales_column_config = {
                "ID_vaccine": st.column_config.TextColumn("ID_vaccine"),
                "Vaccine name": st.column_config.TextColumn("Vaccine name"),
            }
            for year in display_years:
                sales_column_config[f"{year} Doses (M)"] = st.column_config.NumberColumn(
                    f"{year} Doses (M)", min_value=0.0, step=0.5
                )
                sales_column_config[f"{year} Price per dose"] = st.column_config.NumberColumn(
                    f"{year} Price per dose", min_value=0.0, step=1.0
                )
            vaccine_df = _render_product_assumption_table(
                session_key="vaccine_sales_table",
                default_factory=lambda: _default_vaccine_sales_table(int(first_year), int(n_years)),
                blank_row_factory=lambda df: _blank_vaccine_sales_row(df, int(first_year), int(n_years)),
                id_column="ID_vaccine",
                name_column="Vaccine name",
                column_config=sales_column_config,
            )

            with st.expander("Yearly Increment Helper", expanded=False):
                select_key = "vaccine_sales_table_row_select"
                selected_id = st.session_state.get(select_key)
                if selected_id is None and not vaccine_df.empty:
                    selected_id = vaccine_df.loc[vaccine_df.index[0], "ID_vaccine"]
                if selected_id in vaccine_df["ID_vaccine"].values:
                    selected_idx = vaccine_df.index[vaccine_df["ID_vaccine"] == selected_id][0]
                else:
                    selected_idx = vaccine_df.index[0] if not vaccine_df.empty else None

                base_year = st.selectbox(
                    "Base year",
                    options=sales_years,
                    index=0,
                    key="vaccine_sales_base_year",
                )
                doses_increment = st.number_input(
                    "Doses increment per year (M)",
                    value=1.0,
                    step=0.5,
                    key="vaccine_sales_dose_increment",
                )
                price_increment = st.number_input(
                    "Price increment per year",
                    value=1.0,
                    step=1.0,
                    key="vaccine_sales_price_increment",
                )

                if st.button("Apply increments", key="vaccine_sales_apply"):
                    if selected_idx is None:
                        st.warning("Select a vaccine row to apply increments.")
                    else:
                        start_idx = sales_years.index(base_year)
                        base_doses = pd.to_numeric(
                            vaccine_df.at[selected_idx, f"{base_year} Doses (M)"],
                            errors="coerce",
                        )
                        base_price = pd.to_numeric(
                            vaccine_df.at[selected_idx, f"{base_year} Price per dose"],
                            errors="coerce",
                        )
                        base_doses = 0.0 if pd.isna(base_doses) else float(base_doses)
                        base_price = 0.0 if pd.isna(base_price) else float(base_price)
                        for offset, year in enumerate(sales_years[start_idx:], start=0):
                            vaccine_df.at[selected_idx, f"{year} Doses (M)"] = (
                                base_doses + doses_increment * offset
                            )
                            vaccine_df.at[selected_idx, f"{year} Price per dose"] = (
                                base_price + price_increment * offset
                            )
                        st.session_state["vaccine_sales_table"] = vaccine_df
                        st.success("Increments applied to doses and price.")

            st.session_state["vaccine_sales_table"] = vaccine_df

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

        with st.expander("Phase-by-phase success probabilities", expanded=True):
            stage_df = _render_product_assumption_table(
                session_key="phase_probability_table",
                default_factory=_default_phase_probabilities_table,
                blank_row_factory=lambda df: {
                    "name": f"Asset {len(df) + 1}",
                    "stage": "Phase I",
                    "Discovery→Preclinical": 0.7,
                    "Preclinical→Phase I": 0.6,
                    "Phase I→Phase II": 0.5,
                    "Phase II→Phase III": 0.4,
                    "Phase III→Approval": 0.6,
                },
                column_config={
                    "stage": st.column_config.SelectboxColumn("Stage", options=STAGE_OPTIONS),
                    "Discovery→Preclinical": st.column_config.NumberColumn("Discovery→Preclinical", min_value=0.0, max_value=1.0, step=0.05),
                    "Preclinical→Phase I": st.column_config.NumberColumn("Preclinical→Phase I", min_value=0.0, max_value=1.0, step=0.05),
                    "Phase I→Phase II": st.column_config.NumberColumn("Phase I→Phase II", min_value=0.0, max_value=1.0, step=0.05),
                    "Phase II→Phase III": st.column_config.NumberColumn("Phase II→Phase III", min_value=0.0, max_value=1.0, step=0.05),
                    "Phase III→Approval": st.column_config.NumberColumn("Phase III→Approval", min_value=0.0, max_value=1.0, step=0.05),
                },
            )
            st.session_state["phase_probability_table"] = stage_df
            st.caption("Probabilities compound from the current stage through approval.")

        with st.expander("Post-patent erosion curves", expanded=True):
            erosion_df = _render_product_assumption_table(
                session_key="erosion_table",
                default_factory=_default_erosion_table,
                blank_row_factory=lambda df: {
                    "name": f"Asset {len(df) + 1}",
                    "Year 1": 1.0,
                    "Year 2": 0.8,
                    "Year 3": 0.6,
                    "Year 4": 0.4,
                    "Year 5": 0.3,
                },
                column_config={
                    "Year 1": st.column_config.NumberColumn("Year 1", min_value=0.0, max_value=1.0, step=0.05),
                    "Year 2": st.column_config.NumberColumn("Year 2", min_value=0.0, max_value=1.0, step=0.05),
                    "Year 3": st.column_config.NumberColumn("Year 3", min_value=0.0, max_value=1.0, step=0.05),
                    "Year 4": st.column_config.NumberColumn("Year 4", min_value=0.0, max_value=1.0, step=0.05),
                    "Year 5": st.column_config.NumberColumn("Year 5", min_value=0.0, max_value=1.0, step=0.05),
                },
            )
            st.session_state["erosion_table"] = erosion_df
            st.caption("Erosion factors apply post-patent to reflect competitive entry.")

        with st.expander("Milestone-based cash flows", expanded=True):
            milestone_df = _render_product_assumption_table(
                session_key="milestone_table",
                default_factory=_default_milestones_table,
                blank_row_factory=lambda df: {
                    "name": f"Asset {len(df) + 1}",
                    "Label": "Milestone",
                    "Year offset": 1,
                    "Amount": 10_000_000,
                    "Probability": 0.5,
                },
                column_config={
                    "Year offset": st.column_config.NumberColumn("Year offset", min_value=0, max_value=40, step=1),
                    "Amount": st.column_config.NumberColumn("Amount", min_value=0.0, step=1_000_000.0),
                    "Probability": st.column_config.NumberColumn("Probability", min_value=0.0, max_value=1.0, step=0.05),
                },
            )
            st.session_state["milestone_table"] = milestone_df

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

        stage_df = st.session_state.get("phase_probability_table", pd.DataFrame())
        erosion_df = st.session_state.get("erosion_table", pd.DataFrame())
        milestone_df = st.session_state.get("milestone_table", pd.DataFrame())

        portfolio = _build_portfolio(product_df, model_cfg, stage_df, erosion_df, milestone_df)
        if portfolio is None:
            st.info("Add at least one product with a name to run valuations.")
        else:
            issues = validate_portfolio(portfolio)
            if issues:
                st.warning("Validation checks flagged issues:\n- " + "\n- ".join(issues))
            valuation_result = ValuationEngine(portfolio).run(validate=False)
            st.success(
                f"Run complete: portfolio rNPV = {valuation_result.rnpv:,.0f} {model_cfg.currency}."
            )
            with st.expander("Assumption audit trail", expanded=False):
                audit_df = _build_assumption_audit(model_cfg, product_df)
                if audit_df.empty:
                    st.info("No deviations from defaults detected.")
                else:
                    st.dataframe(audit_df)

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

    with dashboard_tab:
        st.subheader("Dashboard")
        if valuation_result is None or model_cfg is None:
            st.info("Configure and run the model to see dashboard metrics.")
        else:
            cons = valuation_result.consolidated
            summary = _build_dashboard_summary(cons)
            kpi_cols = st.columns(4)
            kpi_cols[0].metric("Portfolio rNPV", f"{valuation_result.rnpv:,.0f} {model_cfg.currency}")
            kpi_cols[1].metric("Peak revenue", f"{summary['peak_sales']:,.0f} ({summary['peak_year']})")
            kpi_cols[2].metric("Avg EBITDA margin", f"{summary['ebitda_margin']:.1%}")
            break_even_label = (
                str(summary["break_even_year"]) if summary["break_even_year"] > 0 else "Not reached"
            )
            kpi_cols[3].metric("Break-even year", break_even_label)

            chart_data = cons[["revenue", "ebitda", "fcff_after_wc"]]
            st.area_chart(chart_data)
            st.bar_chart(cons["fcff_after_wc"], use_container_width=True)

            with st.expander("Comparable multiples", expanded=True):
                comps_df = _render_product_assumption_table(
                    session_key="comps_table",
                    default_factory=_default_comps_table,
                    blank_row_factory=lambda df: {
                        "Company": f"Peer {len(df) + 1}",
                        "EV/EBITDA": 9.0,
                        "EV/Sales": 4.0,
                        "Notes": "",
                    },
                    id_column=None,
                    name_column="Company",
                )
                ev_ebitda = pd.to_numeric(comps_df["EV/EBITDA"], errors="coerce").dropna()
                ev_sales = pd.to_numeric(comps_df["EV/Sales"], errors="coerce").dropna()
                last_year = cons.index[-1]
                last_ebitda = cons.loc[last_year, "ebitda"]
                last_sales = cons.loc[last_year, "revenue"]
                if not ev_ebitda.empty:
                    implied_ev_ebitda = last_ebitda * ev_ebitda.median()
                    st.metric("Median EV/EBITDA multiple", f"{ev_ebitda.median():.1f}x")
                    st.write(f"Implied EV (EBITDA): {implied_ev_ebitda:,.0f} {model_cfg.currency}")
                if not ev_sales.empty:
                    implied_ev_sales = last_sales * ev_sales.median()
                    st.metric("Median EV/Sales multiple", f"{ev_sales.median():.1f}x")
                    st.write(f"Implied EV (Sales): {implied_ev_sales:,.0f} {model_cfg.currency}")

            with st.expander("Investor-ready export", expanded=False):
                audit_df = _build_assumption_audit(model_cfg, st.session_state.get("product_table", pd.DataFrame()))
                packet = _assemble_investor_packet(model_cfg, valuation_result, audit_df, None)
                st.download_button(
                    "Download investor packet (ZIP)",
                    data=packet,
                    file_name="investor_packet.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

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

            with st.expander("Key driver sensitivity (tornado)", expanded=True):
                sensitivity_df = _build_sensitivity_table(portfolio, base_rnpv)
                st.dataframe(sensitivity_df.style.format({"rNPV": "{:.0f}", "Delta vs base": "{:+,.0f}"}))
                st.bar_chart(
                    sensitivity_df.set_index("Driver")["Delta vs base"],
                    use_container_width=True,
                )

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
                mc_cols = st.columns(5)
                n_sims = mc_cols[0].number_input("Simulations", min_value=100, max_value=5000, value=1000, step=100)
                rev_sigma = mc_cols[1].number_input("Revenue sigma", min_value=0.01, max_value=0.5, value=0.15, step=0.01)
                cost_sigma = mc_cols[2].number_input("Cost sigma", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
                corr = mc_cols[3].number_input("Rev/Cost corr", min_value=-0.9, max_value=0.9, value=0.3, step=0.05)
                seed = mc_cols[4].number_input("Random seed", min_value=0, value=42)

                if st.button("Run Monte Carlo simulation"):
                    sims = MonteCarloEngine(portfolio).simulate(
                        n_sims=int(n_sims),
                        revenue_sigma=float(rev_sigma),
                        cost_sigma=float(cost_sigma),
                        corr_rev_cost=float(corr),
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
                horizon = st.slider("Forecast steps", 5, 25, 10)
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
            col1, col2, col3, col4, col5 = st.columns(5)
            rev_mult = col1.slider("Revenue multiplier", 0.25, 2.5, 1.0)
            cost_mult = col2.slider("Cost multiplier", 0.5, 2.0, 1.0)
            dr_shift = col3.slider("Discount rate shift", -0.05, 0.1, 0.0)
            prob_mult = col4.slider("Success prob multiplier", 0.5, 1.5, 1.0)
            ttm_shift = col5.slider("Launch delay (years)", -2, 5, 0)

            override_map: Dict[str, Dict[str, float | int]] = {}
            with st.expander("Product-specific overrides", expanded=False):
                override_df = _render_product_assumption_table(
                    session_key="scenario_override_table",
                    default_factory=lambda: pd.DataFrame(
                        {
                            "name": [prod.config.name for prod in portfolio.products],
                            "revenue_multiplier": 1.0,
                            "cost_multiplier": 1.0,
                            "success_prob_multiplier": 1.0,
                            "time_to_market_shift": 0,
                            "rd_cost_multiplier": 1.0,
                            "capex_cost_multiplier": 1.0,
                        }
                    ),
                    blank_row_factory=lambda df: {
                        "name": f"Asset {len(df) + 1}",
                        "revenue_multiplier": 1.0,
                        "cost_multiplier": 1.0,
                        "success_prob_multiplier": 1.0,
                        "time_to_market_shift": 0,
                        "rd_cost_multiplier": 1.0,
                        "capex_cost_multiplier": 1.0,
                    },
                    id_column=None,
                    name_column="name",
                )
                override_map = {
                    row["name"]: {
                        "revenue_multiplier": float(row.get("revenue_multiplier", 1.0)),
                        "cost_multiplier": float(row.get("cost_multiplier", 1.0)),
                        "success_prob_multiplier": float(row.get("success_prob_multiplier", 1.0)),
                        "time_to_market_shift": int(row.get("time_to_market_shift", 0)),
                        "rd_cost_multiplier": float(row.get("rd_cost_multiplier", 1.0)),
                        "capex_cost_multiplier": float(row.get("capex_cost_multiplier", 1.0)),
                    }
                    for row in override_df.to_dict("records")
                    if row.get("name")
                }

            scenario = Scenario(
                name="Custom scenario",
                revenue_multiplier=float(rev_mult),
                cost_multiplier=float(cost_mult),
                discount_rate_shift=float(dr_shift),
                success_prob_multiplier=float(prob_mult),
                time_to_market_shift=int(ttm_shift),
                product_overrides=override_map,
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
