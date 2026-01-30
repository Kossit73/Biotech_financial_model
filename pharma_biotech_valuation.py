"""Pharma‑Biotech – Risk‑Adjusted DCF Valuation (PDF‑linked)."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import hashlib
import json
import re

import numpy as np
import pandas as pd

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None


# ----------------------------
# Page linking / syncing
# ----------------------------


@dataclass(frozen=True)
class PageLink:
    """Metadata linking a function/logic block to a PDF page."""

    page_no: int
    title: str
    must_contain: Tuple[str, ...] = ()

    def deep_link(self, pdf_path: Path) -> str:
        return f"{pdf_path.as_posix()}#page={self.page_no}"


def linked_to_page(page_no: int, *, title: str, must_contain: Sequence[str] = ()) -> Callable:
    """Attach PDF page linkage metadata to a function."""

    def _decorator(fn: Callable) -> Callable:
        setattr(fn, "_page_link", PageLink(page_no=page_no, title=title, must_contain=tuple(must_contain)))
        return fn

    return _decorator


# ----------------------------
# PDF access
# ----------------------------


class PdfDocument:
    def __init__(self, pdf_path: Path):
        if pdfplumber is None:
            raise ImportError("pdfplumber is required. Install via: pip install pdfplumber")
        self.path = Path(pdf_path)
        self._pdf = pdfplumber.open(str(self.path))

    @property
    def n_pages(self) -> int:
        return len(self._pdf.pages)

    def page_text(self, page_no: int) -> str:
        """1-indexed page access."""
        if page_no < 1 or page_no > self.n_pages:
            raise ValueError(f"page_no out of range: {page_no} (1..{self.n_pages})")
        txt = self._pdf.pages[page_no - 1].extract_text() or ""
        return re.sub(r"[ \t]+", " ", txt)

    def page_hash(self, page_no: int) -> str:
        txt = self.page_text(page_no)
        return hashlib.sha256(txt.encode("utf-8")).hexdigest()


@dataclass
class PageIndexEntry:
    page_no: int
    sha256: str
    title: str
    first_lines: List[str]


def _guess_page_title(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    skip = ("Strictly private", "Pharma-Biotech", "Confidential", "efinancialmodels.com", "Powered by")
    for ln in lines:
        if not any(ln.startswith(s) for s in skip):
            return ln[:120]
    return lines[0][:120]


def build_page_index(pdf: PdfDocument, *, max_first_lines: int = 6) -> List[PageIndexEntry]:
    idx: List[PageIndexEntry] = []
    for page_no in range(1, pdf.n_pages + 1):
        text = pdf.page_text(page_no)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        idx.append(
            PageIndexEntry(
                page_no=page_no,
                sha256=pdf.page_hash(page_no),
                title=_guess_page_title(text),
                first_lines=lines[:max_first_lines],
            )
        )
    return idx


def assert_synced(pdf: PdfDocument, linked_fns: Iterable[Callable]) -> None:
    for fn in linked_fns:
        page_link: Optional[PageLink] = getattr(fn, "_page_link", None)
        if not page_link:
            continue
        text = pdf.page_text(page_link.page_no)
        missing = [tok for tok in page_link.must_contain if tok not in text]
        if missing:
            raise ValueError(
                f"Sync check failed for {fn.__name__} (page {page_link.page_no} '{page_link.title}'): "
                f"missing tokens: {missing}"
            )


# ----------------------------
# Inputs with page provenance
# ----------------------------


@dataclass
class Sourced:
    """A value + where it came from."""

    value: Any
    page_no: int
    raw_match: str = ""

    def to_jsonable(self) -> Dict[str, Any]:
        return {"value": self.value, "page_no": self.page_no, "raw_match": self.raw_match}


@dataclass
class GlobalAssumptions:
    wacc: float = 0.10
    terminal_ev_ebitda_multiple: float = 8.0

    income_tax_rate: float = 0.25
    cost_inflation: float = 0.01

    ar_days: int = 30
    inv_days: int = 30
    ap_days: int = 30

    rd_amort_years: int = 10
    fixed_asset_depr_years: int = 8

    sales_ramp: Tuple[float, float, float, float, float] = (0.20, 0.60, 1.00, 1.00, 1.00)

    vc_holding_years: int = 7
    vc_discount_rate: float = 0.20


@dataclass
class ProductInputs:
    product_id: int
    name: str

    stage: Optional[str] = None
    consolidation: bool = True
    success_probability: float = 1.0

    first_forecast_year: Optional[int] = None
    time_to_market_years: Optional[int] = None
    market_entry_year: Optional[int] = None
    patent_duration_years: Optional[int] = None
    end_patent_year: Optional[int] = None

    customers_per_year_patent: Optional[int] = None
    price_per_customer_patent: Optional[float] = None

    customers_per_year_post: Optional[int] = None
    price_per_customer_post: Optional[float] = None

    adj_patent_pct: float = 0.80
    adj_post_pct: float = 0.70

    cogs_patent_pct: float = 0.30
    cogs_post_pct: float = 0.50
    marketing_sales_pct: float = 0.15
    ga_pct_of_sales: float = 0.01
    ga_fixed_usd: float = 500_000.0
    royalty_pct_of_revenue: float = 0.0

    sales_growth_pct: float = 0.005

    launch_cost_usd: float = 0.0

    remaining_rnd_usd: float = 0.0
    rnd_allocation_years: int = 5
    remaining_capex_usd: float = 0.0
    capex_allocation_years: int = 5

    sources: Dict[str, int] = field(default_factory=dict)


# ----------------------------
# PDF extraction (page-linked)
# ----------------------------


def _to_float_pct(s: str) -> float:
    return float(s.replace("%", "").strip()) / 100.0


def _to_int(s: str) -> int:
    return int(s.replace(",", "").strip())


def _to_float(s: str) -> float:
    return float(s.replace(",", "").strip())


@linked_to_page(
    5,
    title="Product Assumptions – Vaccine(s) Timeline",
    must_contain=("Vaccine(s) Timeline", "Market Entry", "End Patent"),
)
def extract_product_timeline(pdf: PdfDocument) -> Dict[int, ProductInputs]:
    text = pdf.page_text(5)
    pattern = re.compile(
        r"(?P<id>\d+)\s+Vaccine\s+\d+\s+(?P<name>[A-Za-z ]+?)\s+"
        r"(?P<stage>Generic Drug|Market|Approval Stage|Phase III|Phase II|Phase I|Pre-Study)\s+"
        r"(?P<prob>\d+\.\d+)%\s+(?P<conso>ON|OFF)\s+"
        r"(?P<first>\d{4})\s+(?P<ttm>-?\d+)\s+(?P<entry>\d{4})\s+(?P<patent>\d+)\s+(?P<end>\d{4})"
    )

    products: Dict[int, ProductInputs] = {}
    for match in pattern.finditer(text):
        pid = int(match.group("id"))
        products[pid] = ProductInputs(
            product_id=pid,
            name=match.group("name").strip(),
            stage=match.group("stage").strip(),
            success_probability=_to_float_pct(match.group("prob")),
            consolidation=(match.group("conso") == "ON"),
            first_forecast_year=int(match.group("first")),
            time_to_market_years=int(match.group("ttm")),
            market_entry_year=int(match.group("entry")),
            patent_duration_years=int(match.group("patent")),
            end_patent_year=int(match.group("end")),
            sources={
                "timeline": 5,
                "stage": 5,
                "success_probability": 5,
                "consolidation": 5,
                "first_forecast_year": 5,
                "time_to_market_years": 5,
                "market_entry_year": 5,
                "patent_duration_years": 5,
                "end_patent_year": 5,
            },
        )

    if not products:
        raise ValueError("Could not parse timeline table on page 5. Check PDF text extraction.")

    return products


@linked_to_page(
    7,
    title="Product Assumptions – Revenue Estimations",
    must_contain=("Revenue Estimations", "Patent Period", "Post Patent"),
)
def extract_revenue_estimations(pdf: PdfDocument, products: Dict[int, ProductInputs]) -> None:
    text = pdf.page_text(7)
    row_re = re.compile(
        r"(?P<cust>[\d,]+)\s+(?P<price>\d+)\s+(?P<rev>[\d,]+)\s+"
        r"(?P<adj_pat>\d+\.\d+)%\s+(?P<adj_post>\d+\.\d+)%\s+"
        r"(?P<cust_post>[\d,]+)\s+(?P<price_post>\d+)\s+(?P<rev_post>[\d,]+)"
    )

    rows = list(row_re.finditer(text))
    if len(rows) < len(products):
        raise ValueError(
            f"Revenue estimation rows found={len(rows)} but products={len(products)}. "
            "Text extraction may have failed; inspect the PDF page 7 text."
        )

    for pid, match in zip(sorted(products.keys()), rows[: len(products)]):
        product = products[pid]
        product.customers_per_year_patent = _to_int(match.group("cust"))
        product.price_per_customer_patent = _to_float(match.group("price"))
        product.customers_per_year_post = _to_int(match.group("cust_post"))
        product.price_per_customer_post = _to_float(match.group("price_post"))
        product.adj_patent_pct = _to_float_pct(match.group("adj_pat"))
        product.adj_post_pct = _to_float_pct(match.group("adj_post"))
        product.sources.update(
            {
                "customers_per_year_patent": 7,
                "price_per_customer_patent": 7,
                "customers_per_year_post": 7,
                "price_per_customer_post": 7,
                "adj_patent_pct": 7,
                "adj_post_pct": 7,
            }
        )


@linked_to_page(
    18,
    title="Summary – VC Method parameters",
    must_contain=("Holding Period Years", "Discount Rate % 20.0%", "Terminal Value EV/EBITDA Multiple"),
)
def extract_vc_method_params(pdf: PdfDocument, assumptions: GlobalAssumptions) -> None:
    text = pdf.page_text(18)
    m_hold = re.search(r"Holding Period Years\s+(?P<y>\d+)", text)
    m_disc = re.search(r"Discount Rate %\s+(?P<p>\d+\.\d+)%", text)
    m_mult = re.search(r"Terminal Value EV/EBITDA Multiple x\s+(?P<m>\d+\.\d+)x", text)

    if m_hold:
        assumptions.vc_holding_years = int(m_hold.group("y"))
    if m_disc:
        assumptions.vc_discount_rate = _to_float_pct(m_disc.group("p"))
    if m_mult:
        assumptions.terminal_ev_ebitda_multiple = float(m_mult.group("m"))


@linked_to_page(
    15,
    title="Consolidated FCFF build – components",
    must_contain=("Free Cash Flows to Firm (FCFF)", "EBIT", "Change in NWC", "R&D Investments", "CAPEX"),
)
def fcff_from_operating_lines(
    *,
    ebit: pd.Series,
    tax_rate: float,
    amortization: pd.Series,
    depreciation: pd.Series,
    change_in_nwc: pd.Series,
    rnd_investments: pd.Series,
    capex: pd.Series,
    cash_reserve: Optional[pd.Series] = None,
    terminal_value: Optional[pd.Series] = None,
) -> pd.Series:
    adj_tax = -np.maximum(ebit, 0.0) * tax_rate

    fcff = (
        ebit
        + adj_tax
        + amortization
        + depreciation
        - change_in_nwc
        - rnd_investments
        - capex
    )

    if cash_reserve is not None:
        fcff = fcff - cash_reserve
    if terminal_value is not None:
        fcff = fcff + terminal_value

    return fcff


@linked_to_page(
    2,
    title="Risk-adjusted DCF concept (probability-weighted product cash flows)",
    must_contain=("probability-weighted", "risk-adjusted", "Net Present Value"),
)
def discount_cashflows(cashflows: pd.Series, *, discount_rate: float) -> float:
    t = np.arange(1, len(cashflows) + 1, dtype=float)
    df = 1.0 / np.power(1.0 + discount_rate, t)
    return float(np.sum(cashflows.values * df))


# ----------------------------
# Forecast engine (simplified)
# ----------------------------


def _years_index(start_year: int, n_years: int) -> pd.Index:
    return pd.Index(range(start_year, start_year + n_years), name="year")


def _apply_sales_ramp(target: float, year_since_launch: int, ramp: Tuple[float, float, float, float, float]) -> float:
    if year_since_launch <= 0:
        return 0.0
    if 1 <= year_since_launch <= 5:
        return target * ramp[year_since_launch - 1]
    return target


def _target_revenue(customers: Optional[int], price: Optional[float], adj_pct: float) -> float:
    if customers is None or price is None:
        return 0.0
    return float(customers) * float(price) * float(adj_pct)


def _compute_revenues(
    years: pd.Index,
    *,
    market_entry_year: int,
    end_patent_year: int,
    target_patent: float,
    target_post: float,
    sales_growth_pct: float,
    sales_ramp: Tuple[float, float, float, float, float],
) -> List[float]:
    year_values = years.to_numpy(dtype=int)
    years_since_launch = year_values - int(market_entry_year) + 1
    in_patent = year_values <= int(end_patent_year)
    targets = np.where(in_patent, float(target_patent), float(target_post))

    ramp_factors = np.ones_like(years_since_launch, dtype=float)
    ramp_map = np.array(sales_ramp, dtype=float)
    ramp_mask = (years_since_launch >= 1) & (years_since_launch <= 5)
    ramp_factors[ramp_mask] = ramp_map[years_since_launch[ramp_mask] - 1]
    ramp_factors[years_since_launch <= 0] = 0.0

    revenues = targets * ramp_factors
    growth_mask = years_since_launch > 5
    if np.any(growth_mask):
        growth_factors = (1.0 + sales_growth_pct) ** (years_since_launch[growth_mask] - 5)
        revenues[growth_mask] *= growth_factors

    return revenues.tolist()


def _apply_straight_line_allocation(
    df: pd.DataFrame,
    *,
    column: str,
    total: float,
    start_year: int,
    years_count: int,
) -> None:
    if total <= 0 or years_count <= 0:
        return
    annual = total / max(years_count, 1)
    for offset in range(years_count):
        year = start_year + offset
        if year in df.index:
            df.loc[year, column] = annual


def _apply_prelaunch_allocation(
    df: pd.DataFrame,
    *,
    column: str,
    total: float,
    market_entry_year: int,
    allocation_years: int,
) -> None:
    if total <= 0 or allocation_years <= 0:
        return
    prelaunch_years = [year for year in df.index if year < market_entry_year][:allocation_years]
    if not prelaunch_years:
        return
    annual = total / len(prelaunch_years)
    df.loc[prelaunch_years, column] = annual


def forecast_product_fcff(
    p: ProductInputs,
    g: GlobalAssumptions,
    *,
    horizon_years: int = 25,
) -> pd.DataFrame:
    if p.first_forecast_year is None:
        raise ValueError(f"Missing first_forecast_year for product {p.product_id} {p.name}")
    if p.market_entry_year is None:
        raise ValueError(f"Missing market_entry_year for product {p.product_id} {p.name}")

    years = _years_index(p.first_forecast_year, horizon_years)
    df = pd.DataFrame(index=years)
    df["product_id"] = p.product_id
    df["name"] = p.name

    target_pat = _target_revenue(
        p.customers_per_year_patent,
        p.price_per_customer_patent,
        p.adj_patent_pct,
    )
    target_post = _target_revenue(
        p.customers_per_year_post,
        p.price_per_customer_post,
        p.adj_post_pct,
    )

    end_patent = p.end_patent_year if p.end_patent_year is not None else (p.market_entry_year + 20)

    df["revenue"] = _compute_revenues(
        years,
        market_entry_year=p.market_entry_year,
        end_patent_year=end_patent,
        target_patent=target_pat,
        target_post=target_post,
        sales_growth_pct=p.sales_growth_pct,
        sales_ramp=g.sales_ramp,
    )

    df["cogs"] = np.where(
        df.index.values <= end_patent,
        df["revenue"] * p.cogs_patent_pct,
        df["revenue"] * p.cogs_post_pct,
    )

    df["marketing_sales"] = df["revenue"] * p.marketing_sales_pct
    df["ga_variable"] = df["revenue"] * p.ga_pct_of_sales
    df["ga_fixed"] = p.ga_fixed_usd

    df["royalties"] = df["revenue"] * p.royalty_pct_of_revenue

    df["launch_cost"] = 0.0
    if p.launch_cost_usd and p.market_entry_year in df.index:
        df.loc[p.market_entry_year, "launch_cost"] = p.launch_cost_usd

    df["ebitda"] = df["revenue"] - (
        df["cogs"]
        + df["marketing_sales"]
        + df["ga_variable"]
        + df["ga_fixed"]
        + df["royalties"]
        + df["launch_cost"]
    )

    df["amortization"] = 0.0
    if p.market_entry_year in df.index:
        _apply_straight_line_allocation(
            df,
            column="amortization",
            total=p.remaining_rnd_usd,
            start_year=p.market_entry_year,
            years_count=g.rd_amort_years,
        )

    df["depreciation"] = 0.0
    if p.market_entry_year in df.index:
        _apply_straight_line_allocation(
            df,
            column="depreciation",
            total=p.remaining_capex_usd,
            start_year=p.market_entry_year,
            years_count=g.fixed_asset_depr_years,
        )

    df["ebit"] = df["ebitda"] - df["amortization"] - df["depreciation"]

    ar = df["revenue"] * (g.ar_days / 365.0)
    inv = df["cogs"] * (g.inv_days / 365.0)
    ap = df["cogs"] * (g.ap_days / 365.0)
    nwc = ar + inv - ap
    df["change_in_nwc"] = nwc.diff().fillna(0.0)

    df["rnd_investments"] = 0.0
    _apply_prelaunch_allocation(
        df,
        column="rnd_investments",
        total=p.remaining_rnd_usd,
        market_entry_year=p.market_entry_year,
        allocation_years=p.rnd_allocation_years,
    )

    df["capex"] = 0.0
    _apply_prelaunch_allocation(
        df,
        column="capex",
        total=p.remaining_capex_usd,
        market_entry_year=p.market_entry_year,
        allocation_years=p.capex_allocation_years,
    )

    df["fcff"] = fcff_from_operating_lines(
        ebit=df["ebit"],
        tax_rate=g.income_tax_rate,
        amortization=df["amortization"],
        depreciation=df["depreciation"],
        change_in_nwc=df["change_in_nwc"],
        rnd_investments=df["rnd_investments"],
        capex=df["capex"],
        cash_reserve=None,
        terminal_value=None,
    )

    prob = float(p.success_probability)
    conso = 1.0 if p.consolidation else 0.0
    df["fcff_prob_weighted"] = df["fcff"] * prob * conso

    return df


def compute_terminal_value_from_last_year(
    ebitda_last: float,
    net_debt_last: float,
    *,
    multiple: float,
) -> Tuple[float, float]:
    terminal_ev = float(ebitda_last) * float(multiple)
    terminal_equity = terminal_ev - float(net_debt_last)
    return terminal_ev, terminal_equity


def consolidate_portfolio(product_dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not product_dfs:
        raise ValueError("No product forecasts provided.")
    years = product_dfs[0].index
    out = pd.DataFrame(index=years)

    cols = [
        "revenue",
        "ebitda",
        "ebit",
        "amortization",
        "depreciation",
        "change_in_nwc",
        "rnd_investments",
        "capex",
        "fcff",
        "fcff_prob_weighted",
    ]
    for col in cols:
        out[col] = sum((df[col].reindex(years).fillna(0.0) for df in product_dfs))

    return out


def risk_adjusted_npv(consolidated: pd.DataFrame, assumptions: GlobalAssumptions) -> Dict[str, float]:
    pv = discount_cashflows(consolidated["fcff_prob_weighted"], discount_rate=assumptions.wacc)
    return {"rnpv": pv, "enterprise_value": pv}


# ----------------------------
# High-level orchestration
# ----------------------------


def extract_inputs(pdf: PdfDocument) -> Tuple[GlobalAssumptions, Dict[int, ProductInputs]]:
    assumptions = GlobalAssumptions()
    products = extract_product_timeline(pdf)
    extract_revenue_estimations(pdf, products)
    extract_vc_method_params(pdf, assumptions)
    return assumptions, products


def run_model(
    pdf_path: Path,
    *,
    horizon_years: int = 25,
) -> Dict[str, Any]:
    pdf = PdfDocument(pdf_path)

    linked_fns = [
        extract_product_timeline,
        extract_revenue_estimations,
        extract_vc_method_params,
        fcff_from_operating_lines,
        discount_cashflows,
    ]
    assert_synced(pdf, linked_fns)

    assumptions, products = extract_inputs(pdf)

    product_dfs = [forecast_product_fcff(prod, assumptions, horizon_years=horizon_years) for prod in products.values()]
    consolidated = consolidate_portfolio(product_dfs)
    valuation = risk_adjusted_npv(consolidated, assumptions)

    vc_discount_factor = 1.0 / ((1.0 + assumptions.vc_discount_rate) ** assumptions.vc_holding_years)
    vc_discounted_equity = valuation["enterprise_value"] * vc_discount_factor
    vc = {
        "holding_years": assumptions.vc_holding_years,
        "discount_rate": assumptions.vc_discount_rate,
        "discount_factor": vc_discount_factor,
        "discounted_equity_value_proxy": vc_discounted_equity,
    }

    return {
        "global_assumptions": asdict(assumptions),
        "products": {pid: asdict(prod) for pid, prod in products.items()},
        "valuation": valuation,
        "vc_method_proxy": vc,
    }


# ----------------------------
# CLI
# ----------------------------


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Pharma‑Biotech Risk‑Adjusted DCF Valuation (PDF‑linked)")
    ap.add_argument("pdf", type=str, help="Path to the PDF export of the model")
    ap.add_argument("--out", type=str, default="./out", help="Output directory")
    ap.add_argument("--horizon", type=int, default=25, help="Forecast horizon (years)")
    args = ap.parse_args(argv)

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)

    pdf = PdfDocument(pdf_path)
    idx = build_page_index(pdf)
    _write_json(out_dir / "page_index.json", [asdict(entry) for entry in idx])

    res = run_model(pdf_path, horizon_years=args.horizon)

    _write_json(
        out_dir / "extracted_inputs.json",
        {"global_assumptions": res["global_assumptions"], "products": res["products"]},
    )
    _write_json(out_dir / "valuation_summary.json", {"valuation": res["valuation"], "vc_method_proxy": res["vc_method_proxy"]})

    print("Wrote:")
    print(" -", out_dir / "page_index.json")
    print(" -", out_dir / "extracted_inputs.json")
    print(" -", out_dir / "valuation_summary.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
