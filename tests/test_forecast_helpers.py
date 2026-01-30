import unittest

import pandas as pd

from pharma_biotech_valuation import (
    _target_revenue,
    _compute_revenues,
    _apply_straight_line_allocation,
    _apply_prelaunch_allocation,
)


class ForecastHelperTests(unittest.TestCase):
    def test_target_revenue_handles_missing_inputs(self) -> None:
        self.assertEqual(_target_revenue(None, 100.0, 0.5), 0.0)
        self.assertEqual(_target_revenue(10, None, 0.5), 0.0)

    def test_target_revenue_calculates_value(self) -> None:
        self.assertEqual(_target_revenue(10, 200.0, 0.5), 1000.0)

    def test_compute_revenues_applies_ramp_and_growth(self) -> None:
        years = pd.Index(range(2024, 2030), name="year")
        revenues = _compute_revenues(
            years,
            market_entry_year=2024,
            end_patent_year=2026,
            target_patent=100.0,
            target_post=50.0,
            sales_growth_pct=0.1,
            sales_ramp=(0.2, 0.4, 0.6, 0.8, 1.0),
        )

        self.assertEqual(revenues[0], 20.0)
        self.assertEqual(revenues[2], 60.0)
        self.assertEqual(revenues[3], 40.0)
        self.assertAlmostEqual(revenues[5], 55.0)

    def test_apply_straight_line_allocation(self) -> None:
        df = pd.DataFrame(index=pd.Index([2024, 2025, 2026], name="year"))
        df["amortization"] = 0.0

        _apply_straight_line_allocation(
            df,
            column="amortization",
            total=300.0,
            start_year=2024,
            years_count=3,
        )

        self.assertEqual(df.loc[2024, "amortization"], 100.0)
        self.assertEqual(df.loc[2026, "amortization"], 100.0)

    def test_apply_prelaunch_allocation(self) -> None:
        df = pd.DataFrame(index=pd.Index([2022, 2023, 2024, 2025], name="year"))
        df["capex"] = 0.0

        _apply_prelaunch_allocation(
            df,
            column="capex",
            total=200.0,
            market_entry_year=2024,
            allocation_years=2,
        )

        self.assertEqual(df.loc[2022, "capex"], 100.0)
        self.assertEqual(df.loc[2023, "capex"], 100.0)
        self.assertEqual(df.loc[2024, "capex"], 0.0)


if __name__ == "__main__":
    unittest.main()
