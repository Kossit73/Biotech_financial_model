import unittest

import numpy as np

from valuation_codex_package.core import (
    ModelConfig,
    ProductConfig,
    Product,
    Portfolio,
    ValuationEngine,
    Scenario,
    ScenarioEngine,
    MonteCarloEngine,
)


class CoreModelTests(unittest.TestCase):
    def test_revenue_series_preexisting_market_respects_ramp(self) -> None:
        model_cfg = ModelConfig(first_year=2024, n_years=5, sales_ramp_factors=[0.5, 1.0, 1.0, 1.0, 1.0])
        product_cfg = ProductConfig(
            name="TestProduct",
            stage="Market",
            success_prob=1.0,
            include_in_consolidation=True,
            preexisting_market=True,
            time_to_market=3,
            patent_years=5,
            patent_revenue_target=100.0,
            post_patent_revenue_target=0.0,
        )
        product = Product(product_cfg, model_cfg)

        revenue = product.build_revenue_series()
        expected = np.array([50.0, 100.0, 100.0, 100.0, 100.0])
        np.testing.assert_allclose(revenue.values, expected)

    def test_valuation_engine_zero_cashflows(self) -> None:
        model_cfg = ModelConfig(first_year=2024, n_years=3)
        product_cfg = ProductConfig(
            name="ZeroProduct",
            stage="Discovery",
            success_prob=1.0,
            include_in_consolidation=True,
            preexisting_market=False,
            time_to_market=3,
            patent_years=5,
            patent_revenue_target=0.0,
            post_patent_revenue_target=0.0,
        )
        product = Product(product_cfg, model_cfg)
        portfolio = Portfolio([product], model_cfg)

        result = ValuationEngine(portfolio).run()
        self.assertEqual(result.rnpv, 0.0)
        self.assertTrue((result.consolidated["fcff_after_wc"] == 0.0).all())

    def test_scenario_engine_applies_discount_shift(self) -> None:
        model_cfg = ModelConfig(first_year=2024, n_years=3)
        product_cfg = ProductConfig(
            name="ScenarioProduct",
            stage="Market",
            success_prob=1.0,
            include_in_consolidation=True,
            preexisting_market=True,
            time_to_market=0,
            patent_years=5,
            patent_revenue_target=100.0,
            post_patent_revenue_target=0.0,
        )
        portfolio = Portfolio([Product(product_cfg, model_cfg)], model_cfg)

        scenario = Scenario(name="RateUp", discount_rate_shift=0.02)
        results = ScenarioEngine(portfolio).run_scenarios([scenario])
        rate_row = results.loc[results["scenario"] == "RateUp", "discount_rate"].iloc[0]
        self.assertAlmostEqual(rate_row, model_cfg.discount_rate + 0.02)

    def test_monte_carlo_reproducible_seed(self) -> None:
        model_cfg = ModelConfig(first_year=2024, n_years=3)
        product_cfg = ProductConfig(
            name="MCProduct",
            stage="Market",
            success_prob=1.0,
            include_in_consolidation=True,
            preexisting_market=True,
            time_to_market=0,
            patent_years=5,
            patent_revenue_target=100.0,
            post_patent_revenue_target=0.0,
        )
        portfolio = Portfolio([Product(product_cfg, model_cfg)], model_cfg)
        engine = MonteCarloEngine(portfolio)

        sims_a = engine.simulate(n_sims=10, random_seed=42)
        sims_b = engine.simulate(n_sims=10, random_seed=42)
        np.testing.assert_allclose(sims_a.values, sims_b.values)


if __name__ == "__main__":
    unittest.main()
