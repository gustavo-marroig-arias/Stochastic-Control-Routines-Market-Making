import unittest

import numpy as np

from stoch_control_trading import (
    LiquidationParams,
    gamma,
    inventory_path,
    optimal_rate,
    simulate_liquidation,
)


class LiquidationTests(unittest.TestCase):
    def test_gamma_terminal_condition(self):
        params = LiquidationParams(T=1.0, q0=10.0, kappa=2.0, terminal_penalty=8.0)
        self.assertAlmostEqual(float(gamma(params.T, params)), -params.terminal_penalty)

    def test_inventory_path_is_monotone_and_has_closed_form_terminal_value(self):
        params = LiquidationParams(T=2.0, q0=50.0, kappa=1.5, terminal_penalty=6.0)
        times = np.linspace(0.0, params.T, 101)
        q = inventory_path(times, params)
        expected_terminal = params.q0 * (params.kappa / params.terminal_penalty) / (
            params.kappa / params.terminal_penalty + params.T
        )

        self.assertAlmostEqual(q[0], params.q0)
        self.assertAlmostEqual(q[-1], expected_terminal)
        self.assertTrue(np.all(np.diff(q) <= 1e-12))

    def test_optimal_rate_is_non_negative(self):
        params = LiquidationParams(q0=20.0)
        times = np.linspace(0.0, params.T, 25)
        q = inventory_path(times, params)
        rate = optimal_rate(times, q, params)
        self.assertTrue(np.all(rate >= 0.0))

    def test_liquidation_comparative_statics(self):
        times = np.array([0.0, 1.0])
        baseline = LiquidationParams(T=1.0, q0=100.0, kappa=1.0, terminal_penalty=10.0)
        higher_penalty = LiquidationParams(T=1.0, q0=100.0, kappa=1.0, terminal_penalty=40.0)
        higher_impact = LiquidationParams(T=1.0, q0=100.0, kappa=3.0, terminal_penalty=10.0)

        baseline_terminal = inventory_path(times, baseline)[-1]
        penalty_terminal = inventory_path(times, higher_penalty)[-1]
        impact_terminal = inventory_path(times, higher_impact)[-1]

        self.assertLess(penalty_terminal, baseline_terminal)
        self.assertGreater(impact_terminal, baseline_terminal)

    def test_simulation_is_reproducible(self):
        params = LiquidationParams(q0=10.0, sigma=0.2)
        first = simulate_liquidation(params, n_steps=20, n_paths=3, seed=123)
        second = simulate_liquidation(params, n_steps=20, n_paths=3, seed=123)

        np.testing.assert_allclose(first["midprice"], second["midprice"])
        np.testing.assert_allclose(first["cash"], second["cash"])
        np.testing.assert_allclose(first["inventory"], second["inventory"])


if __name__ == "__main__":
    unittest.main()
