import unittest

import numpy as np

from stoch_control_trading import (
    MarketMakingParams,
    inventory_grid,
    market_making_generator,
    optimal_quotes,
    simulate_market_maker,
    solve_market_making_ode,
)


class MarketMakingTests(unittest.TestCase):
    def test_generator_shape_and_entries(self):
        params = MarketMakingParams()
        q_grid = inventory_grid(params)
        A = market_making_generator(params)

        self.assertEqual(A.shape, (len(q_grid), len(q_grid)))
        np.testing.assert_allclose(np.diag(A), -(params.kappa / params.Delta) * params.phi * q_grid**2)
        self.assertAlmostEqual(A[3, 4], params.lam * np.exp(-1.0))
        self.assertAlmostEqual(A[4, 3], params.lam * np.exp(-1.0))

    def test_terminal_theta_and_positive_transformed_value(self):
        params = MarketMakingParams()
        solution = solve_market_making_ode(params, np.array([0.0, params.T]))
        q_grid = solution["q_grid"]

        self.assertTrue(np.all(solution["W"] > 0.0))
        np.testing.assert_allclose(solution["theta"][-1], -params.alpha * q_grid**2, atol=1e-12)

    def test_boundary_quote_suppression_positive_interior_and_symmetry(self):
        params = MarketMakingParams()
        solution = solve_market_making_ode(params, np.array([0.0]))
        quotes = optimal_quotes(solution["theta"][0], params)
        bid = quotes["bid"]
        ask = quotes["ask"]

        self.assertTrue(np.isinf(ask[0]))
        self.assertTrue(np.isinf(bid[-1]))
        self.assertTrue(np.all(bid[:-1] > 0.0))
        self.assertTrue(np.all(ask[1:] > 0.0))
        np.testing.assert_allclose(bid[:-1], ask[:0:-1], atol=1e-12)

    def test_inventory_skew_comparative_statics(self):
        params = MarketMakingParams()
        solution = solve_market_making_ode(params, np.array([0.0]))
        quotes = optimal_quotes(solution["theta"][0], params)
        q_grid = quotes["q_grid"]
        bid = quotes["bid"]
        ask = quotes["ask"]

        zero_idx = int(np.flatnonzero(np.isclose(q_grid, 0.0))[0])
        long_idx = int(np.flatnonzero(np.isclose(q_grid, 4.0))[0])
        short_idx = int(np.flatnonzero(np.isclose(q_grid, -4.0))[0])

        self.assertGreater(bid[long_idx], bid[zero_idx])
        self.assertLess(ask[long_idx], ask[zero_idx])
        self.assertLess(bid[short_idx], bid[zero_idx])
        self.assertGreater(ask[short_idx], ask[zero_idx])

    def test_nonpositive_interior_quotes_are_rejected(self):
        params = MarketMakingParams(q_min=0.0, q_max=1.0)
        theta = np.array([0.0, 2.0])

        with self.assertRaises(ValueError):
            optimal_quotes(theta, params)

    def test_simulation_is_reproducible_and_respects_inventory_bounds(self):
        params = MarketMakingParams()
        first = simulate_market_maker(params, n_steps=50, n_paths=4, seed=9)
        second = simulate_market_maker(params, n_steps=50, n_paths=4, seed=9)

        np.testing.assert_allclose(first["inventory"], second["inventory"])
        np.testing.assert_allclose(first["cash"], second["cash"])
        self.assertGreaterEqual(first["inventory"].min(), params.q_min)
        self.assertLessEqual(first["inventory"].max(), params.q_max)
        self.assertTrue(np.all(np.isin(first["fill_side"], [-1, 0, 1])))
        np.testing.assert_allclose(
            first["objective_value"],
            first["terminal_value"] - first["running_inventory_penalty"],
        )
        self.assertTrue(np.all(first["running_inventory_penalty"] >= 0.0))

    def test_simulated_objective_is_consistent_with_bellman_value(self):
        params = MarketMakingParams()
        solution = solve_market_making_ode(params, np.array([0.0]))
        q_grid = solution["q_grid"]
        q0_index = int(np.flatnonzero(np.isclose(q_grid, params.q0))[0])
        bellman_value = (
            params.x0 + params.s0 * params.q0 + solution["theta"][0, q0_index]
        )

        simulation = simulate_market_maker(params, n_steps=300, n_paths=400, seed=2024)
        objective_values = simulation["objective_value"]
        mc_mean = float(np.mean(objective_values))
        mc_stderr = float(np.std(objective_values, ddof=1) / np.sqrt(objective_values.size))
        half_width = 4.0 * mc_stderr

        self.assertLessEqual(abs(mc_mean - bellman_value), half_width)

    def test_buy_and_sell_cash_signs(self):
        buy_params = MarketMakingParams(T=0.1, q_min=-1.0, q_max=1.0, q0=-1.0, lam=1000.0, s0=100.0)
        buy = simulate_market_maker(buy_params, n_steps=1, n_paths=1, seed=1)
        self.assertGreater(buy["inventory"][0, -1], buy["inventory"][0, 0])
        self.assertLess(buy["cash"][0, -1], buy["cash"][0, 0])

        sell_params = MarketMakingParams(T=0.1, q_min=-1.0, q_max=1.0, q0=1.0, lam=1000.0, s0=100.0)
        sell = simulate_market_maker(sell_params, n_steps=1, n_paths=1, seed=1)
        self.assertLess(sell["inventory"][0, -1], sell["inventory"][0, 0])
        self.assertGreater(sell["cash"][0, -1], sell["cash"][0, 0])


if __name__ == "__main__":
    unittest.main()
