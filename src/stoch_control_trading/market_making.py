"""Symmetric limit-order market-making toy model via the Bellman ODE reduction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class MarketMakingParams:
    """Parameters for the symmetric market-making example."""

    T: float = 1.0
    Delta: float = 1.0
    q_min: float = -5.0
    q_max: float = 5.0
    lam: float = 1.5
    kappa: float = 1.0
    alpha: float = 0.02
    phi: float = 0.005
    sigma: float = 0.5
    s0: float = 100.0
    q0: float = 0.0
    x0: float = 0.0

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.Delta <= 0:
            raise ValueError("Delta must be positive.")
        if self.q_min >= self.q_max:
            raise ValueError("q_min must be smaller than q_max.")
        if self.lam <= 0:
            raise ValueError("lam must be positive.")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if self.phi < 0:
            raise ValueError("phi must be non-negative.")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative.")
        grid = inventory_grid(self)
        if not np.any(np.isclose(grid, self.q0)):
            raise ValueError("q0 must lie on the inventory grid.")


def inventory_grid(params: MarketMakingParams) -> np.ndarray:
    """Return the evenly spaced inventory grid."""

    n_float = (params.q_max - params.q_min) / params.Delta
    n_steps = int(round(n_float))
    if not np.isclose(n_float, n_steps):
        raise ValueError("q_min and q_max must be separated by an integer number of Delta.")
    return params.q_min + params.Delta * np.arange(n_steps + 1)


def market_making_generator(params: MarketMakingParams) -> np.ndarray:
    """Build the linear ODE matrix A for W'(t) + A W(t) = 0."""

    q_grid = inventory_grid(params)
    n = len(q_grid)
    A = np.zeros((n, n), dtype=float)
    off_diag = params.lam * np.exp(-1.0)

    for i, q in enumerate(q_grid):
        A[i, i] = -(params.kappa / params.Delta) * params.phi * q**2
        if i > 0:
            A[i, i - 1] = off_diag
        if i < n - 1:
            A[i, i + 1] = off_diag

    return A


def solve_market_making_ode(
    params: MarketMakingParams,
    times: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Solve the transformed linear Bellman system on the inventory grid."""

    if times is None:
        times = np.linspace(0.0, params.T, 101)
    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    if np.any(times < -1e-12) or np.any(times > params.T + 1e-12):
        raise ValueError("times must lie in [0, T].")

    q_grid = inventory_grid(params)
    A = market_making_generator(params)
    terminal_w = np.exp(-(params.kappa / params.Delta) * params.alpha * q_grid**2)

    W = np.vstack([expm((params.T - t) * A) @ terminal_w for t in times])
    if np.any(W <= 0):
        raise FloatingPointError("The transformed value W must remain positive.")

    theta = (params.Delta / params.kappa) * np.log(W)
    return {
        "times": times,
        "q_grid": q_grid,
        "A": A,
        "W": W,
        "theta": theta,
    }


def optimal_quotes(theta: np.ndarray, params: MarketMakingParams) -> dict[str, np.ndarray]:
    """Recover optimal bid and ask distances from theta(t, q).

    Bid quotes are suppressed at q_max and ask quotes are suppressed at q_min by
    returning ``np.inf`` at those boundaries. The closed-form formulas correspond
    to the interior optimum for non-negative quote distances, so finite quotes must
    remain strictly positive for this toy implementation.
    """

    theta_arr = np.asarray(theta, dtype=float)
    squeeze = theta_arr.ndim == 1
    if squeeze:
        theta_arr = theta_arr[None, :]
    if theta_arr.ndim != 2:
        raise ValueError("theta must be one- or two-dimensional.")

    n_grid = len(inventory_grid(params))
    if theta_arr.shape[1] != n_grid:
        raise ValueError("theta has incompatible inventory dimension.")

    bid = np.full_like(theta_arr, np.inf, dtype=float)
    ask = np.full_like(theta_arr, np.inf, dtype=float)
    bid[:, :-1] = 1.0 / params.kappa - (theta_arr[:, 1:] - theta_arr[:, :-1]) / params.Delta
    ask[:, 1:] = 1.0 / params.kappa - (theta_arr[:, :-1] - theta_arr[:, 1:]) / params.Delta
    finite_quotes = np.concatenate((bid[np.isfinite(bid)], ask[np.isfinite(ask)]))
    if np.any(finite_quotes <= 0.0):
        raise ValueError(
            "Finite quote distances must be positive for the interior-optimum regime."
        )

    if squeeze:
        bid = bid[0]
        ask = ask[0]
    return {
        "q_grid": inventory_grid(params),
        "bid": bid,
        "ask": ask,
    }


def _grid_index(q_grid: np.ndarray, q: float) -> int:
    matches = np.flatnonzero(np.isclose(q_grid, q))
    if len(matches) != 1:
        raise ValueError("Inventory is not on the grid.")
    return int(matches[0])


def simulate_market_maker(
    params: MarketMakingParams,
    n_steps: int = 500,
    n_paths: int = 64,
    seed: int | None = 11,
) -> dict[str, np.ndarray]:
    """Simulate the controlled market maker with a first-event discretization.

    In each time step, the simulator samples whether at least one fill arrives
    from the combined bid/ask intensity. Conditional on a fill, the side is
    selected according to the relative bid and ask intensities. This keeps at
    most one inventory jump per step.
    """

    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, params.T, n_steps + 1)
    dt = params.T / n_steps

    solution = solve_market_making_ode(params, times)
    quotes = optimal_quotes(solution["theta"], params)
    q_grid = solution["q_grid"]
    initial_index = _grid_index(q_grid, params.q0)

    brownian_steps = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
    midprice = np.empty((n_paths, n_steps + 1), dtype=float)
    midprice[:, 0] = params.s0
    midprice[:, 1:] = params.s0 + params.sigma * np.cumsum(brownian_steps, axis=1)

    cash = np.zeros((n_paths, n_steps + 1), dtype=float)
    inventory = np.zeros((n_paths, n_steps + 1), dtype=float)
    fill_side = np.zeros((n_paths, n_steps), dtype=int)

    for path in range(n_paths):
        q_index = initial_index
        x = params.x0
        cash[path, 0] = x
        inventory[path, 0] = q_grid[q_index]

        for step in range(n_steps):
            s = midprice[path, step]
            bid_distance = quotes["bid"][step, q_index]
            ask_distance = quotes["ask"][step, q_index]

            bid_intensity = 0.0
            ask_intensity = 0.0
            if np.isfinite(bid_distance):
                bid_intensity = params.lam * np.exp(-params.kappa * bid_distance)
            if np.isfinite(ask_distance):
                ask_intensity = params.lam * np.exp(-params.kappa * ask_distance)

            total_intensity = bid_intensity + ask_intensity
            if total_intensity > 0.0:
                fill_arrives = rng.random() < 1.0 - np.exp(-total_intensity * dt)
                if fill_arrives:
                    bid_probability = bid_intensity / total_intensity
                    if rng.random() < bid_probability:
                        x -= params.Delta * (s - bid_distance)
                        q_index += 1
                        fill_side[path, step] = 1
                    else:
                        x += params.Delta * (s + ask_distance)
                        q_index -= 1
                        fill_side[path, step] = -1

            cash[path, step + 1] = x
            inventory[path, step + 1] = q_grid[q_index]

    terminal_value = (
        cash[:, -1]
        + inventory[:, -1] * midprice[:, -1]
        - params.alpha * inventory[:, -1] ** 2
    )
    running_inventory_penalty = params.phi * np.sum(inventory[:, :-1] ** 2, axis=1) * dt
    objective_value = terminal_value - running_inventory_penalty
    return {
        "times": times,
        "midprice": midprice,
        "cash": cash,
        "inventory": inventory,
        "fill_side": fill_side,
        "terminal_value": terminal_value,
        "running_inventory_penalty": running_inventory_penalty,
        "objective_value": objective_value,
        "q_grid": q_grid,
        "bid_quotes": quotes["bid"],
        "ask_quotes": quotes["ask"],
    }
