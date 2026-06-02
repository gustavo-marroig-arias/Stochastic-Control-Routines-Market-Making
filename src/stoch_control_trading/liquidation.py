"""Closed-form optimal liquidation model.

The implementation uses the no-permanent-impact case. Inventory follows
``dQ_t = -a_t dt`` and the temporary execution price is ``S_t - kappa a_t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class LiquidationParams:
    """Parameters for the closed-form liquidation example."""

    T: float = 1.0
    q0: float = 100.0
    s0: float = 100.0
    sigma: float = 1.0
    kappa: float = 1.0
    terminal_penalty: float = 10.0

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.q0 < 0:
            raise ValueError("q0 must be non-negative.")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative.")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        if self.terminal_penalty <= 0:
            raise ValueError("terminal_penalty must be positive.")


def _as_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


def gamma(t: ArrayLike, params: LiquidationParams) -> np.ndarray:
    """Return the Riccati coefficient gamma(t).

    In the mathematical notation, the terminal inventory penalty is theta:
    ``gamma(t) = -(1 / theta + (T - t) / kappa)^(-1)``.
    """

    t_arr = _as_array(t)
    denominator = (1.0 / params.terminal_penalty) + (params.T - t_arr) / params.kappa
    return -1.0 / denominator


def optimal_rate(t: ArrayLike, q: ArrayLike, params: LiquidationParams) -> np.ndarray:
    """Return the optimal selling rate a*(t, q)."""

    return -gamma(t, params) * _as_array(q) / params.kappa


def inventory_path(times: ArrayLike, params: LiquidationParams) -> np.ndarray:
    """Return the deterministic optimal inventory path for the closed-form policy."""

    t_arr = _as_array(times)
    if np.any(t_arr < -1e-12) or np.any(t_arr > params.T + 1e-12):
        raise ValueError("times must lie in [0, T].")
    impact_to_penalty = params.kappa / params.terminal_penalty
    scale = (impact_to_penalty + params.T - t_arr) / (impact_to_penalty + params.T)
    return params.q0 * scale


def simulate_liquidation(
    params: LiquidationParams,
    n_steps: int = 500,
    n_paths: int = 32,
    seed: int | None = 7,
) -> dict[str, np.ndarray]:
    """Simulate cash, midprice, inventory, and terminal value under the policy.

    Inventory and selling rate are deterministic in this no-permanent-impact setup;
    the Monte Carlo component enters through the exogenous midprice.
    """

    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, params.T, n_steps + 1)
    dt = params.T / n_steps
    q_path = inventory_path(times, params)
    rates = optimal_rate(times[:-1], q_path[:-1], params)

    brownian_steps = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
    midprice = np.empty((n_paths, n_steps + 1), dtype=float)
    midprice[:, 0] = params.s0
    midprice[:, 1:] = params.s0 + params.sigma * np.cumsum(brownian_steps, axis=1)

    execution_price = midprice[:, :-1] - params.kappa * rates[None, :]
    cash = np.zeros((n_paths, n_steps + 1), dtype=float)
    cash[:, 1:] = np.cumsum(execution_price * rates[None, :] * dt, axis=1)

    inventory = np.tile(q_path, (n_paths, 1))
    terminal_value = (
        cash[:, -1]
        + inventory[:, -1] * midprice[:, -1]
        - params.terminal_penalty * inventory[:, -1] ** 2
    )

    return {
        "times": times,
        "inventory": inventory,
        "selling_rate": np.tile(np.r_[rates, rates[-1]], (n_paths, 1)),
        "midprice": midprice,
        "cash": cash,
        "terminal_value": terminal_value,
    }
