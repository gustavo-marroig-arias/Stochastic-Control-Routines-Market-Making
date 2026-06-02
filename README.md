# Stochastic Control Routines for Optimal Execution and Market Making

[![Tests](https://github.com/gustavo-marroig-arias/Stochastic-Control-Routines-Market-Making/actions/workflows/tests.yml/badge.svg)](https://github.com/gustavo-marroig-arias/Stochastic-Control-Routines-Market-Making/actions/workflows/tests.yml)

Compact Python implementations of stochastic-control routines for optimal liquidation and finite-grid inventory-based market making, with simulations, diagnostics, and tests.

**Code structure.** Core model code lives in `src/stoch_control_trading/`; the notebook is only the reproducible presentation layer.

This repository implements two compact stochastic-control examples used in execution and market-making theory:

- closed-form optimal liquidation with temporary price impact and terminal inventory penalty;
- symmetric limit-order market making on a finite inventory grid, solved through a Bellman-equation reduction.

The project is deliberately synthetic. It is intended to demonstrate the mechanics of stochastic-control formulations, HJB/Bellman reductions, inventory-risk trade-offs, and simulation under controlled fill intensities. It is not a calibrated trading strategy or a production market-making system.

## Repository Structure

```text
notebooks/stochastic_control_trading.ipynb   Reproducible notebook and plots
src/stoch_control_trading/                   Model implementations
tests/                                       Unit tests for formulas and simulations
docs/model_specification.pdf                 Rendered mathematical model specification
docs/model_specification.tex                 LaTeX source for the model specification
docs/model_specification.md                  GitHub-friendly pointer to the PDF
docs/results_discussion.md                   Formal interpretation of model outputs
outputs/figures/                             Generated figures
pyproject.toml                               Editable package metadata
scripts/render_model_spec_pdf.py             PDF renderer for the mathematical specification
```

## Setup

```bash
conda env create -f environment.yml
conda activate sc-control-mm
python -m pip install -e . --no-build-isolation
```

## One-Command Reproducibility

After creating and activating the conda environment, this command installs the package, regenerates the mathematical specification PDF, runs the tests, executes the notebook, and checks that the figures were regenerated:

```bash
python -m pip install -e . --no-build-isolation &&
MPLCONFIGDIR=.cache/matplotlib python scripts/render_model_spec_pdf.py &&
MPLCONFIGDIR=.cache/matplotlib python -m unittest discover -s tests &&
MPLCONFIGDIR=.cache/matplotlib jupyter nbconvert --to notebook --execute notebooks/stochastic_control_trading.ipynb --output executed_stochastic_control_trading.ipynb &&
python -c "from pathlib import Path; figs=sorted(Path('outputs/figures').glob('*.png')); assert len(figs)==3, figs; print(f'{len(figs)} figures generated')"
```

Expected output: the PDF step prints `Wrote .../docs/model_specification.pdf`, the test run ends with `OK` after `Ran 13 tests`, and the final line prints `3 figures generated`.

The notebook saves figures to `outputs/figures/`. The full mathematical specification is in [docs/model_specification.pdf](docs/model_specification.pdf).

## Models Implemented

The complete mathematical statement is in [docs/model_specification.pdf](docs/model_specification.pdf). This README summarizes the implemented state dynamics, controls, and numerical reduction.

### Optimal Liquidation

The liquidation model describes an agent selling an initial inventory over a fixed horizon. The control is the selling rate `a_t`. Trading faster reduces terminal inventory but worsens execution price through temporary impact.

State variables are inventory `Q_t`, midprice `S_t`, and cash `X_t`. The no-permanent-impact dynamics are:

$$\mathrm{d}Q_t = -a_t \mathrm{d}t$$

$$\mathrm{d}S_t = \sigma \mathrm{d}W_t$$

$$\widehat{S}_t = S_t-\kappa a_t$$

$$\mathrm{d}X_t = a_t\widehat{S}_t \mathrm{d}t$$

The objective is terminal cash plus terminal mark-to-market inventory value, penalized by unsold inventory:

$$
\mathbb{E}\left[X_T + Q_T S_T - \theta Q_T^2\right]
$$

For this no-permanent-impact case, the Riccati coefficient and optimal selling rate are:

$$\gamma(t) = -\left(\frac{1}{\theta}+\frac{T-t}{\kappa}\right)^{-1}$$

$$a^*(t,q) = -\frac{\gamma(t)}{\kappa}q$$

The deterministic inventory path under this policy is

$$
Q(t)=Q(0)\frac{\kappa/\theta+T-t}{\kappa/\theta+T}.
$$

### Limit-Order Market Making

The market-making model describes a dealer posting one bid and one ask around an exogenous midprice. The controls are quote distances `delta_b` and `delta_a`. Closer quotes fill more often; wider quotes earn more spread conditional on a fill.

State variables are midprice `S_t`, cash `X_t`, and inventory `Q_t`. Fill intensities decay with quote distance:

$$\lambda_b(\delta_b) = \lambda e^{-\kappa\delta_b}$$

$$\lambda_a(\delta_a) = \lambda e^{-\kappa\delta_a}$$

With bid-fill and ask-fill counting processes `N_t^b` and `N_t^a`, the state dynamics are:

$$\mathrm{d}S_t = \sigma \mathrm{d}W_t$$

$$\mathrm{d}Q_t = \Delta \mathrm{d}N_t^b-\Delta \mathrm{d}N_t^a$$

$$\mathrm{d}X_t = -\Delta(S_t-\delta_b)\mathrm{d}N_t^b + \Delta(S_t+\delta_a)\mathrm{d}N_t^a$$

The objective is terminal marked-to-market wealth with terminal and running inventory penalties:

$$
\mathbb{E}\left[X_T + Q_T S_T - \alpha Q_T^2 - \int_0^T \phi Q_t^2 \mathrm{d}t\right]
$$

The implementation uses the symmetric finite-grid, interior-positive quote regime. The value-function ansatz is

$$
v(t,S,x,q)=x+Sq+\theta(t,q).
$$

The quote distances are recovered from neighboring inventory values:

$$\delta_{b}^{*}(t,q) = \frac{1}{\kappa} - \frac{\theta(t,q+\Delta)-\theta(t,q)}{\Delta}$$

$$\delta_{a}^{*}(t,q) = \frac{1}{\kappa} - \frac{\theta(t,q-\Delta)-\theta(t,q)}{\Delta}$$

Boundary quotes that would move inventory outside the finite grid are suppressed. The nonlinear Bellman equation becomes a linear ODE after

$$
\theta(t,q)=\frac{\Delta}{\kappa}\log W(t,q).
$$

The resulting system is solved by matrix exponential:

$$
W(t)=\exp((T-t)A)W(T).
$$

See [docs/model_specification.pdf](docs/model_specification.pdf) for the full Bellman equation, terminal condition, ODE matrix, admissibility conditions, boundary convention, and first-event simulation discretization.

## Results

### Liquidation Policy

![Optimal liquidation inventory paths](outputs/figures/liquidation_inventory_paths.png)

Higher terminal inventory penalty accelerates liquidation, while higher temporary impact slows it down. In this no-permanent-impact, risk-neutral formulation, the optimal selling rate is constant along the deterministic inventory trajectory.

### Market-Making Quote Skew

![Optimal market-making quote distances](outputs/figures/market_making_quotes.png)

Inventory skews the optimal quotes. When inventory is long, the ask is placed closer to the midprice and the bid is placed farther away. When inventory is short, the opposite happens. This is the central inventory-control mechanism in the model.

### Market-Making Simulation

![Market-making simulation summary](outputs/figures/market_making_simulation_summary.png)

The simulation uses the optimal quote distances to generate a first-event discrete-time approximation of controlled Poisson fills. The histogram compares terminal marked-to-market value with the objective-adjusted value after subtracting the running inventory penalty. Both should be interpreted as model diagnostics, not as performance estimates.

For a fuller interpretation, see [docs/results_discussion.md](docs/results_discussion.md).

## Tests

The test suite checks:

- liquidation terminal condition, monotone inventory path, nonnegative selling rate, comparative statics, and fixed-seed reproducibility;
- market-making ODE matrix structure, terminal condition, quote symmetry, boundary suppression, positive finite quote distances, inventory-skew comparative statics, inventory bounds, cash-update signs, objective-value accounting, Bellman-value simulation sanity check, and fixed-seed reproducibility.

Run:

```bash
MPLCONFIGDIR=.cache/matplotlib python -m unittest discover -s tests
```

## Scope And Limitations

- Synthetic model only; no calibration to real exchange data.
- No tick size, queue priority, latency, fees, adverse selection, hidden liquidity, or order-book state.
- Symmetric market-making arrival rates and distance-decay parameters.
- First-event time-discretization is used for simulation, while the model is continuous time.
- The midprice volatility affects simulated mark-to-market outcomes, but not the closed-form quote policy in the symmetric ansatz used here.
- The closed-form quote formulas are used only in the interior-positive regime; the code raises an error if finite quote distances become nonpositive.
- The market-making simulation reports terminal marked-to-market value and an objective-adjusted value that subtracts the discretized running inventory penalty.

## Model Lineage

The liquidation example is a temporary-impact optimal execution model with a quadratic terminal inventory penalty. The market-making example belongs to the inventory-risk market-making family associated with Avellaneda and Stoikov, and with Guéant, Lehalle, and Fernandez-Tapia. This repository implements a compact symmetric finite-grid version for numerical transparency.

References:

- Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit order book. Quantitative Finance.
- Guéant, O., Lehalle, C.-A. and Fernandez-Tapia, J. (2013). Dealing with the inventory risk: a solution to the market making problem. Mathematics and Financial Economics.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
