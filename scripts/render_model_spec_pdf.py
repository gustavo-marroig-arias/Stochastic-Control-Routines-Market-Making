"""Render the formal model specification PDF.

This script avoids relying on a local TeX installation. It uses Matplotlib's
math renderer to produce a GitHub-readable PDF for the formal model statement.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "model_specification.pdf"


class PdfRenderer:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.pdf = PdfPages(output)
        self.fig = None
        self.y = 0.94
        self.page_no = 0

    def close(self) -> None:
        self._finish_page()
        self.pdf.close()

    def _new_page(self) -> None:
        self._finish_page()
        self.page_no += 1
        self.fig = plt.figure(figsize=(8.27, 11.69))
        self.fig.patch.set_facecolor("white")
        self.y = 0.94
        self.fig.text(
            0.5,
            0.03,
            f"Stochastic Control Routines for Optimal Execution and Market Making | {self.page_no}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

    def _finish_page(self) -> None:
        if self.fig is not None:
            self.pdf.savefig(self.fig, bbox_inches="tight")
            plt.close(self.fig)
            self.fig = None

    def _ensure_space(self, needed: float) -> None:
        if self.fig is None or self.y - needed < 0.08:
            self._new_page()

    def title(self, text: str, subtitle: str) -> None:
        self._new_page()
        self.fig.text(0.08, self.y, text, fontsize=18, weight="bold", va="top")
        self.y -= 0.045
        self.fig.text(0.08, self.y, subtitle, fontsize=11, color="#444444", va="top")
        self.y -= 0.045

    def heading(self, text: str) -> None:
        self._ensure_space(0.06)
        self.y -= 0.012
        self.fig.text(0.08, self.y, text, fontsize=14, weight="bold", va="top")
        self.y -= 0.04

    def subheading(self, text: str) -> None:
        self._ensure_space(0.34)
        self.fig.text(0.08, self.y, text, fontsize=11, weight="bold", va="top")
        self.y -= 0.032

    def paragraph(self, text: str, width: int = 100) -> None:
        lines = textwrap.wrap(text, width=width)
        self._ensure_space(0.021 * len(lines) + 0.01)
        for line in lines:
            self.fig.text(0.08, self.y, line, fontsize=9.5, va="top")
            self.y -= 0.021
        self.y -= 0.006

    def bullets(self, items: list[str]) -> None:
        self._ensure_space(0.025 * len(items) + 0.01)
        for item in items:
            lines = textwrap.wrap(item, width=92)
            self.fig.text(0.095, self.y, u"\u2022", fontsize=9.5, va="top")
            for j, line in enumerate(lines):
                self.fig.text(0.115, self.y, line, fontsize=9.5, va="top")
                if j < len(lines) - 1:
                    self.y -= 0.021
            self.y -= 0.024
        self.y -= 0.004

    def equations(self, lines: list[str], fontsize: float = 10.5) -> None:
        self._ensure_space(0.032 * len(lines) + 0.01)
        for line in lines:
            self.fig.text(0.5, self.y, f"${line}$", fontsize=fontsize, ha="center", va="top")
            self.y -= 0.034
        self.y -= 0.006


def render() -> None:
    r = PdfRenderer(OUTPUT)
    r.title(
        "Mathematical Model Specification",
        "Stochastic Control Routines for Optimal Execution and Market Making",
    )
    r.paragraph(
        "This document states the two implemented stochastic-control models without relying on external notes. "
        "The project is synthetic and model-based; it is not a calibrated trading strategy or a production "
        "market-making system."
    )

    r.heading("1. Closed-Form Optimal Liquidation")
    r.subheading("State, control, and parameters")
    r.bullets(
        [
            "Q_t: remaining inventory.",
            "S_t: unaffected midprice.",
            "X_t: cash account.",
            "a_t: selling rate, with positive values representing sales.",
            "T, sigma, kappa, theta: horizon, midprice volatility, temporary impact, and terminal inventory penalty.",
        ]
    )
    r.paragraph(
        "Admissible selling rates are progressively measurable, nonnegative, and square-integrable over the horizon. "
        "For the nonnegative initial inventories used in this project, the optimizer below is nonnegative."
    )
    r.subheading("Dynamics")
    r.equations(
        [
            r"dQ_t = -a_t\,dt",
            r"dS_t = \sigma\,dW_t",
            r"\widehat{S}_t = S_t-\kappa a_t",
            r"dX_t = a_t\widehat{S}_t\,dt = a_t(S_t-\kappa a_t)\,dt",
        ]
    )
    r.subheading("Objective and closed form")
    r.equations(
        [
            r"\sup_a\,\mathbb{E}\left[X_T + Q_T S_T - \theta Q_T^2\right]",
            r"V(t,S,x,q)=x+Sq+\gamma(t)q^2",
            r"\gamma'(t)=-\frac{\gamma(t)^2}{\kappa},\quad \gamma(T)=-\theta",
            r"\gamma(t)=-\left(\frac{1}{\theta}+\frac{T-t}{\kappa}\right)^{-1}",
            r"a^*(t,q)=-\frac{\gamma(t)}{\kappa}q",
            r"Q(t)=Q(0)\frac{\kappa/\theta+T-t}{\kappa/\theta+T}",
        ],
        fontsize=10.0,
    )
    r.paragraph(
        "The finite terminal penalty does not impose the hard constraint Q_T = 0. Residual terminal inventory remains "
        "unless the terminal penalty is taken to a limiting hard-liquidation regime."
    )

    r.heading("2. Symmetric Limit-Order Market Making")
    r.subheading("State, controls, and parameters")
    r.bullets(
        [
            "S_t: midprice; X_t: cash account; Q_t: inventory.",
            "delta_b and delta_a: bid and ask quote distances from the midprice.",
            "Admissible quote distances are predictable and nonnegative.",
            "Delta, lambda, kappa, sigma, alpha, phi: order size, baseline intensity, fill-decay, volatility, terminal inventory penalty, and running inventory penalty.",
            "q_min and q_max: finite inventory bounds. Boundary quotes are suppressed by setting the corresponding quote distance to infinity.",
        ]
    )
    r.subheading("Fill intensities and state dynamics")
    r.equations(
        [
            r"\lambda_b(\delta_b)=\lambda e^{-\kappa\delta_b},\quad \lambda_a(\delta_a)=\lambda e^{-\kappa\delta_a}",
            r"dS_t=\sigma\,dW_t",
            r"dQ_t=\Delta\,dN_t^b-\Delta\,dN_t^a",
            r"dX_t=-\Delta(S_t-\delta_b)\,dN_t^b+\Delta(S_t+\delta_a)\,dN_t^a",
        ],
        fontsize=10.0,
    )
    r.subheading("Objective and terminal condition")
    r.equations(
        [
            r"\sup_{\delta_b,\delta_a}\mathbb{E}\left[X_T+Q_T S_T-\alpha Q_T^2-\int_0^T \phi Q_t^2\,dt\right]",
            r"v(T,S,x,q)=x+Sq-\alpha q^2",
        ],
        fontsize=9.5,
    )
    r.subheading("Bellman equation")
    r.equations(
        [
            r"0=\partial_t v+\frac{1}{2}\sigma^2\partial_{SS}v-\phi q^2",
            r"\quad+\lambda\sup_{\delta_b\geq0}e^{-\kappa\delta_b}\left[v(t,S,x-\Delta(S-\delta_b),q+\Delta)-v(t,S,x,q)\right]\mathbf{1}_{q<q_{\max}}",
            r"\quad+\lambda\sup_{\delta_a\geq0}e^{-\kappa\delta_a}\left[v(t,S,x+\Delta(S+\delta_a),q-\Delta)-v(t,S,x,q)\right]\mathbf{1}_{q>q_{\min}}",
        ],
        fontsize=8.4,
    )

    r.heading("3. Bellman Reduction and Optimal Quotes")
    r.subheading("Ansatz and jump differences")
    r.equations(
        [
            r"v(t,S,x,q)=x+Sq+\theta(t,q)",
            r"v(t,S,x-\Delta(S-\delta_b),q+\Delta)-v(t,S,x,q)=\Delta\delta_b+\theta(t,q+\Delta)-\theta(t,q)",
            r"v(t,S,x+\Delta(S+\delta_a),q-\Delta)-v(t,S,x,q)=\Delta\delta_a+\theta(t,q-\Delta)-\theta(t,q)",
        ],
        fontsize=8.8,
    )
    r.paragraph(
        "Since the ansatz is linear in S, the second derivative with respect to S is zero. Thus sigma affects simulated "
        "marked-to-market outcomes but does not enter the transformed ODE or quote policy in this symmetric model."
    )
    r.subheading("Quote first-order conditions")
    r.equations(
        [
            r"g_b(t,q)=\frac{\theta(t,q+\Delta)-\theta(t,q)}{\Delta},\quad g_a(t,q)=\frac{\theta(t,q-\Delta)-\theta(t,q)}{\Delta}",
            r"\delta_b^*(t,q)=\frac{1}{\kappa}-g_b(t,q),\quad \delta_a^*(t,q)=\frac{1}{\kappa}-g_a(t,q)",
            r"\delta_b^*(t,q)=\frac{1}{\kappa}-\frac{\theta(t,q+\Delta)-\theta(t,q)}{\Delta}",
            r"\delta_a^*(t,q)=\frac{1}{\kappa}-\frac{\theta(t,q-\Delta)-\theta(t,q)}{\Delta}",
            r"\delta_b^*(t,q_{\max})=+\infty,\quad \delta_a^*(t,q_{\min})=+\infty",
        ],
        fontsize=9.4,
    )
    r.paragraph(
        "The closed-form quote formulas are used only in the interior-positive regime. If finite quote distances become "
        "nonpositive, the constrained-control problem would need to be solved with the nonnegativity constraint active."
    )
    r.subheading("Linear ODE transform")
    r.equations(
        [
            r"\theta(t,q)=\frac{\Delta}{\kappa}\log W(t,q)",
            r"W'(t)+AW(t)=0,\quad W(t)=\exp((T-t)A)W(T)",
            r"W(T,q)=\exp\left(-\frac{\kappa}{\Delta}\alpha q^2\right)",
            r"A_{q,q}=-\frac{\kappa}{\Delta}\phi q^2,\quad A_{q,q+\Delta}=\lambda e^{-1},\quad A_{q,q-\Delta}=\lambda e^{-1}",
        ],
        fontsize=9.2,
    )

    r.heading("4. Simulation Discretization")
    r.paragraph(
        "The simulation approximates continuous-time Poisson fills over a time step Delta t with a first-event "
        "discretization."
    )
    r.equations(
        [
            r"\lambda_b=\lambda e^{-\kappa\delta_b},\quad \lambda_a=\lambda e^{-\kappa\delta_a},\quad \Lambda=\lambda_b+\lambda_a",
            r"p_{\mathrm{fill}}=1-\exp(-\Lambda\,\Delta t)",
            r"\mathbb{P}(\mathrm{bid}\mid\mathrm{fill})=\frac{\lambda_b}{\Lambda},\quad \mathbb{P}(\mathrm{ask}\mid\mathrm{fill})=\frac{\lambda_a}{\Lambda}",
            r"X_{n+1}=X_n-\Delta(S_n-\delta_b),\quad Q_{n+1}=Q_n+\Delta\quad \mathrm{(bid\ fill)}",
            r"X_{n+1}=X_n+\Delta(S_n+\delta_a),\quad Q_{n+1}=Q_n-\Delta\quad \mathrm{(ask\ fill)}",
            r"R_T^{\Delta t}=\sum_{n=0}^{N-1}\phi Q_n^2\,\Delta t",
            r"X_N+Q_NS_N-\alpha Q_N^2-R_T^{\Delta t}",
        ],
        fontsize=9.0,
    )
    r.paragraph(
        "The simulator reports both terminal marked-to-market value and the objective-consistent discretized payoff. "
        "The latter subtracts the pathwise running inventory penalty."
    )
    r.close()


if __name__ == "__main__":
    render()
    print(f"Wrote {OUTPUT}")
