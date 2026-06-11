# tailcor

Non-asymptotic, distribution-free decomposition of linear correlation vs non-linear tail contagion — a Python implementation of **Ricci, Tristani & Vergote (2023) "TailCoR"** (Banco de España WP 1227, Journal of Empirical Finance). Pure numpy + scipy, no C++ deps, no copula fitting, no asymptotic limits.

## What it does

Given two paired time series `(X, Y)`, TailCoR returns three numbers per window:

- **Linear** — always 1.0. Bivariate-Gaussian reference at the empirical Pearson correlation.
- **Composite** — empirical tail inter-quantile range divided by the Gaussian reference. Equals 1.0 under exact Gaussian, >1 when the joint tail is fatter than Gaussian predicts.
- **Non-linear** — `composite − linear`. The "pure tail-contagion" component that correlation cannot explain.

Why it matters: during the 2008 + 2011 crises, linear correlations between Euro-area sovereign bonds *collapsed to ~1 then dropped*, while tail contagion *spiked*. Classical EVT was too slow to catch it (extremes are scarce at acceleration); TailCoR tracked it in real time because it does not need asymptotic-limit samples.

## Install

```bash
pip install tailcor
```

Requires Python ≥ 3.9, numpy ≥ 1.23, scipy ≥ 1.10.

## Quickstart

```python
import numpy as np
from tailcor import tailcor, rolling_tailcor

rng = np.random.default_rng(0)
x = rng.standard_normal(1000)
y = 0.5 * x + rng.standard_normal(1000)

result = tailcor(x, y, q=0.95)
print(result.composite)   # ~1.0 under Gaussian
print(result.nonlinear)   # ~0.0 under Gaussian
print(result.rho)         # empirical Pearson

# rolling time series:
r = rolling_tailcor(x, y, window=252, q=0.95)
print(r.composite.shape)  # (1000 - 252 + 1,)
```

## The math (1-minute version)

For bivariate `(X, Y)` with empirical Pearson correlation `ρ`, project onto the sum axis standardised by the Gaussian variance:

```
Z = (X + Y) / √(2(1 + ρ))
```

Under bivariate standard Gaussian at any `ρ`, `Z` is `N(0, 1)` — so its inter-quantile range at level `q` equals `Φ⁻¹(q) − Φ⁻¹(1−q)` (the Gaussian reference IQR at that quantile).

Define

```
S(q) = IQR_q(Z) / ( Φ⁻¹(q) − Φ⁻¹(1−q) )
```

- `S(q) ≈ 1` under exact bivariate Gaussian (**any** `ρ`)
- `S(q) > 1` when the empirical tail is fatter than Gaussian (**tail contagion**)
- `S(q) < 1` when the empirical tail is thinner (rare — anti-concentration)

The linear component is exactly 1.0 by construction (Gaussian reference). The non-linear component is `S(q) − 1`. "A few lines of basic coding", per the paper — and literally is (see [`tailcor/core.py`](tailcor/core.py)).

## Quantile profile

Fat-tailed joint distributions show TailCoR *rising with q*:

```python
from tailcor import tailcor_profile
import numpy as np

rng = np.random.default_rng(1)
# Student-t copula with df=4 (canonical tail dependence)
g = rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=15000).T
chi2 = rng.chisquare(4, size=15000)
t = g * np.sqrt(4 / chi2)
x, y = t[0], t[1]

for r in tailcor_profile(x, y):
    print(f"q={r.q:.2f}  composite={r.composite:.3f}  nonlinear={r.nonlinear:+.3f}")
```

Output: composite starts ~1 at `q=0.6`, grows past `1.5` at `q=0.99` — the classic "tail contagion signature" that Ricci et al. use as a crisis early-warning.

## Correctness

15 unit tests cover:
- Bivariate Gaussian at six `ρ` levels (−0.5 … 0.8): composite within `[0.9, 1.1]` of 1, non-linear near 0
- Student-t copula `df=4`: composite > 1.15 even at `ρ=0` (tail contagion with zero Pearson)
- Student-t copula `df=30`: nearly Gaussian (`|non-linear| < 0.15`)
- `ρ` recovered to within 0.05 on `n=10000` Gaussian samples
- NaN-row handling (10% injection preserves a valid result)
- `q` monotonicity on tail-dependent copulas (composite grows with `q`)
- Input-validation error paths
- Rolling: Gaussian → Student-t regime shift produces post-shift composite lift > 0.1

## Use cases

- **Crisis early warning** on bivariate index pairs (e.g. equity vs sovereign, or two credit spreads)
- **Risk decomposition**: split pairwise CoVaR into what Gaussian explains and what it can't
- **Stress-test input**: feed the non-linear component into correlation-matrix generators (CorrGAN and similar) as a fat-tail signal
- **Regime detection** complementary to HMM / change-point methods: the non-linear component reacts to copula-shape changes that HMM trained on returns may miss

## Roadmap

**v0.2 (planned):**
- Alternative rotations (principal-component, anti-diagonal) — `rotation="sum"|"pc"|"anti"`
- Tail-CoVaR helper — conditional-VaR-style wrapper on top of the core decomposition
- Block-bootstrap confidence intervals for composite + non-linear
- Multivariate extension via pairwise aggregation matrix

## Authors

- **Pierre Samson** ([@darw007d](https://github.com/darw007d)) — idea, use-case, design decisions
- **Claude Opus** (Anthropic) — implementation and tests

Originally motivated by a quantitative-finance application (tail-contagion decomposition). Sister package to [phawkes](https://pypi.org/project/phawkes/) (Hawkes processes) and [fisherrao](https://pypi.org/project/fisherrao/) (information geometry). Same "small, tested, publishable" ethos.

## Citation

If this library contributes to a published result, please cite the underlying paper:

- Ricci, L., Tristani, O. & Vergote, O. (2023). *TailCoR*. Banco de España Working Paper 1227. Journal of Empirical Finance.

## License

MIT — see [LICENSE](LICENSE).
