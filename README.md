# ExperimentLens

**E-commerce A/B Testing & Experimentation Analytics**

An experimentation analytics framework for a two-arm conversion experiment:
data validation, experiment-validity checks, frequentist and Bayesian inference,
simulation-based calibration, and an explicit business decision engine.

> **A p-value below 0.05 is never sufficient to ship.**

---

## The headline result

The experiment on the [Kaggle A/B Testing dataset](https://www.kaggle.com/datasets/sheng21/marketing-campaign-results)
(294,478 rows) concluded **DON'T SHIP** — and the interesting part is *why*.

```
CONTROL       12.0173%        TREATMENT     11.8726%
absolute      -0.1448 pp      relative      -1.2048%
95% CI        [-0.382, +0.093] pp             p (2-sided)  0.2320
94% HDI       [-3.05%, +0.65%]               P(B > A)      0.1197
P(lift > +1%) 0.0125                          E[revenue]   -5,188
```

Three independent criteria fire: the credible interval's **optimistic end
(+0.65%) is still below the 1% minimum worthwhile lift**, the posterior puts only
an 11.97% chance on treatment beating control, and expected revenue is negative.

A non-significant p-value alone would only mean *"not proven."* The
recommendation rests on a stronger claim — the interval **excludes the
worthwhile range** — which is why the verdict is a refusal rather than a shrug.

> **On correlated evidence.** The frequentist interval, the posterior interval,
> `P(B > A)` and the observed lift are **not** independent confirmations. They
> are four summaries of the same 2×2 table of conversion counts. The dashboard
> groups them into *Frequentist view / Bayesian view / Business view* and says
> so explicitly, because agreement between them shows the conclusion is robust
> to how you summarise it — it does not multiply the evidence.

A design finding is reported alongside it: the minimum detectable effect was
**+2.53%**, above the 1% practical threshold, so this experiment was never able
to settle the question it was run to answer. That is surfaced as a warning and
does **not** override the evidence.

### How the decision rule is meant to work

The two gates are deliberately asymmetric, and this is a policy choice rather
than an accident:

```
SHIP        requires ALL  4 criteria to pass
DON'T SHIP  requires ANY ONE of 3 criteria
```

Refusing a change is reversible — the experiment can be rerun. Shipping a
regression is not. Each refusing criterion also stands on its own merits:
nothing worth shipping exists in the credible range, or the action has negative
expected value, or the change is more likely neutral-to-worse than better.
A policy requiring two refusing criteria was considered and rejected: it makes
the engine *more* willing to ship, which is the wrong direction for a change
nobody is obliged to make.

---

## What this demonstrates

| Signal | Where |
|---|---|
| Hypothesis testing | Two-proportion z-test, chi-square GOF for SRM, simulation-based Type I calibration |
| Bayesian inference | PyMC Beta-Binomial, `P(B>A)`, 94% HDI, posterior expected revenue |
| Experiment design | Power, MDE, required sample size, achieved vs analytic power |
| Causal / decision discipline | Validity gates, minimum practical lift, three-outcome decision engine |
| Engineering | `src/` layout, 367 tests, mypy strict, ruff, GitHub Actions CI, Streamlit dashboard |

---

## Three rules the codebase follows

1. **Validate before cleaning.** Raw counts are snapshotted at ingest and SRM is
   computed before any exclusion. Every removal is recorded with its rationale
   and its effect on the population.
2. **Test before visualizing.** All statistical and decision logic lives in
   tested pure-Python modules. The Streamlit app contains **no** statistical
   computation — enforced by `tests/test_architecture.py`, which fails the
   build if the app imports `scipy`/`statsmodels`/`pymc` directly or the package
   imports `streamlit`.
3. **Make every conclusion reproducible.** Fixed seeds injected as arguments,
   known-answer tests, and simulation calibration with tolerance bands.

---

## Findings worth reading

**The dataset's two "data quality issues" are one issue.** The widely-cited
3,893 group/landing-page mismatches are exactly the 3,893 users who appear
twice under conflicting assignments. Verified directly:

```
users with group conflict       : 3893
users with any contaminated row : 3893
identical sets                  : True
```

Resolving duplicates as conflicts therefore leaves **zero** contaminated rows
for the contamination filter to remove. Both problems are the same population.

**Repeated users break the model, not just the aesthetics.** A repeated
`user_id` violates the independence assumption of the binomial model and can
bias arm counts in either direction. Every repeated user is classified into one
of four dispositions before any rate is computed:

| Class | Condition | Policy |
|---|---|---|
| `GROUP_CONFLICT` | more than one `(group, page)` assignment | user excluded |
| `OUTCOME_INSTALLED` | conflicting `converted` values | user excluded |
| `TIMESTAMP_SPLIT` | consistent, multiple timestamps | collapse to first exposure |
| `REDUNDANT` | identical repeated rows | collapse to one |

**The test was validated, not just executed.** Simulating under a true null:

| Nominal α | Empirical α | Error |
|---|---|---|
| 0.01 | 0.0107 | 0.0007 |
| 0.05 | 0.0471 | 0.0029 |
| 0.10 | 0.1040 | 0.0040 |

Null p-values are uniform (KS p = 0.09), and empirical power matches the
closed-form calculation to **±0.004** at every sample size, one- and two-sided.
The power and MDE figures were additionally reproduced by hand — a
from-scratch noncentral-normal calculation reproduces the package's MDE
(+2.5272% vs +2.5273%) and required sample size (619,975 vs 620,235).

**Power is one-sided, the reported p-value is two-sided.** `power.py` defaults to
`alternative="larger"`, matching the pre-registered hypothesis that treatment
beats control; the p-value is reported two-sided as a more conservative summary.
A one-sided test is the more powerful design for a directional hypothesis, so
the two figures are **not** directly comparable. This is stated wherever both
appear.

**SRM is reported on every candidate population**, not only the one that drives
the decision, so the conclusion can be shown not to depend on the choice:

| Population | n | p | Verdict |
|---|---|---|---|
| raw rows (repeat users counted twice) | 294,478 | 0.8915 | PASS |
| one row per user, all users | 290,584 | 0.9941 | PASS |
| one row per user, conflicts excluded | 286,691 | 0.8445 | **PASS (used)** |
| cleaned analysis population | 286,691 | 0.8445 | PASS |

---

## Architecture

```
experimentlens/
├── src/experimentlens/
│   ├── config.py            paths, seed, column contract, Thresholds
│   ├── data/
│   │   ├── schema.py        SchemaReport — required vs optional columns
│   │   ├── loader.py        raw ingest + immutable RawSnapshot
│   │   ├── duplicates.py    four dispositions + conflict policy
│   │   └── clean.py         auditable CleaningStep ledger
│   ├── stats/
│   │   ├── frequentist.py   two-proportion z-test (pooled + unpooled SE)
│   │   ├── effects.py       lift, Cohen's h, Wald and Katz intervals
│   │   ├── validity.py      SRM, contamination, arm statistics
│   │   ├── power.py         power, MDE, required sample size
│   │   ├── bayesian.py      PyMC Beta-Binomial + closed-form cross-check
│   │   └── decision.py      SHIP / DON'T SHIP / INCONCLUSIVE
│   ├── simulation.py        Type I calibration, power curves
│   └── viz.py               Plotly figures (render only)
├── app/streamlit_app.py     presentation layer, zero statistics
└── tests/                   349 tests
```

---

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev,bayes,app,viz]"
```

**No GPU required.** The Bayesian component is a conjugate Beta-Binomial model
and runs on CPU in seconds.

### Getting the data

The raw Kaggle CSV is **not committed** (redistribution terms). Download it and
save it as `data/raw/ab_data.csv` — `dataset/ab_data.csv` also works.

| Column | Type | Required |
|---|---|---|
| `user_id` | integer | yes |
| `timestamp` | datetime | yes |
| `group` | `control` \| `treatment` | yes |
| `landing_page` | `old_page` \| `new_page` | yes |
| `converted` | 0 \| 1 | yes |
| `country` | string | **optional** — supplementary diagnostics only |

`country` is registered as optional because this dataset variant does not
include it; the loader never assumes it exists.

### Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Verification

```bash
pytest        # 349 tests
ruff check .
mypy          # strict
```

Key tests, by intent:

- `test_significance_alone_never_ships` — a conclusive p-value with a
  sub-threshold lift cannot produce SHIP at any alpha
- `test_p_value_is_not_a_decision_criterion` — changing only the p-value does not
  change the outcome
- `test_waterfall_running_total_equals_the_analysis_population` — regression
  cover for a chart that once reported 1.15M users from a 294k dataset
- `test_sampled_probability_matches_closed_form` — the PyMC posterior agrees with
  the exact conjugate answer
- `test_empirical_power_matches_analytic` — simulation agrees with the closed form
- `test_no_streamlit_in_package` — the layer boundary is mechanical, not a comment

---

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

- `ruff check` and `ruff format --check`
- `mypy` (strict)
- `pytest`
- wheel build
- a headless dashboard smoke test that boots Streamlit and waits for a
  response, so a broken import fails CI rather than the demo

---

## Assumptions

Declared in `src/experimentlens/config.py`, not read from UI state:

| Assumption | Value | Rationale |
|---|---|---|
| Estimand | Difference in conversion probability, B vs A | Primary binary outcome |
| Allocation | 50 / 50 | Declared null for the SRM check |
| Significance level | 0.05 (0.001 for SRM) | Low SRM alpha so only gross imbalance is flagged |
| Minimum practical lift | 1% relative | Business threshold, separate from α |
| HDI mass | 0.94 | Posterior reporting interval |
| Prior | Beta(1, 1) flat | Uninformative; washes out at this sample size |
| Revenue per conversion | 25.0 | **Placeholder — replace with a real figure** |
| Seed | 42 | Reproducibility |

---

## Limitations

Stated plainly, because a result that hides its caveats is worth less:

- **Revenue per conversion is a placeholder.** The modelled revenue figures are
  the posterior mean of `(pB − pA) × control users × revenue per conversion` — a
  *modelled* quantity, not observed revenue. They scale linearly with the
  assumption, though the *sign* of the conclusion does not depend on its value.
- **The 1% practical lift is assumed, not derived.** It should come from the
  business's own margin. The dashboard exposes it as a control so the decision
  can be re-run at different thresholds.
- **Power and MDE are one-sided; the reported p-value is two-sided.** See above.
  Both are defensible, but they describe different tests.
- **The frequentist and Bayesian results are not independent evidence.** They are
  summaries of one dataset.
- **The Wald z-test is asymptotic.** With ~143k users per arm it is safe here,
  but it would not be for small counts; a Fisher exact test would be preferable
  there.
- **Post-hoc power is reported but never used to decide.** Observed-effect power
  is a function of the estimate it informs, so letting it gate the outcome would
  be circular.
- **No variance reduction.** CUPED, sequential testing, and uplift modelling
  would all raise power; they are out of scope here.
- **Observational, not randomised.** This is a logged experiment dataset, so
  causal claims rest on the randomisation assumption having held. SRM passing is
  necessary but not sufficient evidence of that.

---

## Licence

MIT — see [LICENSE](LICENSE).