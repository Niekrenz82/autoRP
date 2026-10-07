# Implementation Plan — P&C GLM Pricing with GenAI Decision Layer

## Goal

Build a pure-premium pricing model (frequency x severity, via `glum` GLMs) on the
French MTPL open dataset. Every modeling step (fitting, feature search, metrics)
is deterministic and reproducible via MLflow. GenAI is only used to make
judgment calls over already-computed, structured summaries (e.g. "which
candidate model should be promoted") — never to touch raw data, features, or
coefficients directly.

## Phase 0 — Environment

- [x] `uv` project initialized, `glum` and `scikit-learn` added as dependencies
- [x] Folder scaffolding created (`data/`, `scripts/`, `autorp/`, `configs/candidates/`,
      `notebooks/`, `tests/`)
- [x] Add MLflow: `uv add mlflow`
- [x] Decide tracking URI location (`sqlite:///mlflow.db` at project root) and add
      `mlflow.db` + `mlruns/` to `.gitignore`

## Phase 1 — Data

- [ ] `scripts/fetch_data.py` — pulls `freMTPL2freq` (OpenML id 41214) and
      `freMTPL2sev` (OpenML id 41215) via `sklearn.datasets.fetch_openml`, writes
      to `data/fremtpl2freq.parquet` and `data/fremtpl2sev.parquet`
- [ ] `autorp/data.py` — `load_freq()` / `load_sev()` functions reading the parquet
      files back into pandas DataFrames
- [ ] Sanity-check in a notebook: row counts, exposure distribution, claim count
      distribution, join frequency + severity on policy ID

**Done when:** both parquet files exist in `data/` and load cleanly via `autorp/data.py`.

## Phase 2 — Baseline model

- [x] `autorp/tracking.py` — sets the MLflow tracking URI (absolute path, so notebooks
      don't create a second store), exposes `setup()`, `promote(model_name, version, alias)`,
      `load_model(model_name, alias)`
- [x] Define the baseline GLM spec: Poisson GLM on a small, obviously-relevant
      feature set (e.g. driver age, vehicle power, region), log-exposure offset,
      no interactions, no regularization tuning
- [x] Fit it, log params/metrics (deviance, D², Gini, A/E, lift-decile table) and
      the model itself to MLflow as a pyfunc (`autorp/pricing_model.py`: freq GLM ×
      sev GLM × large-loss loading), `registered_model_name="pnc-pricing-baseline"`
- [x] Tag/alias this registered version as the baseline (e.g. MLflow alias
      `baseline`)

**Done when:** `pnc-pricing-baseline` exists in the MLflow registry with one
version aliased `baseline`, and its metrics are logged.

## Phase 3 — Feature engineering

- [ ] `autorp/features.py` — reusable, deterministic transforms: binning
      continuous variables, encoding categoricals, building the exposure offset
      column
- [ ] Keep transforms as pure functions (DataFrame in, DataFrame out) so any
      candidate spec can reuse the same building blocks

**Done when:** feature functions are unit-testable and used by both the
baseline and candidate fitting code (no duplicated logic).

## Phase 4 — Candidate model search (deterministic)

- [ ] `configs/candidates/*.toml` (or `.yaml`) — one file per candidate spec:
      feature list, distribution, link function, regularization (`alpha`, `l1_ratio`)
- [ ] `autorp/models.py` — reads a candidate config, fits the GLM, logs an MLflow
      run (not yet registered — just tracked)
- [ ] Decide candidate generation approach: start with a handful of hand-written
      configs; consider a simple grid/Optuna search later once the pipeline works
      end-to-end

**Done when:** you can add a new `.toml` file and get a new tracked MLflow run
without touching any other code.

## Phase 5 — Deterministic evaluation

- [ ] `autorp/evaluate.py` — given a candidate run ID, pulls the baseline's
      metrics from MLflow (via alias `baseline`), computes deltas
      (deviance, AIC/BIC, Gini, lift-decile differences)
- [ ] `autorp/types.py` — `ModelMetrics`, `FeatureDiff`, `ModelComparisonInput`,
      `ModelDecision` dataclasses (the structured contract between deterministic
      code and the GenAI layer)

**Done when:** running evaluation on any candidate run ID returns a fully
populated `ModelComparisonInput` with no missing fields.

## Phase 6 — GenAI decision layer

- [ ] `autorp/decide.py` — takes a `ModelComparisonInput`, returns a
      `ModelDecision` (`selected_candidate_id`, `rationale`, `confidence`)
- [ ] Hard-validate the LLM's chosen ID against the actual candidate list before
      acting on it — reject/error on anything else
- [ ] On selection, call `promote()` to alias the winning MLflow model version
      as the new baseline (or as a tagged "candidate-in-review" stage, if you
      want a human approval step before full promotion)

**Done when:** you can run the full loop — fit candidates, evaluate against
baseline, let the GenAI layer pick a winner, promote it — and trace every step
back through MLflow's run history.

## Phase 7 — Iterate

- [ ] Expand candidate configs (more features, interactions, regularization paths)
- [ ] Add severity-side candidates, not just frequency
- [ ] Combine frequency x severity into a pure premium estimate and evaluate that
      combined output the same way (baseline vs. candidate pure premium)
- [ ] Optional: light GenAI narration layer for generating a plain-language model
      documentation summary from the final chosen model's metrics

## Explicit non-goals (kept deterministic, never GenAI)

- Feature engineering choices
- Hyperparameter search / regularization paths
- Any value that flows directly into a priced number
