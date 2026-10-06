# Auto P&C Pricing Priors

Domain priors for motor (auto) risk-premium modelling. Injected into the GenAI
decision layer as grounding; `[enforced]` rules should also become assertions in
code or fields on `ModelComparisonInput`.

These are strong defaults, not laws. A prior that conflicts with the data is a
prompt to investigate, not to override silently. The actuary in the loop always wins.

Sources at the bottom. `[enforced]` = should be machine-checked. `[advisory]` = judgment.

---

## 1. Model structure

**P-01 — Model frequency and severity separately.** Don't default to a single
Tweedie pure-premium model. `[advisory]`
*Why:* frequency and severity have different drivers. Each is individually less
noisy than pure premium, so real effects stay visible instead of being swamped by
severity variance. Tweedie also assumes frequency and severity move in the same
direction, which is often false.

**P-02 — Frequency: Poisson (or negative binomial), log link, `log(exposure)` as an
offset.** `[enforced]`
*Why:* claim counts scale with time on risk. An offset forces a coefficient of
exactly 1 on log exposure. Using exposure as a weight, or as an ordinary
covariate, is wrong and silently distorts every other coefficient.

**P-03 — Severity: Gamma (or inverse Gaussian), log link, fitted on claim records
only, weighted by claim count.** `[enforced]`
*Why:* zero-claim policies carry no severity information. Weighting by claim count
makes an aggregated record of 3 claims count three times.

**P-04 — Pure premium = frequency prediction × severity prediction.** With log links
on both, relativities multiply. `[enforced]`

**P-05 — Model each coverage/peril separately, combine at the end.** `[advisory]`
*Why:* bodily injury, property damage, theft and glass have different drivers and
very different tails. One model across all of them averages away real structure.

**P-06 — Use a multiplicative (log link) structure.** `[advisory]`
*Why:* it matches how rating tables are actually implemented and filed, and keeps
every predicted premium positive.

## 2. Data and target preparation

**P-07 — Every claim must match exactly one policy record.** Count orphans and
duplicates before modelling. A merge must never silently change row counts. `[enforced]`
*Why:* a key matching multiple policy records double-counts claims; a key matching
none orphans them. Both bias frequency directly.

**P-08 — Cap large losses. Never delete them.** `[enforced]`
*Why:* deleting truncates the loss distribution and biases pure premium downward.
Capping keeps the systematic part of severity while removing noise. Set the cap
high enough that genuine severity variation survives, low enough that a handful of
claims don't dominate the fit.

**P-09 — Reinstate the capped excess as a separate loading across the portfolio.** `[enforced]`
*Why:* the excess is real expected cost. Cap and stop, and the rate is deficient by
exactly the excess ratio.

**P-10 — Exposure is earned vehicle-years. Check for `exposure > 1` and
`exposure <= 0` before fitting.** `[enforced]`

**P-11 — Replace impossible values with the base level and add an `is_imputed`
flag; don't drop the rows.** `[advisory]`
*Why:* errors are usually systematic (one branch miscoding). Dropping the rows
removes the only evidence the problem exists.

## 3. What to expect from auto rating factors

**P-12 — Driver age is among the strongest factors and is U-shaped:** very high at
18–25, falling to a minimum around 50–65, rising again at older ages. `[advisory]`
*Why:* a fitted age curve that comes out flat or monotone signals a specification
problem or confounding — not a discovery.

**P-13 — Age is confounded with experience, bonus-malus and vehicle choice.** Don't
read a raw age coefficient as a pure age effect. `[advisory]`

**P-14 — Prior claims history / bonus-malus is usually the single strongest
frequency predictor.** `[advisory]`
*Why:* it is the closest thing to a direct observation of that policyholder's own risk.

**P-15 — Territory is strong for both frequency and severity, and is
high-dimensional.** Smooth or cluster it; don't use raw postcode dummies. `[advisory]`
*Why:* traffic density, theft, repair cost and litigation all vary geographically,
but raw location dummies overfit badly.

**P-16 — Mileage and vehicle use predict frequency strongly, severity weakly.** `[advisory]`

**P-17 — Vehicle characteristics (power, weight, value, age) drive severity far more
than frequency.** `[advisory]`

**P-18 — Frequency and severity respond to different variables.** A variable that
matters a lot for one and not at all for the other is normal, not a bug. `[advisory]`

## 4. Credibility and smoothness

**P-19 — Full credibility standard is 1,082 claims** (90% confidence of landing
within 5% of the true rate). `[enforced]`

**P-20 — Partial credibility is `Z = sqrt(n / 1082)`.** 270 claims gives Z = 0.5. `[enforced]`

**P-21 — A level with very few claims should be grouped or credibility-weighted
toward the portfolio mean,** not left standing as its own level. `[advisory]`

**P-22 — Relativities across an ordered variable should be smooth.** A large jump
between adjacent bands is almost always low volume or overfit, not a real cliff. `[advisory]`

## 5. Validation

**P-23 — Split the data before anything else** — 60/40 or 70/30 train/test, or
40/30/30 train/validation/test — and keep the split fixed throughout. `[enforced]`

**P-24 — Prefer an out-of-time split over a random one.** `[advisory]`
*Why:* a random split can put the same event in both sets, so the test set is not
truly unseen and validation comes out over-optimistic.

**P-25 — Measure all lift and actual-vs-predicted on holdout data only.** `[enforced]`

**P-26 — Use the test set sparingly.** Once many model choices have been made
against it, it has become a training set. `[advisory]`

**P-27 — Judge a quantile/decile plot on three things:** (1) predicted tracks actual
within each bucket, (2) *actual* increases monotonically across buckets — small
reversals are fine, (3) the spread between first and last bucket is wide. `[enforced]`
*Why:* a wider first-to-last spread means the model separates best from worst risks
more sharply. A published example: current plan spread 0.55→1.30 vs. new model
0.40→1.60, so the new model wins on criterion 3.

**P-28 — To compare two models head to head, use a double lift chart:** sort by
(model A prediction ÷ model B prediction), bucket, and see which tracks actual more
closely in the extreme buckets. `[advisory]`
*Why:* the extreme buckets are exactly where the two models disagree.

**P-29 — Gini and AUROC are not independent evidence.**
`AUROC = 0.5 × normalised Gini + 0.5`. Never count a gain in both as two reasons. `[enforced]`

**P-30 — Every parameter is a degree of freedom, and more degrees of freedom always
improve training fit.** A training-set improvement is not evidence. `[enforced]`

**P-31 — There is no magic p-value cutoff.** Significance is one input alongside
business rationale, volume and stability. `[advisory]`

**P-32 — Never include two variables carrying near-identical information.** `[enforced]`
*Why:* aliasing. The fit breaks or splits the effect arbitrarily between them.

**P-33 — Cross-validation is of limited use when variables are hand-selected,**
because selection has already seen all the data. Use a fixed holdout instead. `[advisory]`

**P-34 — Once a model is finally chosen, refit it on all the data.** `[advisory]`

## 6. Regulatory

**P-35 — Gender is banned as a rating factor in the EU** (Test-Achats, effective
2012-12-21) **and in CA, HI, MA, MT, PA, NC, and parts of MI.** Never let an
automated search select it without a jurisdiction check. `[enforced]`

**P-36 — Credit-based insurance scores are banned in CA, HI, MA, MI** and
restricted in MD, OR, UT. `[enforced]`

**P-37 — Watch for proxy discrimination:** a permitted variable can reconstruct a
banned one. `[advisory]`

**P-38 — A factor must be explainable and filable, not merely predictive.** `[advisory]`

---

## Sources

- Goldburd, Khare, Tevet & Guller, *Generalized Linear Models for Insurance Rating*,
  CAS Monograph No. 5, 2nd ed. (2025) — §4.1–4.3, §5.1–5.3, §7.1–7.3.
  Basis for P-01 to P-11, P-23 to P-34.
- CAS / American Academy of Actuaries credibility material — classical full
  credibility standard and the square-root rule. Basis for P-19 to P-21.
- Court of Justice of the EU, *Test-Achats* (C-236/09), gender ruling effective
  2012-12-21; NAIC and state DOI material on credit-based insurance scores.
  Basis for P-35 to P-37.
