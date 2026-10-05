# Decision policy under cost and prevalence: protocol (written before any result)

Written 2026-10-05, before `experiments_decision.py` existed. The hold-outs have been scored
many times, by this model and earlier ones, but no cost- or prevalence-based threshold has ever
been evaluated on them.

## Question

The calibrated probability assumes the training mix: **37.27% phishing**. A real inbox has
0.1–5%, and a missed phishing email costs more than a false alarm. The production threshold
(0.519) maximizes F1 on the training mix and ignores both.

1. How should the probability be re-read at another phishing share, and which threshold then
   minimizes expected cost?
2. Does that re-reading, derived from training data only, hold on the hold-outs, whose emails
   come from other sources?
3. How much does an analyst review band buy, and at what review load?

## Costs and shares (the grid)

- **Cost ratio** `r = C_FN / C_FP` ∈ {1, 10, 100, 1000}. `C_FP = 1`: the unit is one false alarm.
- **Phishing share** `π` ∈ {0.001, 0.01, 0.05}, the same shares as the precision table in the README.
- **Primary cell: `r = 100`, `π = 0.01`.** The other 11 cells are descriptive.

Expected cost per email, from the class-conditional rates measured on a set:

```
EC(t) = π · r · FNR(t)  +  (1 − π) · FPR(t)
```

Reported per 1,000 emails, next to the best trivial policy, `min(π·r, 1 − π)` (flag nothing or
flag everything).

## Policies (all fixed from training data only)

- **P0, production:** `t = 0.519` in every cell.
- **P1, prior shift (the candidate).** Re-read the probability at share `π`, then flag when the
  expected cost of passing exceeds that of flagging:

  ```
  k     = [π / (1 − π)] / [π_t / (1 − π_t)],   π_t = 0.3727
  p_d   = odds⁻¹( k · odds(p) )                 probability at share π
  flag  ⇔ p_d > 1 / (1 + r)   ⇔   p > 1 / (1 + r·k)
  ```

  No free parameter, so its thresholds are known before any data is touched:

  | `r` \ `π` | 0.001 | 0.01 | 0.05 |
  |---|---:|---:|---:|
  | 1 | 0.998 | 0.983 | 0.919 |
  | 10 | 0.983 | 0.855 | 0.530 |
  | **100** | 0.856 | **0.370** | 0.101 |
  | 1000 | 0.373 | 0.056 | 0.011 |

- **P2, empirical (diagnostic).** On the training out-of-fold scores, the threshold that
  minimizes `EC(t)` directly. It needs the class-wise score distributions to carry over, but not
  the probability's level. If the Platt calibration is right in training, P1 ≈ P2.
- **Oracle (reference only).** The threshold that minimizes `EC(t)` on the hold-out itself.
  Fitted on the test set, so optimistic by construction.

P2 and the oracle search 2,001 thresholds evenly spaced in logit between 10⁻⁴ and 1 − 10⁻⁴.
The cost is a step function, so the minimum is often a range: take the middle (in logit) of the
lowest-cost run of grid points, and the longest run if several tie.

## Review band (abstention)

A third action, **review**: an analyst looks at the email, at cost `c` per email (in false-alarm
units), and is assumed to be always right. With the re-read probability `p_d`, take the action
with the lowest expected cost (Chow's rule):

```
pass   : r · p_d        flag : 1 − p_d        review : c
review band:  c / r  <  p_d  <  1 − c      (exists only if c < r / (1 + r))
```

`EC` then adds `c` × the review rate, and the review rate at share `π` is
`π · P(review | phishing) + (1 − π) · P(review | legitimate)`.

- **Named values:** `c` ∈ {0.05, 0.1, 0.25, 0.5}. **Primary: `c = 0.25`.** This is an
  assumption, not a measurement: a review is taken to cost a quarter of a false alarm.
- **Risk–coverage curve:** 50 values of `c` from 0 to `r/(1+r)`, at the primary cell. For each:
  coverage (1 − review rate), cost on the auto-decided emails, and total cost.

## Data

- **Training out-of-fold scores:** the production pipeline and training set, grouped 5-fold CV,
  seed 42, Platt-calibrated, exactly as in `fit_with_cv_threshold`. **Sanity check:** the
  plateau rule on these scores must give back 0.519. If it does not, stop and investigate.
- **Hold-outs,** scored by the saved `model.joblib`, unchanged:
  - **A:** Nazario 2025 (454) + Apache (1,478);
  - **B:** the same 454 + Ubuntu (917);
  - **C:** Phishing Pot (214) + OSGeo (767).
- **Not used:** the lockbox (spent; only its confusion counts at 0.519 were saved).
- **Private, descriptive only:** the author's 674 promotions (169 English), same filters as
  `COMMERCIAL.md`. Share flagged under P0 and P1 in every cell, and share sent to review
  with `c = 0.25`.
  Not in `run_all.sh`.

## What is reported

For every set, cell and policy: threshold, FNR, FPR, the implied precision at share `π` (as in
the README's precision table), and `EC` per 1,000 emails, with 95% intervals
from a stratified bootstrap (phishing and legitimate resampled separately, 2,000 resamples,
seed 0). Differences between policies use the same resamples (paired).

Also:
- **Calibration transfer:** on each hold-out, ECE and a 10-bin reliability table after
  re-reading the probability at that hold-out's own share (A 23.5%, B 33.1%, C 21.8%), and,
  for comparison, ECE without re-reading. The training out-of-fold ECE is 0.0027.
- **Regret:** `EC(P1) − EC(oracle)` and `EC(P0) − EC(oracle)` in every cell.
- **The review band** at the primary cell: the four named values of `c` as a table, and the
  risk–coverage curve.

## Decision rule

**P1 qualifies** if, at the primary cell (`r = 100`, `π = 0.01`):

1. its expected cost is lower than P0's on at least two of A, B and C (point estimates); and
2. on none of A, B and C is it significantly higher (the paired 95% interval of
   `EC(P1) − EC(P0)` lies entirely above 0).

**The review band qualifies** if P1 qualifies and, at the primary cell with `c = 0.25`, the same
two conditions hold for `EC(P1 + review)` against `EC(P1)`.

**If they qualify:** the CLI gets opt-in flags (`--prevalence`, `--cost-ratio`, and
`--review-cost` for the three-way verdict). **The default stays 0.519,** so every number
reported so far keeps describing the CLI's default decision.

**If not:** nothing ships, and the result is reported as is.

All other cells, P2, the oracle, the regret and the commercial set are descriptive and do not
decide.

## Caveats, fixed in advance

- **The prior-shift formula assumes only the share changes.** It is exact only if phishing and
  legitimate emails each look the same in deployment as in training. The hold-outs come from
  other sources, so they test exactly this assumption, and a failure is a finding, not a bug.
- **The hold-outs are not blind.** All three were scored at 0.519 many times. No threshold here
  is fitted on them, and P1 has no free parameter, but their score distributions were seen.
- **Few errors.** At high thresholds the false-alarm rate rests on a handful of emails, and at
  `π = 0.001` the cost is almost all false alarms. Expect wide intervals there.
- **The tails of the probability are a 2-parameter sigmoid.** Thresholds such as 0.998 or 0.011
  rest on very few training emails.
- **Costs are assumptions, not measurements.** That is why they are a grid. The reviewer is
  assumed to be perfect.
- **A and B share their 454 phishing emails,** so they are not independent.
- **Some of C's "phishing" is marketing spam,** so its FNR is inflated, more so at high
  thresholds.

## Predictions

None. The author had no basis for them yet (no hands-on experience with mail triage), so none
were written down rather than guessed after the fact.
