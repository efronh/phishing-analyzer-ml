# Fixing the commercial false alarms: protocol (written before any result)

Written 2026-10-04, before `experiments_commercial_fix.py` was first run.

## The problem

The production model flags 87.6% of English promotional emails (and 94.7% of the others) as
phishing ([`COMMERCIAL.md`](COMMERCIAL.md)). The exploratory breakdown points to two causes:

1. **Kaggle's "phishing" class is mostly bulk spam,** so the model learned marketing language as
   a phishing signal.
2. **No legitimate training email is marketing,** so it never saw the opposite.

## Candidates

All candidates use the production procedure and hyperparameters: lure features
(`feature_version=3`), grouped 5-fold CV, Platt calibration, plateau threshold, then a refit on
all of the candidate's training data.

| | Training data | Tests cause | Eligible for production |
|---|---|---|---|
| **P0** | production model as saved (`model.joblib`) | – | (baseline) |
| **P1** | Nazario 2019–2024 phishing, SpamAssassin ham, 2024 mailing lists: **no Kaggle at all** | 1 | yes |
| **P2** | P1 + **Kaggle's legitimate emails only** (its "phishing" class removed) | 1 | yes |
| **P3** | production data + the author's promotional emails | 2 | **no**, diagnostic only |

**Why P3 can never become production.** Its training data is private, so a model trained on it
could not be rebuilt from this repository. It answers one question only: does seeing legitimate
marketing fix the problem?

**How P3 is tested.** Promotional emails are split into 5 folds **by sender domain**. Each
email is scored by a model trained on production data plus the promotions of the *other*
senders, so the model has never seen that sender. Hold-outs A, B and C are scored by one more
model trained on production data plus all promotions.

## What is measured

- **Commercial false alarms:** English, non-English and all, with 95% Wilson intervals. The
  set is the same 674 emails as in `COMMERCIAL.md`, with the same masking, junk filter and
  sender cap. McNemar test against P0 on the same emails.
- **Hold-outs A, B, C:** recall, false alarms and F1. McNemar against P0 on the phishing emails
  (recall) and on the legitimate emails (false alarms) separately.

## Decision rule

Only P1 and P2 are eligible. A candidate **qualifies** only if all of the following hold:

1. Its false-alarm rate on **English** promotions is significantly lower than P0's (McNemar
   p < 0.05).
2. Recall on the Nazario 2025 phishing (the 454 phishing emails of A and B) is **not**
   significantly lower than P0's (McNemar p < 0.05 with more misses).
3. False alarms on the legitimate emails of A, B and C are **not** significantly higher than
   P0's (McNemar p < 0.05, on each set).

If both qualify, the one with the lower false-alarm rate on **all** promotions wins. If neither
qualifies, production stays as it is.

**Hold-out C recall is reported but does not decide.** Phishing Pot labels are partly noisy:
some samples are plain marketing spam. A model that stops flagging marketing is expected to
"miss" some of them.

## Caveats, fixed in advance

- **The commercial set is no longer blind.** It was used to find the problem. A candidate that
  wins here still needs a fresh commercial test before the README can call the problem fixed.
- **One inbox, 45 senders, mostly Turkish.** The English subset (169 emails) is the fairer
  measure, because every candidate is trained on English only, apart from P3.
- **P3's languages:** P3 adds mostly Turkish promotions to training, so it may simply learn
  "Turkish = legitimate". That is why its English rate on unseen senders matters most.
