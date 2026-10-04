# Less reliance on tone: protocol (written before any result)

Written 2026-10-04, before `experiments_tone.py` was first run.

## Problem

The model flags polished, corporate-sounding mail almost regardless of intent:
- 87.6% of English promotions ([`COMMERCIAL.md`](COMMERCIAL.md));
- 15 of 16 benign LLM-written emails in an exploratory audit ([`LLM_PHISHING.md`](LLM_PHISHING.md)).

Generic words push hardest: "your" has a coefficient of +5.4, "you" +3.6 and "email" +3.4.
Character n-grams carry the same signal below the word level.

Public legitimate marketing mail does not exist. The one candidate found
(`marketeam/Marketing-Emails`) turned out to be internal team correspondence with no links. So
this experiment changes the **model**, not the data.

## Candidates

All candidates use the production data, lure features (`feature_version=3`), C=16, grouped
5-fold CV, Platt calibration and the plateau threshold, then a refit on all the data. Only the
text block changes:

| | Word TF-IDF | Character n-grams |
|---|---|---|
| **P0** | as in production | as in production (the saved `model.joblib`) |
| **M1** | English stop words removed (scikit-learn's list) | as in production |
| **M2** | as in production | **removed** |
| **M3** | English stop words removed | **removed** |

## What is measured

| Set | What | Why |
|---|---|---|
| Promotions: English, non-English, all | false alarms | the target; **not blind**, it was used to find the problem |
| Hold-outs A, B, C | recall and false alarms | |
| Zenodo LLM-written phishing (4,986) | recall | a guard |
| The 16 "benign" emails of the exploratory Greco audit | flagged | descriptive, too small to decide |

All comparisons use McNemar tests against P0 on the same emails.

## Decision rule

A candidate **qualifies** only if all hold:

1. False alarms on **English** promotions are significantly lower than P0's (p < 0.05).
2. Recall on Nazario 2025 phishing (the phishing of A and B) is **not** significantly lower.
3. Recall on hold-out C phishing is **not** significantly lower. This differs from the earlier
   commercial-fix protocol: here the text block itself changes, so C recall is a guard.
4. False alarms on the legitimate emails of A, B and C are **not** significantly higher, on
   each set.
5. Recall on Zenodo LLM-written phishing is **not** significantly lower.

If more than one qualifies, the one with the lowest false-alarm rate on English promotions wins.

**Even a winner does not replace production yet.** The commercial set is not blind, so a winner
becomes the **candidate** for the blind test on the research inbox, and production changes
only if it passes there. If none qualifies, the result is reported as is.
