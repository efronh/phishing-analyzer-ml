# Lockbox test protocol

Written on 2026-10-02, after the lockbox data was downloaded and **before any model was evaluated on it**.

## Why

By the end of development, hold-outs A and B had each been measured at every step, from v1 to v4. Hold-out C's errors were inspected after the recall study. Rules were fixed in advance at each step, but looking at the same test sets again and again can still fit them without anyone noticing. The lockbox is one last test set that nobody looked at during development.

## Data (`download_lockbox.sh`)

- **Phishing:** 1,000 Phishing Pot samples, drawn at random with seed 2026 from the 7,814 older samples. Hold-out C used only the newest 800, so none of these were ever loaded. The repository is pinned to commit `49f6377`. As for hold-out C, only samples that the stopword rule in `prepare_data.py` classifies as English are kept. The source is the same as hold-out C, but the emails are different.
- **Legitimate:** 2026 mailing-list archives (January, April, July) from three organizations used nowhere else: ISC `bind-users`, Samba `samba`, and Mailman `mailman-users`.
- Duplicates within the lockbox are removed, and so is every lockbox email that is a near-copy (TF-IDF cosine ≥ 0.8) of a training email. The report states how many were dropped.

## What is measured

The saved production model (`model.joblib` from `train.py`), exactly as shipped:

- recall and false-alarm rate, each with a 95% Wilson confidence interval, plus precision, F1 and ROC-AUC;
- the false-alarm rate for each mailing list;
- the phase-1 rule engine, at the threshold the CLI uses (score ≥ 50), as a baseline.

## Commitments

1. `evaluate_lockbox.py` runs once. It refuses to run again if `reports/LOCKBOX.md` exists, unless `--force` is passed, and any forced rerun has to be disclosed in the README.
2. No model, feature or threshold is changed because of the lockbox result. If something changes after the lockbox has been evaluated, the README must say the lockbox is no longer blind.
3. The result is reported whatever it is, good or bad.
