# Phishing Email Analyzer: Phase 2 (Machine Learning)

[Phase 1](https://github.com/efronh/phising_mail_analayzer) was a rule-based phishing detector: keyword, URL, IP, domain, attachment and sender heuristics, using only the standard library. Phase 2 adds a machine-learning classifier on top of it. It trains the classifier on public datasets and measures it honestly, including on real phishing from sources the model never saw during training.

## Final test: emails the model never saw

The shipped model was tested as-is, with no re-tuning, on three hold-out sets. None of their sources appear in training:

| Set | Phishing | Legitimate |
|---|---|---|
| A | 454 real phishing from 2025 (Nazario). Only 4 are near-copies of training emails. | 1,478 from 8 Apache mailing lists, 2025 |
| B | the same 454 | 917 from Ubuntu mailing lists, 2025 |
| C | 214 English phishing from Phishing Pot, a honeypot collection (2026) | 767 from OSGeo mailing lists, 2025 |

Brackets show 95% confidence intervals.

| | Set | Recall | False alarms | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Phase 1: rule engine | A | 30.0% (25.9–34.3) | 14.1% (12.4–15.9) | 0.34 | 0.64 |
| | B | 30.0% | 14.0% (11.9–16.4) | 0.38 | 0.61 |
| | C | 27.1% (21.6–33.4) | 31.3% (28.1–34.7) | 0.23 | 0.46 |
| First ML version (v1) | A | 97.6% | 8.6% | 0.87 | 0.98 |
| | B | 97.6% | 24.9% | 0.79 | 0.94 |
| **Current model** | A | **99.1%** (97.8–99.7) | **1.5%** (1.0–2.2) | **0.97** | 0.998 |
| | B | **99.1%** (97.8–99.7) | **1.4%** (0.8–2.4) | **0.98** | 0.998 |
| | C | **93.0%** (88.8–95.7) | **0.4%** (0.1–1.1) | **0.96** | 0.998 |

- **Phishing caught: 30% (rules) → 99% (ML)** on 2025 Nazario phishing, and 93% on Phishing Pot.
- **False alarms: 8.6–24.9% (v1) → 0.4–1.5%.**
- **Stable across retraining:** over 5 training seeds, recall on 2025 Nazario phishing stays within 98.9–99.1% and false alarms on set A within 1.6–2.3% (see [below](#a-stable-threshold)).
- **What it still gets wrong:** the remaining false alarms are mostly one-line "Unsubscribe" emails that users send to a list by mistake, plus a few project announcements. The missed phishing is mostly short, link-free lures and phishing disguised as a product newsletter. Some Phishing Pot samples are plain marketing spam, so its labels are partly noisy.
- **But not on commercial mail:** on the author's own promotional emails, **88% of English promotions are flagged as phishing** (see [below](#commercial-mail-the-blind-spot)). Every legitimate test set above is a mailing list, so these false-alarm rates hold for that kind of mail only.
- **The lockbox test confirms it** (see [below](#lockbox-test-one-time-blind-evaluation)): 96.9% recall at 0.5% false alarms on emails and organizations never used during development.

These are the CLI's own decisions (see [How it works](#how-it-works)). Details by source and example errors: [`reports/HOLDOUT_2025.md`](reports/HOLDOUT_2025.md).

## Lockbox test: one-time blind evaluation

By the end of development every hold-out had been looked at at least once. So one more set was put aside. Its rules ([protocol](reports/LOCKBOX_PROTOCOL.md)) were written down before it was used. Then the finished model was evaluated on it **exactly once**, and nothing changed afterwards.

- **Phishing:** 457 English emails from 1,000 random *older* Phishing Pot samples. Hold-out C used only the newest 800, so none of these were ever loaded during development.
- **Legitimate:** 562 emails from the 2026 mailing lists of three organizations used nowhere else: ISC (`bind-users`), Samba and Mailman (`mailman-users`).
- None of these emails is a near-copy of a training email.

| | Recall | False alarms | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Phase 1: rule engine | 42.2% (37.8–46.8) | 7.8% (5.9–10.3) | 0.56 | 0.80 |
| **Current model** | **96.9%** (94.9–98.2) | **0.5%** (0.2–1.6) | **0.98** | **0.999** |

The errors match the patterns seen on the hold-outs:

- **3 false alarms in 562 emails.**
- **The missed phishing** is mostly crypto-wallet "desktop app" impersonations (CoinTracker, Rabby, Lido) and very short link-free messages.
- **Label noise:** one "missed phishing" sample is actually an ordinary mailing-list reply.

Full report: [`reports/LOCKBOX.md`](reports/LOCKBOX.md).

**What these numbers mean in a real inbox.** The test sets are 22–33% phishing; a real inbox, after the provider's filters, is usually well under 1%. Recall and the false-alarm rate don't depend on that share, but precision (the share of alerts that are real) does. The table shows the precision implied by the measured recall and false-alarm rate, at the same threshold, for different phishing shares:

| Set | Precision on the test set | 0.1% phishing | 1% phishing | 5% phishing |
|---|---:|---:|---:|---:|
| A | 95.3% | 6.2% (4.3–9.3) | 40.2% (31.2–50.9) | 77.8% (70.3–84.4) |
| B | 97.2% | 6.5% (4.1–11.1) | 41.4% (29.9–55.8) | 78.6% (69.0–86.8) |
| C | 98.5% | 19.2% (8.2–45.5) | 70.6% (47.3–89.4) | 92.6% (82.4–97.8) |
| Lockbox | 99.3% | 15.4% (6.4–38.9) | 64.7% (40.8–86.5) | 90.5% (78.2–97.1) |

- At 1% phishing, **only about 2 alerts in 5 are real on sets A and B.** Precision of 95–99% on the test sets says almost nothing about a real inbox.
- The intervals are wide because the false-alarm rates rest on only 3–22 errors.
- PR-AUC drops the same way: 0.994–0.998 on the test mix, but 0.86–0.93 once the legitimate emails are re-weighted to 0.1% phishing. Details: [`reports/HOLDOUT_2025.md`](reports/HOLDOUT_2025.md#precision-in-a-real-inbox).
- The lockbox row comes from its saved confusion counts. The lockbox model was not run again.
- **The calibrated probability has the same limit:** it assumes 37% phishing, as in training.

This text-only model would be one layer of a mail filter, not the whole filter.

## Commercial mail: the blind spot

Every legitimate test email above comes from open-source mailing lists. The first test on commercial mail shows that the low false-alarm rate does not carry over.

**Data.** The set is 674 promotional emails (newsletters and campaigns) from the author's own inbox.
- 45 sender domains, at most 30 emails per sender; 169 of the emails are English.
- **Private:** the emails are not published, so this result cannot be reproduced from the repository.
- **Masking:** before scoring, the author's name and addresses, every recipient address (in plain, URL-encoded and base64 form), and phone and card numbers were removed.
- **Report:** it holds counts only, with no subjects, senders or bodies.

The rules were committed before the run ([protocol](reports/COMMERCIAL_PROTOCOL.md)).

| | Emails | Flagged as phishing (every one is a false alarm) |
|---|---:|---:|
| English promotions | 169 | **87.6%** (81.8–91.7) |
| Other languages (mostly Turkish) | 505 | 94.7% (92.3–96.3) |
| For comparison: hold-out A's mailing lists | 1,478 | 1.5% (1.0–2.2) |

- **Why (exploratory, found after the result):** the text features put promotional mail on the phishing side.
  - The mean word + character contribution is **+2.4** for English promotions, against −7.4 for the 2024 mailing lists in training and +5.5 for Kaggle "phishing".
  - The words that push hardest are marketing language: "your", "email", "unsubscribe", "this email".
  - The cause is mainly the **missing marketing on the legitimate side**: no legitimate training email is marketing. The first guess, that Kaggle's spam-heavy "phishing" class ([finding 2](#key-findings)) taught it, explains little: removing it barely helps (see below), while adding legitimate promotions does.
- **Not a language effect:** English promotions are flagged almost as often as Turkish ones.
- **The URL features don't fix it:** 91.8% instead of 92.9% (McNemar p = 0.17).
- **HTML signals, measured on the same mail:** none passes the pre-registered rule (at least 5% of training phishing and at least 3× the rate on promotions).
  - Hidden text, for example, is in 75% of promotions (newsletter preview lines) and 7% of phishing.
  - Only free-hosting links pass: 19% of phishing, 0% of promotions.
- **Limits:**
  - One inbox, 45 senders, mostly Turkish brands.
  - The comparison phishing is older (2019–2024).
  - "Legitimate" relies on the Promotions category plus a junk filter (iCloud routed the email to the inbox and it passes DMARC).

**What it means:** as it stands, the model behaves more like a bulk-mail detector than a phishing detector.

Full tables: [`reports/COMMERCIAL.md`](reports/COMMERCIAL.md).

### First attempt to fix it

Three candidates against production, with the decision rule committed first ([protocol](reports/COMMERCIAL_FIX_PROTOCOL.md)). Only P1 and P2 use public data and could replace production. P3 is a diagnostic: its promotions are scored by models that never saw their sender (5 folds by sender domain).

| | English promotions | All promotions | A false alarms | 2025 recall | C recall |
|---|---:|---:|---:|---:|---:|
| **Production** | 87.6% | 92.9% | 1.5% | 99.1% | 93.0% |
| P1: no Kaggle | 84.6% | 76.1% | 2.6% | 99.3% | 66.4% |
| P2: only Kaggle's legitimate emails | 82.2% | 76.7% | 3.6% | 98.9% | 75.2% |
| P3: + private promotions in training | **37.3%** | **15.0%** | 0.9% | 97.1% | 85.0% |

- **Removing Kaggle's spam does not fix it.**
  - English promotions barely move: 84.6% (p = 0.23) and 82.2% (p = 0.01).
  - False alarms on the hold-outs rise significantly (A: 1.5% → 2.6–3.6%).
  - Phishing Pot recall collapses to 66–75%.
  - Neither candidate qualifies, so **production stays as it is.**
- **Seeing legitimate marketing does most of the work.** With the private promotions in training, English promotions from unseen senders drop from 87.6% to 37.3%.
  - It costs recall: 9 more of the 454 phishing emails from 2025 are missed (p = 0.004). Phishing and marketing look alike.
- **37% is still far too high.** Most of the training promotions were Turkish; the non-English rate falls to 7.5%, partly because the model can learn "Turkish = legitimate". The fix needs **English** legitimate marketing mail at scale, which no public dataset provides.

Full tables: [`reports/COMMERCIAL_FIX.md`](reports/COMMERCIAL_FIX.md).

## Sender authentication: SPF, DKIM, DMARC (measurement only)

These headers record whether a mail really comes from the domain it shows. They are **not model features**: the mailing-list archives used as legitimate training mail strip them (0% present), so any header feature would learn the source. The measurement compares training phishing (Nazario 2019–2024, 1,571 emails) with the author's private promotions (676 emails, without the DMARC part of the junk filter). Phishing Pot and the lockbox are kept back for a later blind test ([protocol](reports/HEADERS_PROTOCOL.md)).

| Signal | Phishing | Promotions |
|---|---:|---:|
| DMARC recorded, not `pass` | 58.2% | 0.6% |
| DKIM not aligned with the `From` domain | 52.4% | 4.6% |
| No DKIM signature | 40.6% | 0.0% |
| `Reply-To` on another domain | 7.1% | 0.0% |
| Sender on a freemail domain | 2.5% | 4.4% |

- **Strong and complementary to the text.** Eight of the nine signals pass the pre-registered rule. Freemail is the exception.
- **The receiver-independent checks are the trustworthy ones.** DKIM alignment and `Reply-To` are computed from the headers themselves. The recorded SPF/DKIM/DMARC results were written by different servers (iCloud for the promotions, the collector or a forwarder for Nazario). "No results at all" (26.6% of phishing, 0% of promotions) is a collection artifact, not a phishing trait.
- **They cannot catch everything** (exploratory, after the result). 95.0% of promotions come from a **verified sender** (DMARC pass and aligned DKIM), and so does **27.0% of phishing**: phishers authenticate their own lookalike domains.
- **What this suggests:** a second stage for verified senders could remove most commercial false alarms. Suppressing alerts for them outright would cost up to a quarter of phishing recall. The design needs its own protocol and a blind test on Phishing Pot.

Full tables: [`reports/HEADERS.md`](reports/HEADERS.md).

### A second stage for verified senders does not help

The pre-registered design ([protocol](reports/SENDER_STAGE_PROTOCOL.md)) works like this. If the sender is verified and not suspicious (no brand on a foreign domain, no punycode, no free-hosting link, no `Reply-To` elsewhere), the model needs a higher score `t_v` to flag the email. `t_v` comes from training out-of-fold scores.

- **The rule picked `t_v` = the production threshold.** More than 5% of verified training phishing already scores below it, so the stage changes nothing and does not qualify. Nothing ships.
- **No other bar would work either.** The sensitivity table below is descriptive and does not decide:

| `t_v` | English promotions flagged | Nazario 2025 recall | Hold-out C recall |
|---|---:|---:|---:|
| production (0.519) | 87.6% | 99.1% | 93.0% |
| 0.99 | 43.8% | 90.3% | 87.4% |
| 0.999 | 29.6% | 84.1% | 85.0% |
| never alert for verified senders | 8.9% | 55.1% | 75.7% |

- **Verified-sender phishing is common in recent mail.** 58.6% of Nazario 2025 phishing comes from a verified sender, against 27.0% in Nazario 2019–2024 (same collector) and 27.6% in Phishing Pot. Authentication says who sent a mail, not whether it is honest.
- **So the fix has to come from the text model:** legitimate English marketing mail in training.

Full tables: [`reports/SENDER_STAGE.md`](reports/SENDER_STAGE.md).

## How false alarms were cut by 25×

The first ML version flagged 25% of Apache release announcements and **half of Ubuntu security notices** as phishing.

**1. Diagnosis.** The problem was a single feature, not the text. On the false alarms, the text features actually voted "legitimate" (−2.4 on average). The rule features voted "phishing" (+5.7), and almost all of that came from one raw count, `url_count`. A linear model extrapolates without limit: an announcement with 30 links scored far beyond anything seen in training. Real phishing has a median of **1** link.

**2. Bounded features.** Raw totals were replaced with ratios, maximums and yes/no flags: "share of links that look suspicious", "worst link's score", "contains an IP link". Counts now use a log scale, and every scaled feature is clipped to ±3.

**3. Modern legitimate training data.** 2,205 emails from 2024 Python, Fedora and GNU mailing lists were added: release announcements, update notices and Q&A. Apache lists were left out on purpose, because Apache release announcements are also posted to a list in hold-out A.

| Version | Hold-out B false alarms *(blind)* | Hold-out A false alarms | Recall (2025 Nazario) |
|---|---:|---:|---:|
| v1: raw-count features | 25.7% | 7.0% | 85.7% |
| v2: bounded features | 15.2% | 10.0% | 97.4% |
| v2 + threshold re-tuned with modern data | 15.2% | 10.0% | 97.4% |
| **v3: bounded features + modern training data** | **0.9%** | **1.2%** | **96.5%** |

These are the current re-runs of the experiment, after all the fixes below. Every split keeps near-duplicates together, and Kaggle no longer contains SpamAssassin copies.

- **The gain from modern data is real, not noise.** v3 vs v2 on the same emails (McNemar test): 132 emails that only v3 gets right against 5 that only v2 gets right, p < 0.001.
- **Bounded features alone are not enough.** They cut false alarms on B (p < 0.001) but not on A (p = 0.62).
- **Re-tuning the threshold does not help at all.** At the max-F1 point it picks the same threshold. The model has to see this kind of email in training.
- **A validation false-alarm rate does not carry over to a new source.** v1 had at most 1% false alarms on validation and 29.7% on hold-out B.
- **No regression on the original data.** In-domain test F1 is 0.982 for v1 and 0.981 for v3.

Full tables: [`reports/FALSE_ALARMS.md`](reports/FALSE_ALARMS.md).

## Getting recall back

Cutting false alarms first cost recall: 97.6% → 94.9% on 2025 phishing. Of the 23 phishing emails missed at that point:

- 57% had **no link**.
- 43% were **very short**.
- 15 used a **payment pretext**: invoice, ACH remittance, RFQ, order confirmation.

Real phishing of this kind was scarce in training.

Two fixes were compared on **hold-out C**, built for this step from two sources used nowhere else. The production version is the one with the best **validation** F1, a rule fixed before any set-C result was seen.

| Version | Change | C recall | C false alarms | 2025 Nazario recall | Validation F1 |
|---|---|---:|---:|---:|---:|
| v3 | – | 91.1% | 0.5% | 97.6% | 0.9830 |
| v3+lure | + lure features: payment / attachment pretext, generic greeting, urgency, credential request, short body | 87.4% | 0.3% | 97.4% | 0.9830 |
| v4 | + 705 real phishing emails from Nazario 2019–2022 | 89.3% | 0.3% | 97.4% | 0.9851 |
| **v4+lure** | both | **91.1%** | 0.4% | **98.5%** | **0.9855** ✔ |

- **The rule picked v4+lure, and we followed it.** Its lead over v4 is 0.0004 validation F1, about 1 email in 3,228. McNemar finds no significant difference between any two versions on any hold-out (p ≥ 0.14), apart from v3+lure vs v3 on set A (p = 0.017).
- **The honest summary:** the four versions are statistically tied on new data. More real phishing and lure features are both cheap and sensible, but neither one clearly wins here.
- **Risk of the lure features:** they could raise false alarms on real business email (actual invoices, actual attachments), which none of the legitimate test sets cover.

Full tables: [`reports/RECALL.md`](reports/RECALL.md).

## URL features: a tie, so the model stays as it is

TF-IDF replaces every link with one token, so the model sees what a link looks like only through the rule features. Eight bounded URL features were tested:

- a link on free hosting (IPFS, r2.dev, web.app and so on)
- a redirect parameter that contains another URL
- percent-encoding
- hyphens in the host
- subdomain depth
- a brand name on someone else's domain
- punycode
- the number of distinct domains

They were chosen from training sources only. The decision rule was committed before the first run ([protocol](reports/URL_FEATURES_PROTOCOL.md)).

| | Fold F1 (grouped 5-fold CV) | A false alarms | B false alarms | C recall |
|---|---:|---:|---:|---:|
| **production model (v4+lure)** | 0.9846 | 1.5% | 1.4% | 93.0% |
| + URL features | 0.9845 | 1.4% | 1.4% | 93.5% |

- **The rule kept the production model.** The mean paired fold-F1 change is −0.0002, with a standard error of 0.0004. No hold-out difference is significant (McNemar p ≥ 0.69).
- **Why there is no gain:**
  - Most of the phishing the model still misses has no link at all.
  - The rule features already cover the strongest link signals: IP links, shorteners, `@` in a URL, suspicious TLDs.
- **The signals are real, but rare in this data:** 32% of Nazario phishing with a link uses free hosting, against 0.4–1.8% of legitimate mail. That pattern is already learned, through text and rule features, well enough.
- **Commercial mail:** the URL features were also tested on promotional mail ([below](#commercial-mail-the-blind-spot)). They lower the false-alarm rate there only from 92.9% to 91.8%; the problem is the text, not the links.

Full tables: [`reports/URL_FEATURES.md`](reports/URL_FEATURES.md).

## A stable threshold

A single 15% validation split gave a threshold that moved a lot from one training seed to the next.

1. **Cross-validation (cv-max).** 5-fold CV produces one F1-vs-threshold curve per fold. Take the peak of their mean, then retrain on all the data.
2. **Plateau center (cv-plateau, the production rule).** The top of the mean F1 curve is nearly flat, so its exact peak drifts with the data. Instead, take every threshold next to the peak whose mean F1 is within **one standard error** of it (the "one-standard-error rule"), and use the middle of that range.

As in production, the folds keep near-duplicate groups together and the scores are calibrated first. The criteria were fixed before running: a smaller threshold spread, and no larger spread in the hold-out results.

| Over 5 seeds | Single split | CV, peak | **CV, plateau center** |
|---|---:|---:|---:|
| Threshold std | 0.058 | 0.038 | **0.019** |
| False alarms A: std / mean | 0.61 / 2.2% | 0.25 / 1.8% | 0.27 / 1.9% |
| False alarms B: std / mean | 1.07 / 1.5% | 0.37 / 1.4% | **0.33** / 1.6% |
| Recall C: std / mean | 1.93 / 93.6% | 1.00 / 93.6% | **0.76** / 94.1% |
| Recall, 2025 Nazario: std | 0.18 | 0.18 | **0.11** |

Standard deviations are in percentage points.

- **The threshold is 3× more stable than with a single split, and 2× more stable than with the CV peak.** Most hold-out spreads shrank too.
- **One criterion was missed by a hair:** the spread of false alarms on set A is 0.02 points larger than with the CV peak, and the in-domain F1 spread is 0.0004 larger.

Full tables, with the plateau for each seed: [`reports/THRESHOLD_STABILITY.md`](reports/THRESHOLD_STABILITY.md).

## Making the numbers trustworthy

A full review of the project found problems that made earlier numbers wrong or too optimistic. All are fixed, and every report was regenerated with `run_all.sh`.

| Problem | Effect | Fix |
|---|---|---|
| **The Kaggle set contains the SpamAssassin corpus.** 92.8% of SpamAssassin's legitimate emails had a near-copy in Kaggle. | The "train on Kaggle, test on SpamAssassin" experiment tested on emails the model had seen. One headline finding was wrong (see finding 3 below). | The 3,796 copies were removed from Kaggle. |
| **The CLI read raw `.eml` files differently from training.** | 15% of verdicts changed depending on the input format: base64 bodies reached the model undecoded. | Training and the CLI share one parser (`email_parsing.py`). 0 of 981 verdicts differ now. |
| **The CLI's decision was not the measured one.** It used an untuned "hybrid" score. | Recall in actual CLI use was 94.7%, not the 97.8% reported. | The CLI decides with the model's threshold. A hybrid weight tuned on validation came out at **1.0**, meaning ML alone, so the hybrid is gone. |
| **Source cues.** The `Subject:` label exists in every source except Kaggle. Nazario phishing is personalized to its recipient ("Salary Upgrade for **Monkey** Staff"). | "subject" was the model's 49th-strongest feature, a format signal rather than content. | The label is no longer a feature, and the recipient's name is removed everywhere. |
| **Near-duplicates.** 27% of training emails have a near-copy: replies quoting each other, or one campaign with new IDs. | Random splits put copies on both sides, which inflates in-domain scores. In practice the effect was small (F1 0.982 vs 0.981). | All splits and CV folds keep near-duplicate groups together. |
| **"Probability" was not calibrated.** | The probability shown was a raw model score. | Platt scaling on out-of-fold scores: calibration error (ECE) 0.0086 → 0.0027. The CLI states that the probability assumes the training mix (37% phishing). |
| **No uncertainty.** | Single numbers from a few hundred emails. | 95% confidence intervals everywhere, and McNemar tests between versions. |
| **Hand-picked hyperparameters.** | – | A grouped 5-fold CV search picked C=16 and character 2–5-grams (CV F1 +0.0017; [`TUNING.md`](reports/TUNING.md)). |
| **The production model left 15% of the data unused.** | – | After evaluation it is retrained on 100% (21,506 emails). |
| **Personal data in reports.** | Email addresses of mailing-list members appeared in the error examples. | Reports show only masked subject lines, body length and link count. |

## Key findings

| | F1 | ROC-AUC |
|---|---:|---:|
| Rule engine alone (Kaggle test) | 0.07 – 0.65 | 0.65 |
| ML, same dataset (Kaggle test, grouped split) | **0.988** | 0.999 |
| ML, grouped 5-fold CV | 0.989 ± 0.002 | 0.999 |
| ML trained on Kaggle only → Nazario + SpamAssassin | **0.36** (false alarms 34%) | 0.68 |
| Same with text features only | 0.49 (false alarms 25%) | 0.82 |
| Trained on half of Nazario + SpamAssassin → the other half | **0.976** | 0.993 |
| Same, plus Kaggle in training | 0.954 | 0.997 |
| Turkish spam, ML (the rule engine fires on only 0.5% of emails) | 0.937 | 0.974 |

1. **The 99% in-domain score is misleading.** The model's strongest "legitimate" features are `enron`, `vince` and `louise`: names from the Enron mailbox inside the Kaggle set. Some of its "phishing" features are the years `2004` and `2005`. It partly learned *where an email came from*, not *what phishing looks like* (shortcut learning).
2. **The Kaggle "phishing" class is mostly bulk spam.** The real phishing the model missed with the most confidence was targeted corporate lures: fake HR memos, DocuSign requests and `Employee.xls` attachments.
3. **A model trained on Kaggle does not transfer, at all.** *Corrected:* an earlier version of this README said such a model "still ranks emails well (ROC-AUC 0.96); only the threshold fails". That came from the Kaggle–SpamAssassin overlap. With the overlap removed, ROC-AUC is 0.68–0.82 and 25–34% of legitimate emails are flagged.
4. **Data from the target domain matters most, and Kaggle may even hurt.** Half of Nazario + SpamAssassin is enough to reach F1 0.976 on the other half. Adding Kaggle's 13,704 emails lowers that to 0.954.
5. **Hand-written rules belong inside the model, in bounded form.** On their own they are near random (ROC-AUC 0.46–0.64 on the hold-outs). As a separate score mixed in with ML, a validation-tuned mix gives them weight 0.
6. **Real data found bugs.** A malformed URL port (`host:27000To`) crashed the phase-1 analyzer, and some 2025 subjects stayed encoded (`=?UTF-8?B?…?=`). Both are fixed and covered by regression tests.

Full tables and top learned features: [`reports/RESULTS.md`](reports/RESULTS.md).

## How it works

```
raw .eml or text ─► subject + decoded body ─┬─► word TF-IDF (1–2 grams) ─────────────────┐
  (email_parsing.py, same code              ├─► char TF-IDF (2–5 grams, "paypa1") ───────┼─► logistic regression (C=16)
   for training and the CLI)                └─► rule engine → 16 bounded features        │      ─► Platt calibration
                                                 + 6 lure features, scaled, ±3 clip ─────┘      ─► P(phishing) ≥ threshold?
```

- **Training and the CLI read emails with the same code.** A raw `.eml` is MIME-decoded (base64 bodies, HTML, encoded subjects) into the same "subject + body" text the model was trained on.
- **The CLI decision is the model's threshold verdict,** the same decision every table here measures. With a model, exit code `1` means phishing. Without one, the phase-1 rule risk decides (HIGH/CRITICAL → `1`).
- **The threshold is never tuned on test data.** It comes from 5-fold grouped CV on calibrated scores, at the middle of the flat top of the F1 curve.
- **Explainable output:** every prediction lists the features of that email that pushed it toward phishing (coefficient × value).
- **Saved models carry a feature-schema number.** A model saved with older feature code is refused instead of silently scoring wrong.
- **Compared models:** Complement Naive Bayes, Linear SVM and logistic regression, each with and without the rule features (ablation). The old raw-count features stay available for comparison (`feature_version=1`).

## Data

| Dataset | Size | Used as |
|---|---|---|
| [Kaggle Phishing Email Dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) | 13,704 after cleaning, dedup and removing 3,796 SpamAssassin copies | Training |
| [Nazario phishing corpus](https://monkey.org/~jose/phishing/) 2023–2024 (CC-BY 4.0) | 784 | Real phishing |
| [Nazario](https://monkey.org/~jose/phishing/) 2019–2022 | 705 | More real phishing |
| [SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/), ham only | 4,108 | Legitimate emails (2002) |
| Python, Fedora and GNU mailing lists, 2024 | 2,205 | Modern legitimate emails |
| [Nazario](https://monkey.org/~jose/phishing/) 2025 | 454 | **Hold-outs A and B only** (phishing) |
| [Apache mailing lists](https://lists.apache.org/) 2025 (8 lists) | 1,478 | **Hold-out A only** (legitimate) |
| [Ubuntu mailing lists](https://lists.ubuntu.com/) 2025 | 917 | **Hold-out B only** (legitimate) |
| [Phishing Pot](https://github.com/rf-peixoto/phishing_pot) (CC BY-NC 4.0), newest 800 samples, English only | 214 | **Hold-out C only** (phishing, 2026) |
| [OSGeo mailing lists](https://lists.osgeo.org/) 2025 (gdal-dev, qgis-user, qgis-developer) | 767 | **Hold-out C only** (legitimate) |
| Phishing Pot (1,000 random older samples) + ISC, Samba and Mailman lists, 2026 | – | **Lockbox only**, evaluated once |
| [Turkish spam emails](https://huggingface.co/datasets/anilguven/turkish_spam_email) | 1,014 | Language experiment |

What `prepare_data.py` does:

- Parses the raw mbox and MIME files and converts HTML to text, keeping link targets.
- Keeps only the subject and body. Other headers (dates, server names) would leak the source.
- Removes source cues:
  - the Nazario recipient's name, address and company name ("monkey");
  - the Phishing Pot anonymization address;
  - mailing-list subject tags such as `[Python-announce]`.
- Keeps only English Phishing Pot samples. An email counts as English when English stopwords are more frequent than those of 6 other languages and make up at least 3% of the words. This rule was fixed before any results were seen.
- Deduplicates, drops empty rows, and truncates 17M-character outliers.
- **Leakage checks:**
  - Kaggle emails that are near-copies (TF-IDF cosine ≥ 0.8) of SpamAssassin are removed.
  - Hold-out emails that also appear in training are removed.
  - Training emails that are near-copies of any hold-out email are dropped.
  - Training and test sets come from different years and organizations.

**Known limitations**

- **All public legitimate test email comes from open-source mailing lists.** No public dataset of recent commercial mail exists. The one commercial test (the author's private promotions) shows **88–95% false alarms**. Receipts, bank notices and business email are still untested, and so is the risk from the lure features.
- **The commercial test is private and narrow:** one inbox, 45 senders, mostly Turkish. It cannot be reproduced from this repository.
- **Hold-out sets hold a few hundred phishing emails each.** Differences of 1–2 points are within noise; read the confidence intervals.
- **The hold-outs were not blind every time.** A and B were measured at every step; C's errors were inspected after the recall study. Rules were fixed in advance at each step, and the lockbox (evaluated once, at the end) exists for exactly this reason.
- **The lockbox reuses a hold-out source.** Its phishing comes from the same source as hold-out C (Phishing Pot), though the emails themselves are new. Its legitimate emails come from three organizations used nowhere else.
- **Training includes Fedora security notices and hold-out B includes Ubuntu ones.** Different organizations, but the same genre.
- **Phishing Pot labels are partly noisy.** Some of its "phishing" samples are plain marketing spam (its FAQ says so). Its CC BY-NC license allows non-commercial use only; the data is downloaded, never redistributed.
- **The Turkish data is spam, not phishing,** and it is already lemmatized.

## Usage

```bash
pip install -r requirements.txt
./download_data.sh            # all datasets, ~330 MB on disk in data/raw/
./run_all.sh                  # every report + the production model, same code and data (~45 min)
python evaluate_lockbox.py    # one-time lockbox test (already run; refuses to run again)
python experiments_commercial.py  # needs your own Mail export in data/raw/own_promo/ (see the protocol)

python main.py samples/suspicious_sample.txt --no-dns                       # rules only
python main.py samples/suspicious_sample.txt --no-dns --model model.joblib  # + ML
python main.py suspicious.eml --no-dns --model model.joblib                 # raw email works too
```

`model.joblib` is a pickle file: load only models you trained yourself.

```
ML phishing probability: 100.0% -> PHISHING (threshold 0.52)
  calibrated on training data where 37% of mail is phishing; in a real inbox phishing is much rarer
Top ML signals:
  - rules__keyword_hits_log (+4.414)
  - rules__suspicious_url_fraction (+2.142)
  - rules__lure_generic_greeting (+1.6)
  - rules__has_url (+1.596)
  - rules__domain_warning_fraction (+1.167)
```

## Project layout

| File | Purpose |
|---|---|
| `analyzer.py` | Rule engine (phase 1, with the port bug fixed) |
| `email_parsing.py` | Raw email → "subject + body" text, shared by training and the CLI |
| `main.py` | Command-line interface |
| `features.py` | Rule engine output → bounded ML features, plus the lure features |
| `ml_model.py` | Pipelines, CV threshold + calibration, classifier wrapper, explanations, save/load |
| `prepare_data.py` | Raw datasets → clean CSV, with the leakage checks |
| `evaluation.py` | Metrics, confidence intervals, McNemar, calibration, near-duplicate groups, grouped splits, redaction |
| `experiments.py` | Model comparison, CV, cross-dataset, domain adaptation, Turkish → `reports/RESULTS.md` |
| `experiments_false_alarms.py` | False-alarm study → `reports/FALSE_ALARMS.md` |
| `experiments_recall.py` | Recall study → `reports/RECALL.md` |
| `html_signals.py` | Language-independent HTML signals (forms, hidden text, link text vs. target). Measured, not used by the model |
| `header_signals.py` | Sender-authentication checks (recorded SPF/DKIM/DMARC, DKIM and Return-Path alignment). Measured, not used by the model |
| `experiments_headers.py` | SPF/DKIM/DMARC measurement → `reports/HEADERS.md` (private data) |
| `experiments_sender_stage.py` | Pre-registered verified-sender second stage → `reports/SENDER_STAGE.md` (private data) |
| `experiments_commercial_fix.py` | Pre-registered attempt to fix the commercial false alarms → `reports/COMMERCIAL_FIX.md` (private data) |
| `experiments_commercial.py` | Commercial-mail test on private data → `reports/COMMERCIAL.md` (not in `run_all.sh`) |
| `experiments_url.py` | URL feature comparison (production vs + URL, pre-registered rule) → `reports/URL_FEATURES.md` |
| `experiments_tuning.py` | Near-duplicate effect + hyperparameter search → `reports/TUNING.md` |
| `train.py` | Trains, evaluates and saves the production model |
| `evaluate_holdout.py` | Tests the saved model on hold-outs A, B and C → `reports/HOLDOUT_2025.md` |
| `threshold_stability.py` | Compares 3 threshold rules over 5 seeds → `reports/THRESHOLD_STABILITY.md` |
| `evaluate_lockbox.py` | One-time lockbox test → `reports/LOCKBOX.md` |
| `run_all.sh`, `download_*.sh` | Regenerate everything; download all data |
| `test_*.py` | 62 tests, including end-to-end CLI runs (`python -m unittest`) |
| `LICENSE`, `DATA_LICENSES.md` | Code license (MIT) and the licenses of the datasets |

`reports/baseline_v1/` keeps the reports of the first ML version for comparison.

**Privacy:** the reports never include email bodies. Error examples show only the subject line, with email addresses and links masked, plus the body length and link count. The mailing-list archives used here are public, but their authors did not sign up to appear in this repository.

## License

The code is MIT-licensed ([`LICENSE`](LICENSE)). No email data or trained model is included. Each dataset keeps its own license: Nazario is CC BY 4.0, Phishing Pot is CC BY-NC 4.0 (used only for testing), and the Kaggle mirror is LGPL-3.0. See [`DATA_LICENSES.md`](DATA_LICENSES.md) for the full list and attributions.

## Next steps

- **Fix the commercial false alarms first, with English legitimate marketing mail in training.** The P3 diagnostic shows this is the lever; removing Kaggle is not. One source would be a research inbox used only for this project and subscribed to English newsletters, which holds no personal data. A second option is a separate bulk-mail stage before the phishing model, as mail providers do with their Promotions folders.
- Attachment names and types as features. Many of the missed lures say "see attached".
- HTML signals (link text vs. link target, forms, hidden text), once there is legitimate HTML mail to train and test on: in the current data 98% of phishing has HTML and almost no legitimate email does, so any HTML feature would learn the source.
- A transformer baseline (DistilBERT) to compare against TF-IDF on the hold-outs.
- Robustness tests: invisible characters, homoglyphs, `hxxp` / `[.]` link obfuscation.
