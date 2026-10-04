# Verified-sender second stage: protocol (written before any result)

Written 2026-10-04, before `experiments_sender_stage.py` was first run. The headers of
Phishing Pot and of Nazario 2025 had never been looked at.

## Idea

The production model flags 87.6% of English promotions. 95% of promotions come from a
**verified sender** (DMARC `pass` and a DKIM signature aligned with the `From` domain), but so
does 27% of training phishing ([`HEADERS.md`](HEADERS.md)). So verification is used to **raise
the bar**, not to switch the alarm off:

```
if verified sender and not suspicious:   phishing  <=>  P(phishing) >= t_v
else:                                    phishing  <=>  P(phishing) >= t      (production, 0.519)
```

- **Verified.** The top-most `Authentication-Results` records `dmarc=pass`, **and** a
  `DKIM-Signature` has `d=` with the same registered domain as `From`.
- **Suspicious.** Any of these, measured or defined earlier, removes the relief:
  - the `From` domain shows a brand name it does not own (`brand_in_foreign_domain`);
  - the `From` domain is punycode (`xn--`);
  - a link points to free hosting (`url_free_hosting_fraction > 0`);
  - `Reply-To` is on another domain.

The production model is unchanged. This is a decision layer on top of it, usable only when the
raw email with headers is available.

**One-way by construction.** The stage can only remove alerts, never add one. False alarms can
only go down and recall can only go down. Emails without headers (all mailing-list hold-outs)
are never verified and keep the production decision.

## Choosing `t_v` (training data only)

1. Out-of-fold scores for the production training set: the production pipeline, grouped
   5-fold CV, seed 42, Platt calibration, exactly as in `fit_with_cv_threshold`.
2. Nazario 2019–2024 training rows are matched to their raw messages by recomputing the text
   exactly as `prepare_data.py` does.
3. `t_v` is the **5th percentile** of the out-of-fold probabilities of training phishing that is
   verified and not suspicious, never below `t`. So, in training, the relief keeps at least 95%
   of the phishing it applies to above the bar.

## Evaluation

- **Recall, blind for headers:**
  - **hold-out C phishing:** 214 English Phishing Pot emails;
  - **Nazario 2025 phishing:** 454 emails, the phishing of A and B.
  - Rows are matched to raw messages the same way, and are scored by the saved production model.
- **False alarms:** the author's 674 promotions, English and all (same set and filters as
  `COMMERCIAL.md`). **Not blind:** this set was used to find the problem, so it gives an upper
  bound on the benefit.
- **Also reported:** the share of verified senders in each set; `t_v`; and, descriptively, a
  sensitivity table for other `t_v` values (0.9, 0.95, 0.99, 0.999, and "never alert").

## Decision rule

The second stage **qualifies** if all three hold:

1. Recall on hold-out C phishing is **not** significantly lower than production (McNemar on
   the phishing emails, p < 0.05 with more misses).
2. The same holds for Nazario 2025 phishing.
3. False alarms on English promotions are significantly lower than production (McNemar,
   p < 0.05).

The sensitivity table does not decide.

**If it qualifies:**
- it is added to the CLI as an opt-in flag for raw `.eml` input;
- the README says the false-alarm benefit is unconfirmed until a blind commercial test (the
  research inbox).

**If it does not qualify:** nothing ships, and the result is reported as is.

## Caveats, fixed in advance

- **Recorded results depend on the receiving server.** Phishing Pot's collector, Nazario's
  server and iCloud all wrote their own. A phishing message that a forwarder broke for DMARC
  looks "unverified" and keeps the strict bar, so recall measured here may be optimistic for
  a direct mailbox.
- **Attackers adapt.** Sending from an authenticated lookalike domain is cheap, and so is
  avoiding the suspicious markers. A real deployment would need domain-age and reputation
  data, which this project cannot measure retrospectively.
