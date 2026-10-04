# Commercial mail: protocol (written before any result)

Written 2026-10-04, while the data was still being exported and before `experiments_commercial.py`
was first run. Nothing in it was changed after seeing results.

## Why

Every legitimate test email so far comes from open-source mailing lists, which are almost
never HTML. So two questions are open:

1. How often do the candidate HTML signals, and the URL features from
   [`URL_FEATURES.md`](URL_FEATURES.md), fire on real commercial mail?
2. What is the model's false-alarm rate on commercial mail?

## Data

- About 2,000 promotional emails (newsletters, campaigns) that the author copied from their
  own inbox's Promotions category. Receipts, orders, banking, password resets and anything
  from a person were left out. Most are Turkish.
- Kept in `data/raw/own_promo/`, outside git, and **never published**. The results cannot be
  reproduced from the repository, and the README says so.
- **Privacy:**
  - The author's name and addresses, listed in a local `redact.txt`, are removed (whole words,
    case-insensitive, including Turkish İ/ı) before any text reaches the model.
  - No script prints or reports a body, subject line or sender; reports hold counts only.
  - Mail is read offline, so no remote image is fetched.
- **Processing:**
  - Same parser as training (`email_parsing.py`).
  - Emails under 40 characters are dropped and exact duplicates removed.
  - At most 30 emails per sender domain are kept (random, seed 42), so one brand cannot
    dominate.
  - The language split uses the existing `looks_english` rule.

## What is measured

**1. Signal rates, the main result.** These are language-independent:
- HTML signals (`html_signals.py`): `html_only`, `form`, `password_input`, `hidden_text`,
  `link_text_domain_mismatch`, `brand_text_foreign_link`, `image_heavy`, `meta_refresh`,
  `script_or_data_link`. Each is counted among emails that have HTML.
- The 7 flag-type URL features, counted among all emails.

The comparison groups are training phishing (Nazario 2019–2024) and SpamAssassin ham. No
hold-out or lockbox email is used.

**Decision rule.** A signal is **promising** (worth trying as a model feature later) only if
it fires on at least 5% of training phishing **and** at least 3× as often as on commercial
mail. The commercial rate is floored at 1/n, so a zero does not give an infinite ratio.

**2. False alarms.** Every email in the set is legitimate, so each phishing verdict is a
false alarm. Two models are compared:
- v3, the saved production model, at its own threshold;
- v4, trained with the production procedure plus the URL features.

Results are reported separately for English and non-English mail, with 95% Wilson intervals
and McNemar v4 vs v3.

The model was trained on English only. The non-English false-alarm rate therefore mixes
language shift with commercial style, and is reported as such, not as the model's
commercial false-alarm rate. Nothing here changes the production model.

## Known limits

- **One inbox:** one person's subscriptions, mostly Turkish brands, so it is not a sample of
  commercial mail in general.
- **Labels:** "legitimate" relies on the Promotions category plus the author's selection. A
  few spam emails may be in the set.
- **Comparison phishing is older:** Nazario 2019–2024 against promotions that are mostly
  recent, so part of any difference can be a year effect.
