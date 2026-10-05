# Removing source shortcuts (layer 1): protocol (written before any result)

Written 2026-10-05, before `experiments_shortcuts.py` existed. No model with the changes below
has been trained or scored.

## Problem

An exploratory diagnosis of the production model, run after the decision-policy result
([`DECISION_POLICY.md`](DECISION_POLICY.md)), suggests that the model partly learned **where an
email came from** instead of **what it asks the reader to do**:

- **Strongest legitimate features:** `enron` (−7.6, the strongest feature of all), `vince`,
  `louise`, the years `2002` and `2000`, the quote marker `>` (as a character n-gram), and `wrote`.
- **Strongest phishing features:** `your` (+5.4), `you`, `email`, the years `2005`, `2004` and
  `2023`, and `spamassassin sightings` (the name of a mailing list).
- **Same genre, different intent:** ROC-AUC is 0.998 for mailing lists against Nazario 2025
  phishing, but 0.78 for English promotions against the same phishing, and 0.66 against Phishing
  Pot.
- **An exploit:** appending a three-line fake quoted reply ("On … wrote:" and lines starting with
  `>`) drops recall from 99.1% to 72.7% on Nazario 2025 and from 93.0% to 67.8% on Phishing Pot.
  This is how thread-hijacking attacks hide phishing inside a real conversation.

The training data explains it:

| Training source | Class | Quote lines (`>`) | `wrote` | Format |
|---|---|---:|---:|---|
| Kaggle (mostly Enron) | legitimate | 11% (inline) | 2% | lowercased, line breaks removed, punctuation spaced (`a . archer`, `12 / 16 / 01`) |
| SpamAssassin ham, 2002 | legitimate | 48% | 34% | raw |
| Mailing lists, 2024 | legitimate | 47% | 44% | raw |
| Kaggle | phishing | 4% (inline) | 0% | mixed |
| Nazario 2019–2024 | phishing | 0–1% | 0% | raw; 3–10% keep raw HTML tags |

These figures, and the diagnosis above, were looked at before writing this protocol. The
templates in the reply-chain test below are new and fixed here.

## The candidate, S

Three changes, each fixed by a rule. **None of them uses a list of words picked from the
results.**

1. **Reply structure removed,** before every feature (word, character and rule features), in
   training, evaluation and the CLI alike:
   - lines whose first non-space character is `>`;
   - any line of at most 300 characters ending in `wrote:`, together with the line before it if
     that line starts with `On ` (a wrapped attribution);
   - separators, whether on a line of their own or inline in Kaggle's flattened text:
     `-----Original Message-----`, `---------- Forwarded message ----------`, and
     `--- Forwarded by <name> on <date> ---` (the separator itself, up to its closing dashes);
   - runs of two or more consecutive header lines (`From:`, `Sent:`, `To:`, `Cc:`, `Date:`,
     `Subject:`), never including the email's own first line;
   - `Re:`, `Fw:` and `Fwd:` prefixes at the start of the subject.

   The text under a separator is **kept**. It is often the forwarded message itself, for
   example a phishing email a user forwards to be checked. If no body text would remain (a
   first line starting with `Subject:` does not count as body), the `>` markers are removed
   instead and the quoted text is kept.
2. **Dates masked,** at the same point: numeric dates (`12/16/01`, `12 / 16 / 01`, `1-4-2005`,
   `2025-03-01`) become `datetoken`, and years 1950–2049 become `yeartoken`.
3. **Format normalized for the text features only** (word and character TF-IDF):
   - leftover HTML tags and entities are removed;
   - `<` and `>` are deleted;
   - every other punctuation mark is surrounded by spaces;
   - runs of whitespace are collapsed.

   This makes every source look like Kaggle's spaced format, so format alone no longer tells
   sources apart. The rule features still see the unnormalized text, because they need links
   and HTML attributes intact.

**Not changed:** names of people and organizations (`enron`, `vince`, `spamassassin`). Removing
them would need a hand-made list. The source audit below measures how much of that remains.

**Ablations (descriptive):** S without change 1, S without change 2, and S without change 3.

## Procedure

Everything else is as in production: the same training data (`train.DEFAULT_DATA`), lure features
(`feature_version=3`), C=16, grouped 5-fold CV with seed 42, Platt calibration on out-of-fold
scores, the plateau threshold, then a refit on all the data.

**P0** is the saved `model.joblib`. Its out-of-fold scores are recomputed for the paired fold-F1
comparison. **Sanity check:** the recomputed P0 threshold must be 0.519.

## Reply-chain test (templates fixed now)

Each template is appended to every phishing email of **Nazario 2025** (454) and **hold-out C**
(214). The benign text is the same in all templates:

> Hi, thanks for sending this over. I have shared it with the team and we will review it this week.

| | Template | What S does with it |
|---|---|---|
| T1 | Gmail-style: `On Tue, Feb 11, 2025 at 3:42 PM Laura Chen <laura.chen@northwind-consulting.com> wrote:` followed by the benign text as three `>` lines (the third is `> Best regards, Laura`) | removes it (targeted) |
| T2 | Apple-style, wrapped: `On Feb 11, 2025, at 15:42, Laura Chen` / `<laura.chen@northwind-consulting.com> wrote:`, a blank line, then two `>` lines | removes it (targeted) |
| T3 | The three `>` lines of T1 without the attribution | removes it (targeted) |
| T4 | Outlook-style: `-----Original Message-----`, then `From: Laura Chen`, `Sent: Tuesday, February 11, 2025 3:42 PM`, `To: Accounts Team`, `Subject: RE: Q1 budget review`, a blank line, then the benign text without markers | removes the separator and headers; the benign text stays |
| T5 | The benign text alone, without markers | keeps it (untargeted) |
| T6 | `Thanks,` / `Laura` | keeps it (untargeted) |

T1–T3 check that the cleaning works end to end; S passes them by construction. **T4–T6 test
what the model learned:** whether benign-sounding text still pulls a phishing email under the
threshold.

## What is measured

- **Reply-chain recall,** per template and set, for P0 and S (McNemar on the modified emails).
- **Hold-outs A, B and C:** recall and false alarms (McNemar against P0).
- **Zenodo LLM-written phishing** (4,986): recall.
- **Grouped 5-fold CV:** mean paired fold-F1 difference (S − P0) and its standard error.
- **Descriptive:**
  - the ablations on all of the above;
  - promotions, English and all, share flagged (private data, not blind);
  - same-genre ROC-AUC (English promotions against Nazario 2025 phishing and against
    hold-out C phishing);
  - the top 30 features on each side for P0 and S, each with a **source audit**: the share of
    training emails containing the feature that come from its most common corpus (Kaggle
    legitimate, Kaggle phishing, SpamAssassin ham, mailing lists 2024, Nazario 2019–2024).

## Decision rule

**S replaces production** only if all of these hold:

1. **Robustness:** on T1, T2 and T3, recall on Nazario 2025 and on hold-out C is significantly
   higher than P0's (McNemar, p < 0.05, more phishing caught by S). That is six tests.
2. **Recall:** on the unmodified emails, recall on Nazario 2025 and on hold-out C is **not**
   significantly lower than P0's.
3. **False alarms:** on the legitimate emails of A, B and C, false alarms are **not**
   significantly higher, on each set.
4. **LLM-written phishing:** recall on Zenodo is **not** significantly lower.

**In-domain CV F1 does not decide.** Shortcuts help on data from the same sources by
definition, so a small drop there is expected. It is reported with its standard error. The same
goes for T4–T6, the ablations, promotions, AUCs and the audit: all descriptive.

**If S qualifies:**
- it becomes production: `FEATURE_SCHEMA` goes to 4, `model.joblib` is retrained with
  `train.py`, and the hold-out report is regenerated;
- the CLI cleans text the same way, and its explanations use the cleaned text;
- reports of earlier studies stay as they are, because they describe the model of their time.

**If not:** nothing ships, and the result is reported as is.

## Caveats, fixed in advance

- **The templates were written for this test.** They are not real hijacked threads, which quote
  a genuine earlier conversation.
- **S passes T1–T3 by construction.** An attacker who knows the cleaner can pad with unquoted
  text; T5 measures that, and S does not fix it.
- **Removing quotes also removes a real legitimate cue.** 38–60% of legitimate hold-out emails
  quote earlier mail. A one-line reply ("Thanks!") is then scored on that line alone, which can
  raise false alarms. The false-alarm guard exists for this reason.
- **Masking dates removes one dating shortcut,** but topics and product names still date an email.
- **The hold-outs are not blind,** and the lockbox has been used.
- **This does not add the missing kind of mail.** Promotions are expected to stay mostly flagged.
  That needs layer 2: legitimate organization-to-customer mail from the research inbox.

## Predictions

None were written.
