# LLM-written phishing: protocol (written before any result)

Written 2026-10-04, before `experiments_llm.py` was first run. The datasets were downloaded and
their format checked: columns, label values, share of links, and 12 random "legitimate" Greco
samples to check the labels. No email had been scored.

## Question

Does the production model, trained on human-written phishing, catch phishing written by
modern LLMs? Or did it partly learn "clumsy writing = phishing"?

## Data (test only, never used for training or tuning)

| Set | Emails | Generators | Use |
|---|---:|---|---|
| **Zenodo**, Gutierrez et al. 2026 (CC BY 4.0) | 4,986 phishing | GPT-4.1, DeepSeek 3.2, Llama 3.3 70B; 5 themes (banking, parcel, IT support, tax, HR) | **primary** |
| **Greco** et al., ITASEC 2024 (CC BY-NC-SA 4.0), LLM part | 1,000 phishing | ChatGPT, WormGPT | secondary |
| Greco, LLM "legitimate" part | 1,000 | ChatGPT | **weak control only** |

- **Formatting.** Zenodo emails become `Subject: <subject>` + body, as in training. Greco has no
  subject line, so the body is used as is (like Kaggle in training). Both go through the
  project's whitespace cleaning.
- **Repairing Greco's CSV.** The text field is unquoted, so commas split it. The label is the
  last field, and the text is everything before it, joined back.
- **Why Greco's legitimate part is only a weak control.** In 12 random samples checked before
  the run, several read like phishing ("time is running out to claim your tax refund"). A
  flagged email there is not necessarily a false alarm, so its rate is reported, but nothing
  is concluded from it.
- **The human-written parts of both sets are not used.** They overlap the training sources
  (Nazario, Enron).
- **Near-duplicates.** Any test email with TF-IDF cosine ≥ 0.8 to a training phishing email is
  counted and excluded from the primary numbers.

## What is measured

The saved production model at its own threshold. No retraining, no tuning.

- **Recall.** Overall, per generator and per theme, with 95% Wilson intervals. A chi-square test
  checks whether recall differs across generators, and separately across themes.
- **Subgroups.** Emails with a real link, with only a bracketed placeholder (13% of Zenodo, for
  example `[Verify Your Account Now]`), or with neither.
- **Errors.** Subject lines of missed emails, masked as usual. The data is public and synthetic.

## Decision rule

**LLM-written phishing counts as harder for the model** if Zenodo recall (near-duplicates
excluded) is significantly lower than recall on hold-out C phishing: 199 of 214, the
lowest-recall human-written hold-out. Significance means a Fisher exact test with p < 0.05.

Hold-out C is a different population (real 2026 honeypot phishing), so this is a comparison of
difficulty, not a controlled experiment.

## Caveats, fixed in advance

- **Generated on request.** These emails came from prompts that asked for phishing in fixed
  themes. Real attackers' LLM-written phishing may differ.
- **Placeholders.** Bracketed placeholders are an artifact of generation: a real attack would
  carry a link there.
- **Topic overlap.** The themes (banking, parcel, IT support, tax, HR) overlap the training
  phishing, so this tests mostly **style**, not new lures.
