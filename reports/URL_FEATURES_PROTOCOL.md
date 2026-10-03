# URL features: protocol (written before any result)

Written 2026-10-03, before `experiments_url.py` was run for the first time. The decision rule
below is fixed; the result report (`URL_FEATURES.md`) follows it whatever the numbers say.

## What is compared

- **v3** (current production features): bounded rule features + lure features.
- **v4**: v3 + 8 bounded URL features (`features.py`, `URL_FEATURE_NAMES`).

Same training data (`train.DEFAULT_DATA`), same hyperparameters, same procedure as production:
grouped 5-fold CV (near-duplicate groups, seed 42), Platt calibration on out-of-fold scores,
plateau-center threshold, then a refit on all the data.

## How the features were chosen

Only training sources were inspected: Nazario 2019–2024, SpamAssassin ham, the 2024 mailing
lists and Kaggle. No hold-out or lockbox email was looked at. Candidate signals were kept when
they separate training phishing from training legitimate mail **and** have a reason to hold
outside this data. `.php`/`.asp` links were dropped: 27% of 2019–22 phishing, 9% of 2002 ham,
1% of 2024 ham, so they mostly encode the year. A list of known-good or known-bad domains was
not built: Kaggle's "phishing" class links mostly to 2002 mailing-list footers, so such a list
would learn the source.

The free-hosting list is defined by a rule, not by which hosts appear in the data: services
where anyone can serve an arbitrary web page or raw file on the service's own domain without
owning one (IPFS gateways, static and serverless hosting, raw cloud-storage buckets, site
builders, tunnels). File-sharing apps that show files in their own viewer (Google Drive,
Dropbox, WeTransfer, SharePoint) are not included. The list includes hosts legitimate
projects also use (`github.io`, `netlify.app`, `s3.amazonaws.com`).

## Decision rule

v4 replaces v3 in production only if **both** hold:

1. **Primary (training data only):** the mean paired difference in fold F1 (v4 − v3, each at
   its own production threshold, same folds) is larger than its standard error.
2. **Guard (hold-outs A, B, C):** on none of the three sets does v4 raise false alarms
   significantly: exact McNemar test on the legitimate emails only, p < 0.05 with more
   emails wrongly flagged by v4 than by v3.

Hold-out recall and F1 are reported but do not decide. A, B and C have all been seen during
earlier development, so they are not blind; the lockbox has been used and is not touched.

## Known blind spot

All legitimate test mail comes from open-source mailing lists. Commercial mail uses click
trackers (redirect URLs, percent-encoding, long subdomains) and free hosting far more often.
A false-alarm rate measured here does not cover it, whichever version wins.
