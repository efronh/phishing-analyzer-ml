# Sender authentication (SPF, DKIM, DMARC): protocol (written before any result)

Written 2026-10-04, before `experiments_headers.py` was first run. Only an inventory of
header **presence** had been looked at, never their values.

## Question

How often do authentication results and related header checks fire on phishing compared with
legitimate commercial mail? This is a measurement only; nothing is added to the model.

## Sources

| Source | Role | Why |
|---|---|---|
| Nazario 2019–2024 (1,571) | phishing | training corpus; 73% carry `Authentication-Results`, 59% a DKIM signature |
| Author's promotional emails | legitimate | the only legitimate source with headers: every one has them |

- **Not used: Phishing Pot.** It has full headers but is kept back, so that any header feature
  built later can be tested on it blind.
- **Not used: the lockbox.** It stays untouched.
- **Not used: mailing-list archives.** They strip these headers (0% present), so they cannot
  be compared.
- **Promotions** are loaded as in `COMMERCIAL.md` (iCloud routed them to `INBOX`, exact
  duplicates removed, at most 30 per sender domain), but **without the DMARC part of the junk
  filter**, which would make DMARC pass 100% by construction. Reports hold counts only.

## What is measured

**Recorded results.** For each mechanism (SPF, DKIM, DMARC), the result in the top-most
`Authentication-Results` header that reports it (the last receiving server): `pass`, `fail`,
`softfail`, `neutral`, `none`, an error, or missing.

**Checks computed from the headers.** These do not depend on any receiving server:
- `no_auth_results`: no `Authentication-Results` header at all.
- `spf_not_pass`, `dkim_not_pass`, `dmarc_not_pass`: a result is recorded and it is not
  `pass`. The denominator is emails with a recorded result.
- `no_dkim_signature`: no `DKIM-Signature` header.
- `dkim_not_aligned`: no DKIM signature whose `d=` registered domain equals the `From`
  registered domain (relaxed alignment).
- `return_path_not_aligned`: the `Return-Path` registered domain differs from the `From`
  registered domain.
- `reply_to_other_domain`: a `Reply-To` exists and its registered domain differs from the
  `From` domain.
- `from_freemail`: the `From` domain is a free mailbox provider (gmail.com, outlook.com,
  hotmail.com, yahoo.com, aol.com, icloud.com, mail.ru, yandex.ru, gmx.com, proton.me and the
  like).

**Decision rule** (as for the HTML signals). A signal is **promising** if it fires on at least
5% of phishing **and** at least 3× as often as on promotions. The promotions rate is floored at
1/n. 95% Wilson intervals throughout.

## Caveats, fixed in advance

- **Different receivers.** The recorded results come from different receiving servers: iCloud
  for the promotions, the collector's server or a forwarder for Nazario. Forwarding can break
  SPF and DMARC, and 64% of Nazario messages carry ARC headers, which suggests forwarding. The
  computed checks are there for this reason.
- **Labels and identity.** Nazario's own header rewriting is unknown. The recipient is the same
  person for the whole corpus, so `To`-based checks are not used.
- **One inbox.** The legitimate side is one inbox and 45 senders, and every promotion comes
  from a sender the author chose to receive mail from. Real legitimate mail also includes
  person-to-person mail from freemail addresses, which this set lacks, so `from_freemail` will
  look stronger here than it is.
