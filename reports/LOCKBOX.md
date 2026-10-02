# Lockbox test: one-time evaluation

Protocol (written before this run): [`LOCKBOX_PROTOCOL.md`](LOCKBOX_PROTOCOL.md). This is the only evaluation of the saved model on this data.

- Model: `model.joblib`, sha256 `46feebe4735c0082…`, trained 2026-10-02 15:11:59 on 21506 emails, threshold 0.519.
- Phishing: 466 of 1000 older Phishing Pot samples are English. Legitimate: ISC, Samba and Mailman mailing lists (January, April and July 2026).
- 0 emails were dropped as near-copies of training emails. Left: **457 phishing, 562 legitimate**.

| model | precision | recall (95% CI) | F1 | false alarms (95% CI) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| phase 1: rule_based (score >= 50) | 0.814 | 42.2% (37.8–46.8) | 0.556 | 7.8% (5.9–10.3) | 0.7978 |
| phase 2: ml (saved threshold) | 0.993 | 96.9% (94.9–98.2) | 0.981 | 0.5% (0.2–1.6) | 0.9987 |

| source | n | true label | flagged by ML | flagged by rules |
|---|---:|---|---:|---:|
| isc-bind-users | 146 | legit | 0.7% | 9.6% |
| mailman-users | 191 | legit | 0.5% | 7.8% |
| phishing_pot_older | 457 | phishing | 96.9% | 42.2% |
| samba-samba | 225 | legit | 0.4% | 6.7% |

## Missed phishing (lowest score)

- **0.002** Subject: [john-users] Re: SNMPv3 format [body: 819 chars, 2 links]
- **0.068** Subject: CoinTracker Desktop — A New Standard for Crypto Clarity & Security [body: 2519 chars, 1 links]
- **0.103** Subject: You have an updated user agreement [body: 18 chars, 0 links]
- **0.257** Subject: [Monronee] Login Details [body: 264 chars, 3 links]
- **0.325** Subject: Important: New Rabby Desktop Upgrade [body: 1524 chars, 1 links]
- **0.326** Subject: Hello [body: 70 chars, 0 links]
- **0.354** Subject: Give the Gift of Comfort This Season 🎁🛁 [body: 656 chars, 3 links]
- **0.405** Subject: Discover Lido V2 Platform [body: 448 chars, 0 links]

## False alarms (highest score)

- **0.881** Subject: Named issues with T-Mobile 5G for Business [body: 9315 chars, 0 links]
- **0.852** Subject: [Samba] GPLv3 Samba Compliance Issue: Innuos Zen mini (Innuos v3.4.4) [body: 2278 chars, 2 links]
- **0.704** Subject: [MM3-users] Re: broken route to accounts [body: 3478 chars, 5 links]
