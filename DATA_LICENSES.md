# Data sources and licenses

The MIT license in [`LICENSE`](LICENSE) covers **the code only**. This repository includes **no email data and no trained model**. `download_data.sh` fetches every dataset from its original source, and each dataset keeps its own license and terms.

| Dataset | Source | License / terms | Used for |
|---|---|---|---|
| Kaggle Phishing Email Dataset | [Hugging Face mirror](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) | LGPL-3.0, as stated on the mirror | Training |
| Nazario phishing corpus, 2019–2025 | [monkey.org/~jose/phishing](https://monkey.org/~jose/phishing/) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): phishing corpus by Jose Nazario | Training (2019–2024); hold-outs A/B (2025) |
| SpamAssassin public corpus (ham only) | [spamassassin.apache.org](https://spamassassin.apache.org/old/publiccorpus/) | See the corpus readme on the project site | Training |
| Phishing Pot | [github.com/rf-peixoto/phishing_pot](https://github.com/rf-peixoto/phishing_pot) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): non-commercial use only | **Testing only** (hold-out C, lockbox), never for training |
| LLM-written phishing, Gutierrez et al. 2026 | [Zenodo 10.5281/zenodo.20250116](https://doi.org/10.5281/zenodo.20250116) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | **Testing only** (LLM-written phishing test) |
| Human-LLM generated phishing-legitimate emails, Greco et al. 2024 (LLM part) | [Kaggle](https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): non-commercial | **Testing only** (LLM-written phishing test) |
| Turkish spam emails | [Hugging Face](https://huggingface.co/datasets/anilguven/turkish_spam_email) | Not stated ("unknown" on the dataset page) | Language experiment only |
| Public mailing-list archives: Python, Fedora, GNU (2024); Apache, Ubuntu, OSGeo (2025); ISC, Samba, Mailman (2026) | the projects' public list archives | No dataset license. Each message's copyright stays with its author. | Training (2024 lists); testing (2025–2026 lists) |

`reports/llm_legit_audit.json` holds only row numbers and labels from a manual audit of the Greco set, not its text.

**Private data, not distributed:** the commercial-mail test (`experiments_commercial.py`) uses promotional emails from the author's own inbox. They are not in the repository and not downloadable. `reports/COMMERCIAL.md` and `reports/commercial.json` contain only aggregate counts and rates: no subject lines, senders, addresses or bodies.

**What the repository contains from these sources:** the reports in `reports/` contain aggregate metrics and, as error examples, short subject lines with email addresses and links masked. They never contain email bodies. Some of these subject lines come from Phishing Pot (CC BY-NC 4.0) and from the mailing lists above.

**If you train and share a model with this code:** the production model is trained on the Kaggle, Nazario, SpamAssassin and 2024 mailing-list data. Check those licenses before redistributing a trained model. Phishing Pot (non-commercial) is used only for evaluation.
