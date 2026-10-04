#!/usr/bin/env python3
# Ticari meşru mail testi (yazarın kendi gelen kutusundaki reklam mailleri, çoğu Türkçe).
# Protokol, sonuçlardan önce yazıldı: reports/COMMERCIAL_PROTOCOL.md
#
# Gizlilik: mailler data/raw/own_promo/ altında kalır (git dışı) ve bu script hiçbir mailin
# gövdesini, konusunu ya da gönderenini yazdırmaz / rapora koymaz - sadece toplam sayılar.
# redact.txt'deki kelimeler (yazarın adı, adresi) metin modele girmeden silinir.
#
# Ölçülenler:
#   1. HTML ve URL sinyallerinin ne sıklıkta tetiklendiği: ticari mail vs eğitimdeki phishing
#      (Nazario 2019-2024) vs eğitimdeki meşru mail (SpamAssassin). Dilden bağımsız.
#   2. Üretim modelinin (v3) ve URL feature'lı v4'ün bu maillerde yanlış alarm oranı,
#      İngilizce ve İngilizce olmayan ayrı (model sadece İngilizce eğitildi).
#
# Kullanım: python experiments_commercial.py

import json
import mailbox
import os
import random
import re
import sys
import time

import numpy as np

from email_parsing import message_to_text, parse_bytes
from evaluation import load_csv, mcnemar_exact, near_duplicate_groups, wilson_ci
from features import URL_FEATURE_NAMES, _url_features
from html_signals import SIGNAL_NAMES, html_signals, sender_domain
from ml_model import PhishingClassifier, fit_with_cv_threshold
from prepare_data import RECIPIENT_REGEX, SA_HAM_DIRS, dedup, looks_english
from train import DEFAULT_DATA

RAW = os.path.join("data", "raw")
PROMO_DIR = os.path.join(RAW, "own_promo")
REPORTS = "reports"
SEED = 42
# tek bir marka seti domine etmesin
MAX_PER_SENDER = 30
# bir sinyal "umut verici" sayılır: phishing'in en az %5'inde ve ticari maildekinin en az 3 katı
MIN_PHISH_RATE = 0.05
MIN_RATIO = 3.0
TRAIN_PHISH_FILES = ["nazario-%d.mbox" % y for y in range(2019, 2025)]


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + msg, flush=True)


def tr_variants(term):
    """Türkçe büyük/küçük harf: "efe" -> "EFE", "Efe"; "i" -> "İ", "ı" -> "I"."""
    up = term.replace("i", "İ").replace("ı", "I").upper()
    low = term.replace("İ", "i").replace("I", "ı").lower()
    return {term, term.lower(), term.upper(), term.title(), up, low, low[:1].replace("i", "İ").upper() + low[1:]}


def load_redactor():
    path = os.path.join(PROMO_DIR, "redact.txt")
    terms = []
    if os.path.exists(path):
        f = open(path, encoding="utf-8", errors="replace")
        terms = [line.strip() for line in f if len(line.strip()) >= 2]
        f.close()
    variants = sorted({v for t in terms for v in tr_variants(t)}, key=len, reverse=True)
    if not variants:
        return None, 0
    # kelime sınırı: "Efe" silinsin ama "Efes" kalsın
    pattern = r"(?<!\w)(" + "|".join(re.escape(v) for v in variants) + r")(?!\w)"
    return re.compile(pattern, re.IGNORECASE), len(terms)


def promo_mbox_paths():
    out = []
    for name in sorted(os.listdir(PROMO_DIR)):
        p = os.path.join(PROMO_DIR, name)
        if name.endswith(".partial.mbox"):
            continue
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "mbox")):
            out.append(os.path.join(p, "mbox"))
        elif name.endswith(".mbox") and os.path.isfile(p):
            out.append(p)
    return out


def read_mbox_messages(path):
    for msg in mailbox.mbox(path, factory=lambda fp: parse_bytes(fp.read())):
        yield msg


def load_promo(redactor):
    rows = []
    for path in promo_mbox_paths():
        for msg in read_mbox_messages(path):
            try:
                text = message_to_text(msg)
                sig = html_signals(msg)
            except Exception:
                continue
            if redactor is not None:
                text = redactor.sub("", text)
            if len(text) < 40:
                continue
            rows.append({"text": text, "label": 0, "source": "own_promo",
                         "sender": sender_domain(msg), "html": sig})
    n_raw = len(rows)
    rows = dedup(rows)
    by_sender = {}
    for r in rows:
        by_sender.setdefault(r["sender"], []).append(r)
    rng = random.Random(SEED)
    capped = []
    for s in sorted(by_sender):
        items = by_sender[s]
        if len(items) > MAX_PER_SENDER:
            items = rng.sample(items, MAX_PER_SENDER)
        capped.extend(items)
    for r in capped:
        r["english"] = looks_english(r["text"])
    return capped, {"messages_read": n_raw, "after_dedup": len(rows), "senders": len(by_sender),
                    "after_sender_cap": len(capped)}


def load_reference():
    """Karşılaştırma için eğitim kaynakları (hold-out'lara dokunulmuyor)."""
    phish, ham = [], []
    for name in TRAIN_PHISH_FILES:
        for msg in read_mbox_messages(os.path.join(RAW, name)):
            try:
                text = RECIPIENT_REGEX.sub("", message_to_text(msg))
                phish.append({"text": text, "html": html_signals(msg)})
            except Exception:
                continue
    for d in SA_HAM_DIRS:
        folder = os.path.join(RAW, d)
        for fname in sorted(os.listdir(folder)):
            if fname == "cmds":
                continue
            f = open(os.path.join(folder, fname), "rb")
            raw = f.read()
            f.close()
            try:
                msg = parse_bytes(raw)
                ham.append({"text": message_to_text(msg), "html": html_signals(msg)})
            except Exception:
                continue
    return phish, ham


def rates(rows):
    n = len(rows)
    out = {"n": n}
    for s in SIGNAL_NAMES:
        k = sum(1 for r in rows if r["html"][s])
        out[s] = {"rate": round(k / n, 4) if n else None, "ci": wilson_ci(k, n)}
    url = [_url_features(r["text"]) for r in rows]
    for i, s in enumerate(URL_FEATURE_NAMES[:-1]):   # sonuncusu sayı, bayrak değil
        k = sum(1 for u in url if u[i] > 0)
        out["url_" + s[4:]] = {"rate": round(k / n, 4) if n else None, "ci": wilson_ci(k, n)}
    k = sum(1 for u in url if any(u))
    out["has_link"] = {"rate": round(k / n, 4) if n else None, "ci": wilson_ci(k, n)}
    return out


def main():
    if not promo_mbox_paths():
        print("No finished .mbox export in " + PROMO_DIR, file=sys.stderr)
        return 1
    started = time.time()
    redactor, n_terms = load_redactor()
    log("redaction terms: " + str(n_terms))
    promo, counts = load_promo(redactor)
    log("commercial set: " + json.dumps(counts) + ", English: " + str(sum(r["english"] for r in promo)))
    phish, ham = load_reference()
    log("reference: " + str(len(phish)) + " training phishing, " + str(len(ham)) + " SpamAssassin ham")

    # 1. sinyal oranları
    groups = {"commercial (all)": promo,
              "commercial, HTML only": [r for r in promo if r["html"]["has_html"]],
              "training phishing (Nazario 2019-24)": phish,
              "training phishing, HTML only": [r for r in phish if r["html"]["has_html"]],
              "SpamAssassin ham": ham}
    signal_rates = {g: rates(rows) for g, rows in groups.items()}
    # umut verici mi? HTML sinyalleri HTML'li mailler içinde, URL sinyalleri bütün maillerde
    verdicts = {}
    for s in SIGNAL_NAMES[1:] + ["url_" + n[4:] for n in URL_FEATURE_NAMES[:-1]]:
        within = "HTML only" if s in SIGNAL_NAMES else None
        p = signal_rates["training phishing, HTML only" if within else "training phishing (Nazario 2019-24)"][s]["rate"]
        c_group = "commercial, HTML only" if within else "commercial (all)"
        c = signal_rates[c_group][s]["rate"]
        n_c = signal_rates[c_group]["n"]
        ratio = p / max(c, 1.0 / max(n_c, 1))
        verdicts[s] = {"phishing_rate": p, "commercial_rate": c, "ratio": round(ratio, 2),
                       "promising": bool(p >= MIN_PHISH_RATE and ratio >= MIN_RATIO)}

    # 2. modelin yanlış alarmları: v3 = kayıtlı üretim modeli, v4 = aynı prosedürle URL feature'lı
    log("loading production model (v3)")
    v3 = PhishingClassifier.load("model.joblib")
    log("training v4 with the production procedure")
    texts, labels = [], []
    for p in DEFAULT_DATA:
        X, y, _ = load_csv(p)
        texts.extend(X)
        labels.append(y)
    labels = np.concatenate(labels)
    v4 = fit_with_cv_threshold(texts, labels, folds=5, seed=SEED, n_jobs=5,
                               groups=near_duplicate_groups(texts), feature_version=4)
    X = [r["text"] for r in promo]
    english = np.array([r["english"] for r in promo])
    pred = {"v3": v3.predict_proba(X) >= v3.threshold, "v4": v4.predict_proba(X) >= v4.threshold}
    false_alarms = {}
    for subset, mask in [("English", english), ("not English", ~english), ("all", np.ones(len(X), bool))]:
        n = int(mask.sum())
        row = {"n": n}
        for name, p in pred.items():
            k = int(p[mask].sum())
            row[name] = {"false_alarms": k, "rate": round(k / n, 4) if n else None, "ci": wilson_ci(k, n)}
        mc = mcnemar_exact(np.zeros(n, int), pred["v4"][mask], pred["v3"][mask]) if n else None
        row["mcnemar_v4_vs_v3"] = mc
        false_alarms[subset] = row

    results = {"counts": counts, "redaction_terms": n_terms, "max_per_sender": MAX_PER_SENDER,
               "english": int(english.sum()), "signal_rates": signal_rates, "verdicts": verdicts,
               "thresholds": {"v3": v3.threshold, "v4": v4.threshold}, "false_alarms": false_alarms,
               "runtime_seconds": round(time.time() - started, 1)}
    os.makedirs(REPORTS, exist_ok=True)
    f = open(os.path.join(REPORTS, "commercial.json"), "w", encoding="utf-8")
    json.dump(results, f, indent=2, default=float)
    f.close()
    write_markdown(results)
    log("Done in " + str(results["runtime_seconds"]) + "s -> reports/COMMERCIAL.md")
    return 0


def pct(x):
    return "-" if x is None else "%.1f%%" % (100 * x)


def write_markdown(r):
    c = r["counts"]
    L = ["# Commercial legitimate mail", ""]
    L.append("Generated by `experiments_commercial.py`. Protocol, written before the first run: "
             "[`COMMERCIAL_PROTOCOL.md`](COMMERCIAL_PROTOCOL.md).")
    L.append("")
    L.append("**Data:** promotional emails from the author's own inbox, mostly Turkish. They are **not "
             "published**, so these numbers cannot be reproduced from this repository. The report holds "
             "counts only: no subject lines, senders or bodies.")
    L.append("")
    L.append("- %d emails read, %d after removing duplicates, %d sender domains, %d after keeping at most "
             "%d per sender. %d of them are English by the project's language rule."
             % (c["messages_read"], c["after_dedup"], c["senders"], c["after_sender_cap"],
                r["max_per_sender"], r["english"]))
    L.append("")
    L.append("## 1. How often each signal fires")
    L.append("")
    L.append("HTML signals are counted among emails that have HTML; URL signals among all emails. "
             "A signal is called **promising** if it fires on at least %d%% of training phishing and at "
             "least %g× as often as on commercial mail (rule fixed in advance). Comparison phishing is "
             "the training corpus (Nazario 2019–2024); no hold-out was used."
             % (int(100 * MIN_PHISH_RATE), MIN_RATIO))
    L.append("")
    sr = r["signal_rates"]
    cols = ["commercial (all)", "commercial, HTML only", "training phishing (Nazario 2019-24)",
            "training phishing, HTML only", "SpamAssassin ham"]
    L.append("| signal | " + " | ".join(cols) + " | phishing ÷ commercial | promising |")
    L.append("|---|" + "---:|" * len(cols) + "---:|---|")
    L.append("| emails | " + " | ".join(str(sr[g]["n"]) for g in cols) + " | | |")
    for s in ["has_html", "has_link"] + list(r["verdicts"]):
        v = r["verdicts"].get(s)
        L.append("| `" + s + "` | " + " | ".join(pct(sr[g][s]["rate"]) for g in cols) + " | "
                 + ("%.1f×" % v["ratio"] if v else "") + " | "
                 + (("**yes**" if v["promising"] else "no") if v else "") + " |")
    L.append("")
    L.append("## 2. False alarms of the model")
    L.append("")
    L.append("Every email here is legitimate, so each phishing verdict is a false alarm. The model was "
             "trained on English only: the non-English rows measure a language shift as much as "
             "commercial mail, and are not a fair false-alarm rate.")
    L.append("")
    L.append("| subset | emails | v3 (production) | v4 (+ URL features) | McNemar p |")
    L.append("|---|---:|---:|---:|---:|")
    for subset, row in r["false_alarms"].items():
        cells = []
        for name in ["v3", "v4"]:
            m = row[name]
            cells.append("-" if m["rate"] is None else "%s (%.1f–%.1f)" % (pct(m["rate"]), 100 * m["ci"][0],
                                                                         100 * m["ci"][1]))
        p = row["mcnemar_v4_vs_v3"]["p_value"] if row["mcnemar_v4_vs_v3"] else None
        L.append("| %s | %d | %s | %s | %s |" % (subset, row["n"], cells[0], cells[1],
                                                 "-" if p is None else "%.3g" % p))
    L.append("")
    f = open(os.path.join(REPORTS, "COMMERCIAL.md"), "w", encoding="utf-8")
    f.write("\n".join(L))
    f.close()


if __name__ == "__main__":
    sys.exit(main())
