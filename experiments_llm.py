#!/usr/bin/env python3
# LLM ile yazılmış phishing testi: üretim modeli, insan yazımı phishing'le eğitildi; modern
# LLM'lerin yazdığı phishing'i yakalayabiliyor mu? Protokol (sonuçlardan önce yazıldı):
# reports/LLM_PHISHING_PROTOCOL.md. SADECE TEST - hiçbir şey eğitilmiyor ya da ayarlanmıyor.
#
# Kullanım: ./download_llm.sh && python experiments_llm.py

import csv
import json
import os
import re
import sys
import time

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

from email_parsing import MAX_CHARS, clean_whitespace
from evaluate_holdout import near_duplicate_mask
from evaluation import describe_email, load_csv, wilson_ci
from ml_model import PhishingClassifier
from train import DEFAULT_DATA

LLM = os.path.join("data", "raw", "llm")
REPORTS = "reports"
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"\[[^\]]{2,40}\]")
# hold-out C phishing: insan yazımı hold-out'lar arasında en düşük recall (reports/HOLDOUT_2025.md)
C_REF = (199, 214)

csv.field_size_limit(10 ** 9)


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + msg, flush=True)


def model_text(subject, body):
    text = ("Subject: " + subject.strip() + "\n\n" + body) if subject and subject.strip() else body
    return clean_whitespace(text)[:MAX_CHARS]


def load_zenodo():
    f = open(os.path.join(LLM, "zenodo", "data", "llm_corpus_sampled.csv"), encoding="utf-8")
    rows = [{"text": model_text(r["subject"], r["body"]), "body": r["body"], "generator": r["model"],
             "theme": r["category"]} for r in csv.DictReader(f)]
    f.close()
    return rows


def load_greco(name):
    """Greco'nun LLM CSV'lerinde metin tırnaksız: etiket son alan, metin ondan öncekilerin birleşimi."""
    f = open(os.path.join(LLM, "greco", "llm-generated", name), encoding="utf-8", errors="replace")
    body = list(csv.reader(f))[1:]
    f.close()
    rows = []
    for r in body:
        text = ",".join(r[:-1]).strip()
        if len(text) >= 40:
            rows.append({"text": model_text("", text), "body": text, "generator": "ChatGPT/WormGPT", "theme": "-"})
    return rows


def rate(k, n):
    return {"k": int(k), "n": int(n), "rate": round(k / n, 4) if n else None, "ci": wilson_ci(int(k), int(n))}


def link_group(body):
    if URL_RE.search(body):
        return "real link"
    if PLACEHOLDER_RE.search(body):
        return "placeholder only"
    return "no link"


def breakdown(rows, pred, key):
    out = {}
    for v in sorted(set(r[key] for r in rows)):
        m = np.array([r[key] == v for r in rows])
        out[v] = rate(pred[m].sum(), m.sum())
    return out


def chi2(groups):
    table = [[g["k"], g["n"] - g["k"]] for g in groups.values()]
    if len(table) < 2:
        return None
    stat, p, _, _ = chi2_contingency(table)
    return round(float(p), 6)


def main():
    if not os.path.exists(os.path.join(LLM, "zenodo", "data", "llm_corpus_sampled.csv")):
        print("Missing LLM data - run ./download_llm.sh first.", file=sys.stderr)
        return 1
    started = time.time()
    model = PhishingClassifier.load("model.joblib")
    sets = {"Zenodo (GPT-4.1, DeepSeek 3.2, Llama 3.3)": load_zenodo(),
            "Greco LLM phishing (ChatGPT, WormGPT)": load_greco("phishing.csv"),
            "Greco LLM 'legitimate' (weak control)": load_greco("legit.csv")}

    # eğitimdeki phishing'e yakın kopyalar
    train_phish = []
    for p in DEFAULT_DATA:
        X, y, _ = load_csv(p)
        train_phish.extend(t for t, l in zip(X, y) if l == 1)

    results = {"threshold": model.threshold, "sets": {}}
    for name, rows in sets.items():
        log(name + ": " + str(len(rows)) + " emails")
        texts = [r["text"] for r in rows]
        dup, _ = near_duplicate_mask(texts, train_phish)
        keep = ~dup
        rows = [r for r, k in zip(rows, keep) if k]
        texts = [r["text"] for r in rows]
        p = model.predict_proba(texts)
        pred = p >= model.threshold
        for r in rows:
            r["links"] = link_group(r["body"])
        entry = {"n_raw": len(keep), "near_duplicates_removed": int(dup.sum()), "flagged": rate(pred.sum(), len(pred)),
                 "median_probability": round(float(np.median(p)), 4),
                 "by_generator": breakdown(rows, pred, "generator"), "by_theme": breakdown(rows, pred, "theme"),
                 "by_links": breakdown(rows, pred, "links")}
        entry["chi2_p_generator"] = chi2(entry["by_generator"])
        entry["chi2_p_theme"] = chi2(entry["by_theme"])
        missed = [i for i in np.argsort(p) if not pred[i]][:10]
        entry["lowest_scored"] = [{"score": round(float(p[i]), 4), "generator": rows[i]["generator"],
                                   "theme": rows[i]["theme"], "text": describe_email(texts[i])} for i in missed]
        results["sets"][name] = entry

    z = results["sets"]["Zenodo (GPT-4.1, DeepSeek 3.2, Llama 3.3)"]["flagged"]
    _, p_fisher = fisher_exact([[z["k"], z["n"] - z["k"]], [C_REF[0], C_REF[1] - C_REF[0]]])
    harder = z["rate"] < C_REF[0] / C_REF[1] and p_fisher < 0.05
    results["decision"] = {"zenodo_recall": z["rate"], "holdout_c_recall": round(C_REF[0] / C_REF[1], 4),
                           "fisher_p": round(float(p_fisher), 6), "llm_phishing_harder": bool(harder)}
    results["runtime_seconds"] = round(time.time() - started, 1)
    log("decision: " + json.dumps(results["decision"]))
    f = open(os.path.join(REPORTS, "llm_phishing.json"), "w", encoding="utf-8")
    json.dump(results, f, indent=2, default=float)
    f.close()
    write_markdown(results)
    log("Done -> reports/LLM_PHISHING.md")
    return 0


def pct(x):
    return "-" if x["rate"] is None else "%.1f%% (%.1f–%.1f)" % (100 * x["rate"], 100 * x["ci"][0], 100 * x["ci"][1])


def write_markdown(r):
    d = r["decision"]
    L = ["# LLM-written phishing", ""]
    L.append("Generated by `experiments_llm.py`. Protocol, written before the first run: "
             "[`LLM_PHISHING_PROTOCOL.md`](LLM_PHISHING_PROTOCOL.md). The saved production model at its own "
             "threshold (%.3f); nothing is trained or tuned on these sets." % r["threshold"])
    L.append("")
    L.append("## Decision")
    L.append("")
    L.append("Zenodo recall **%.1f%%** against %.1f%% on hold-out C's human-written phishing (Fisher p = %.3g): "
             "**LLM-written phishing is %s for the model** under the pre-registered rule."
             % (100 * d["zenodo_recall"], 100 * d["holdout_c_recall"], d["fisher_p"],
                "harder" if d["llm_phishing_harder"] else "not harder"))
    L.append("")
    L.append("## Overall")
    L.append("")
    L.append("For the phishing sets *flagged* is the recall. Greco's \"legitimate\" set is a weak control: random "
             "samples read like phishing, so a flagged email there is not necessarily a false alarm.")
    L.append("")
    L.append("| set | emails | near-duplicates of training removed | flagged | median P(phishing) |")
    L.append("|---|---:|---:|---:|---:|")
    for name, e in r["sets"].items():
        L.append("| %s | %d | %d | %s | %.3f |" % (name, e["flagged"]["n"], e["near_duplicates_removed"],
                                               pct(e["flagged"]), e["median_probability"]))
    L.append("")
    zname = "Zenodo (GPT-4.1, DeepSeek 3.2, Llama 3.3)"
    e = r["sets"][zname]
    for key, title, chi in [("by_generator", "By generator (Zenodo)", "chi2_p_generator"),
                            ("by_theme", "By theme (Zenodo)", "chi2_p_theme"),
                            ("by_links", "By link type (Zenodo)", None)]:
        L.append("## " + title)
        L.append("")
        L.append("| group | emails | recall |")
        L.append("|---|---:|---:|")
        for g, x in e[key].items():
            L.append("| %s | %d | %s |" % (g, x["n"], pct(x)))
        if chi:
            L.append("")
            L.append("Chi-square test of independence: p = %.3g." % e[chi])
        L.append("")
    g = r["sets"]["Greco LLM phishing (ChatGPT, WormGPT)"]
    L.append("## By link type (Greco phishing)")
    L.append("")
    L.append("| group | emails | recall |")
    L.append("|---|---:|---:|")
    for k, x in g["by_links"].items():
        L.append("| %s | %d | %s |" % (k, x["n"], pct(x)))
    L.append("")
    for name in [zname, "Greco LLM phishing (ChatGPT, WormGPT)"]:
        L.append("## Lowest-scored phishing: " + name)
        L.append("")
        for x in r["sets"][name]["lowest_scored"]:
            L.append("- **%.3f** (%s, %s) %s" % (x["score"], x["generator"], x["theme"], x["text"].replace("|", "\\|")))
        L.append("")
    f = open(os.path.join(REPORTS, "LLM_PHISHING.md"), "w", encoding="utf-8")
    f.write("\n".join(L))
    f.close()


if __name__ == "__main__":
    sys.exit(main())
