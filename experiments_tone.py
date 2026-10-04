#!/usr/bin/env python3
# "Ton" deneyi: model kurumsal/kibar yazım tonunu phishing sanıyor. Genel kelimeleri (stop words)
# ve/veya harf grubu (char n-gram) bloğunu çıkarmak bunu azaltır mı, yakalamayı bozmadan?
# Protokol (sonuçlardan önce yazıldı): reports/TONE_PROTOCOL.md
# Üretim modeli DEĞİŞMEZ: kazanan olursa sadece araştırma gelen kutusundaki kör test için aday olur.
#
# Kullanım: python experiments_tone.py   (reklam maili exportu ve ./download_llm.sh gerekir)

import json
import os
import sys
import time

import numpy as np

from evaluation import compute_metrics, format_table_ci, load_csv, mcnemar_exact, near_duplicate_groups, wilson_ci
from experiments_commercial import load_promo, load_redactor, promo_mbox_paths
from experiments_llm import load_greco, load_zenodo
from experiments_url import load_holdouts
from ml_model import PRODUCTION_FEATURE_VERSION, PhishingClassifier, fit_with_cv_threshold
from train import DEFAULT_DATA

REPORTS = "reports"
SEED = 42
FOLDS = 5
VARIANTS = {"M1": {"stop_words": "english"}, "M2": {"use_char": False},
            "M3": {"stop_words": "english", "use_char": False}}
NAMES = {"P0": "P0: production", "M1": "M1: stop words removed", "M2": "M2: no character n-grams",
         "M3": "M3: both"}


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + msg, flush=True)


def rate(pred):
    k, n = int(pred.sum()), len(pred)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci": wilson_ci(k, n)}


def main():
    if not promo_mbox_paths():
        print("No finished .mbox export in data/raw/own_promo/", file=sys.stderr)
        return 1
    started = time.time()
    redactor, _ = load_redactor()
    promo, _ = load_promo(redactor)
    Xp = [r["text"] for r in promo]
    english = np.array([r["english"] for r in promo])
    holdouts = load_holdouts()
    zenodo = [r["text"] for r in load_zenodo()]
    f = open(os.path.join(REPORTS, "llm_legit_audit.json"), encoding="utf-8")
    audit = json.load(f)
    f.close()
    greco = load_greco("legit.csv")
    benign = [greco[x["row"]]["text"] for x in audit["labels"] if x["label"] == "benign"]

    texts, labels = [], []
    for p in DEFAULT_DATA:
        X, y, _ = load_csv(p)
        texts.extend(X)
        labels.append(y)
    labels = np.concatenate(labels)
    groups = near_duplicate_groups(texts)

    models = {"P0": PhishingClassifier.load("model.joblib")}
    for name, kw in VARIANTS.items():
        log(name + ": training " + json.dumps(kw))
        models[name] = fit_with_cv_threshold(texts, labels, folds=FOLDS, seed=SEED, n_jobs=FOLDS, groups=groups,
                                             feature_version=PRODUCTION_FEATURE_VERSION, **kw)

    pred, hold, info = {}, {}, {}
    for name, m in models.items():
        log("scoring " + name)
        pred[name] = {"promo": m.predict_proba(Xp) >= m.threshold,
                      "zenodo": m.predict_proba(zenodo) >= m.threshold,
                      "benign": m.predict_proba(benign) >= m.threshold}
        hold[name] = {}
        for key, (X, yy) in holdouts.items():
            p = m.predict_proba(X)
            hold[name][key] = compute_metrics(yy, p, m.threshold)
            pred[name][key] = p >= m.threshold
        info[name] = {"threshold": m.threshold}

    results = {"info": info, "commercial": {}, "holdouts": hold, "zenodo": {}, "benign_audit": {}, "tests": {}}
    zeros = np.zeros(len(Xp), dtype=int)
    for name in models:
        pp = pred[name]["promo"]
        results["commercial"][name] = {"English": rate(pp[english]), "not English": rate(pp[~english]), "all": rate(pp)}
        results["zenodo"][name] = rate(pred[name]["zenodo"])
        results["benign_audit"][name] = rate(pred[name]["benign"])
        if name == "P0":
            continue
        t = {"promo_english": mcnemar_exact(zeros[english], pp[english], pred["P0"]["promo"][english]),
             "zenodo": mcnemar_exact(np.ones(len(zenodo), int), pred[name]["zenodo"], pred["P0"]["zenodo"])}
        for key, (X, yy) in holdouts.items():
            ph, lg = yy == 1, yy == 0
            t[key + "_phishing"] = mcnemar_exact(yy[ph], pred[name][key][ph], pred["P0"][key][ph])
            t[key + "_legit"] = mcnemar_exact(yy[lg], pred[name][key][lg], pred["P0"][key][lg])
        results["tests"][name] = t

    # mcnemar_exact(y, a, b): only_a_correct = sadece aday doğru, only_b_correct = sadece P0 doğru
    def worse(r):
        return r["only_b_correct"] > r["only_a_correct"] and r["p_value"] < 0.05

    decision = {}
    for name in VARIANTS:
        t = results["tests"][name]
        e = t["promo_english"]
        crit = {"english_promotions_lower": e["only_a_correct"] > e["only_b_correct"] and e["p_value"] < 0.05,
                "nazario_2025_recall_not_lower": not worse(t["A_phishing"]),
                "c_recall_not_lower": not worse(t["C_phishing"]),
                "holdout_false_alarms_not_higher": not any(worse(t[k + "_legit"]) for k in ["A", "B", "C"]),
                "zenodo_recall_not_lower": not worse(t["zenodo"])}
        decision[name] = dict(crit, qualifies=all(crit.values()))
    qualified = [n for n in VARIANTS if decision[n]["qualifies"]]
    results["decision"] = decision
    results["candidate"] = (min(qualified, key=lambda n: results["commercial"][n]["English"]["rate"])
                            if qualified else None)
    results["runtime_seconds"] = round(time.time() - started, 1)
    log("decision: " + json.dumps({n: d["qualifies"] for n, d in decision.items()}) + " -> candidate "
        + str(results["candidate"]))
    f = open(os.path.join(REPORTS, "tone.json"), "w", encoding="utf-8")
    json.dump(results, f, indent=2, default=float)
    f.close()
    write_markdown(results)
    log("Done in " + str(results["runtime_seconds"]) + "s -> reports/TONE.md")
    return 0


def pct(x):
    return "-" if x["rate"] is None else "%.1f%% (%.1f–%.1f)" % (100 * x["rate"], 100 * x["ci"][0], 100 * x["ci"][1])


def write_markdown(r):
    L = ["# Less reliance on tone", ""]
    L.append("Generated by `experiments_tone.py`. Protocol, written before the first run: "
             "[`TONE_PROTOCOL.md`](TONE_PROTOCOL.md). Production is not changed by this experiment: a winner only "
             "becomes the candidate for the blind research-inbox test.")
    L.append("")
    L.append("## Decision")
    L.append("")
    L.append("**" + ("Candidate for the blind test: " + NAMES[r["candidate"]] + "." if r["candidate"]
                     else "No variant qualifies.") + "**")
    L.append("")
    yn = lambda b: "yes" if b else "**no**"
    L.append("| variant | English promotions lower | 2025 recall not lower | C recall not lower | hold-out false "
             "alarms not higher | LLM recall not lower | qualifies |")
    L.append("|---|---|---|---|---|---|---|")
    for n, d in r["decision"].items():
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            NAMES[n], yn(d["english_promotions_lower"]), yn(d["nazario_2025_recall_not_lower"]),
            yn(d["c_recall_not_lower"]), yn(d["holdout_false_alarms_not_higher"]), yn(d["zenodo_recall_not_lower"]),
            "**yes**" if d["qualifies"] else "no"))
    L.append("")
    L.append("## Promotions, LLM-written phishing and the benign audit")
    L.append("")
    L.append("| variant | threshold | English promotions flagged | all promotions flagged | Zenodo LLM phishing "
             "recall | benign LLM emails flagged (audit) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for n in r["commercial"]:
        c = r["commercial"][n]
        b = r["benign_audit"][n]
        L.append("| %s | %.3f | %s | %s | %s | %d of %d |" % (NAMES[n], r["info"][n]["threshold"], pct(c["English"]),
                                                          pct(c["all"]), pct(r["zenodo"][n]), b["k"], b["n"]))
    L.append("")
    for key, title in [("A", "Hold-out A"), ("B", "Hold-out B"), ("C", "Hold-out C")]:
        L.append("## " + title)
        L.append("")
        L.append(format_table_ci([(NAMES[n], h[key]) for n, h in r["holdouts"].items()]))
        L.append("")
    L.append("## McNemar against P0")
    L.append("")
    L.append("| variant | comparison | only variant correct | only P0 correct | p |")
    L.append("|---|---|---:|---:|---:|")
    for n, t in r["tests"].items():
        for k, x in t.items():
            L.append("| %s | %s | %d | %d | %.3g |" % (NAMES[n], k, x["only_a_correct"], x["only_b_correct"],
                                                     x["p_value"]))
    L.append("")
    f = open(os.path.join(REPORTS, "TONE.md"), "w", encoding="utf-8")
    f.write("\n".join(L))
    f.close()


if __name__ == "__main__":
    sys.exit(main())
