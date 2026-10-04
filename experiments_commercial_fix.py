#!/usr/bin/env python3
# Ticari maildeki yanlış alarmı düzeltme deneyi. Protokol (sonuçlardan önce yazıldı):
# reports/COMMERCIAL_FIX_PROTOCOL.md
#   P0: kayıtlı üretim modeli
#   P1: Kaggle'sız (Nazario 2019-24 + SpamAssassin ham + 2024 mail listeleri)
#   P2: P1 + Kaggle'ın sadece meşru mailleri
#   P3: üretim verisi + yazarın reklam mailleri (gönderene göre fold; SADECE teşhis, özel veri)
# Gizlilik: experiments_commercial.py ile aynı - hiçbir mail içeriği yazdırılmaz / raporlanmaz.
#
# Kullanım: python experiments_commercial_fix.py

import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import GroupKFold

from evaluation import compute_metrics, format_table_ci, load_csv, mcnemar_exact, near_duplicate_groups, wilson_ci
from experiments_commercial import load_promo, load_redactor, promo_mbox_paths
from experiments_url import load_holdouts
from ml_model import PRODUCTION_FEATURE_VERSION, PhishingClassifier, fit_with_cv_threshold
from train import DEFAULT_DATA

DATA = os.path.join("data", "processed")
REPORTS = "reports"
SEED = 42
FOLDS = 5
ELIGIBLE = ["P1", "P2"]


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + msg, flush=True)


def csv_rows(name, label=None):
    X, y, _ = load_csv(os.path.join(DATA, name))
    if label is None:
        return X, y
    keep = y == label
    return [t for t, k in zip(X, keep) if k], y[keep]


def concat(*parts):
    return sum((p[0] for p in parts), []), np.concatenate([p[1] for p in parts])


def fit(texts, labels):
    return fit_with_cv_threshold(texts, labels, folds=FOLDS, seed=SEED, n_jobs=FOLDS,
                                 groups=near_duplicate_groups(texts), feature_version=PRODUCTION_FEATURE_VERSION)


def rate(pred):
    k, n = int(pred.sum()), len(pred)
    return {"false_alarms": k, "n": n, "rate": round(k / n, 4) if n else None, "ci": wilson_ci(k, n)}


def main():
    if not promo_mbox_paths():
        print("No finished .mbox export in data/raw/own_promo/", file=sys.stderr)
        return 1
    started = time.time()
    redactor, _ = load_redactor()
    promo, counts = load_promo(redactor)
    Xp = [r["text"] for r in promo]
    english = np.array([r["english"] for r in promo])
    senders = np.array([r["sender"] for r in promo])
    holdouts = load_holdouts()
    log("commercial set: " + str(len(Xp)) + " emails, " + str(int(english.sum())) + " English")

    production = concat(*[csv_rows(os.path.basename(p)) for p in DEFAULT_DATA])
    no_kaggle = concat(csv_rows("nazario_sa.csv"), csv_rows("modern_legit.csv"), csv_rows("nazario_older.csv"))
    kaggle_legit = csv_rows("kaggle.csv", label=0)
    data = {"P1": no_kaggle, "P2": concat(no_kaggle, kaggle_legit)}

    # her aday için: ticari maillerde karar + hold-out'larda karar
    commercial_pred, holdout_prob, holdout_pred, info = {}, {}, {}, {}

    def score_holdouts(name, model):
        holdout_prob[name], holdout_pred[name] = {}, {}
        for key, (X, yy) in holdouts.items():
            p = model.predict_proba(X)
            holdout_prob[name][key] = (p, model.threshold)
            holdout_pred[name][key] = p >= model.threshold

    log("P0: saved production model")
    p0 = PhishingClassifier.load("model.joblib")
    commercial_pred["P0"] = p0.predict_proba(Xp) >= p0.threshold
    score_holdouts("P0", p0)
    info["P0"] = {"n_train": len(production[1]), "threshold": p0.threshold}

    for name in ["P1", "P2"]:
        X, y = data[name]
        log(name + ": training on " + str(len(y)) + " emails (" + str(int(y.sum())) + " phishing)")
        m = fit(X, y)
        commercial_pred[name] = m.predict_proba(Xp) >= m.threshold
        score_holdouts(name, m)
        info[name] = {"n_train": len(y), "phishing_share": round(float(y.mean()), 4), "threshold": m.threshold}

    # P3: gönderene göre 5 fold - her reklam maili, o göndereni hiç görmemiş bir modelle puanlanır
    pred3 = np.zeros(len(Xp), dtype=bool)
    thresholds = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=FOLDS).split(Xp, groups=senders)):
        X, y = concat(production, ([Xp[i] for i in tr], np.zeros(len(tr), dtype=int)))
        log("P3 fold " + str(k + 1) + "/" + str(FOLDS) + ": " + str(len(set(senders[tr]))) + " senders in training")
        m = fit(X, y)
        pred3[te] = m.predict_proba([Xp[i] for i in te]) >= m.threshold
        thresholds.append(m.threshold)
    commercial_pred["P3"] = pred3
    log("P3: model with all promotions, for the hold-outs")
    X, y = concat(production, (Xp, np.zeros(len(Xp), dtype=int)))
    m = fit(X, y)
    score_holdouts("P3", m)
    info["P3"] = {"n_train": len(y), "threshold_folds": thresholds, "threshold_all": m.threshold}

    # ---- sonuçlar
    zeros = np.zeros(len(Xp), dtype=int)
    commercial = {}
    for name, pred in commercial_pred.items():
        row = {"English": rate(pred[english]), "not English": rate(pred[~english]), "all": rate(pred)}
        if name != "P0":
            for subset, mask in [("English", english), ("all", np.ones(len(Xp), bool))]:
                r = mcnemar_exact(zeros[mask], pred[mask], commercial_pred["P0"][mask])
                row["mcnemar_" + subset] = r
        commercial[name] = row

    hold = {}
    tests = {}
    for name in commercial_pred:
        hold[name] = {key: compute_metrics(yy, *holdout_prob[name][key]) for key, (X, yy) in holdouts.items()}
        if name == "P0":
            continue
        t = {}
        for key, (X, yy) in holdouts.items():
            new, old = holdout_pred[name][key], holdout_pred["P0"][key]
            phish, legit = yy == 1, yy == 0
            # mcnemar_exact(y, a, b): only_a_correct = sadece aday doğru, only_b_correct = sadece P0 doğru
            t[key] = {"phishing": mcnemar_exact(yy[phish], new[phish], old[phish]),
                      "legit": mcnemar_exact(yy[legit], new[legit], old[legit])}
        tests[name] = t

    def worse(r):
        return r["only_b_correct"] > r["only_a_correct"] and r["p_value"] < 0.05

    decision = {}
    for name in ELIGIBLE:
        c = commercial[name]["mcnemar_English"]
        crit1 = c["only_a_correct"] > c["only_b_correct"] and c["p_value"] < 0.05
        # A ve B aynı 454 phishing'i paylaşıyor: recall testi A'nın phishing kısmında
        crit2 = not worse(tests[name]["A"]["phishing"])
        crit3 = not any(worse(tests[name][k]["legit"]) for k in ["A", "B", "C"])
        decision[name] = {"commercial_english_lower": crit1, "recall_2025_not_lower": crit2,
                          "holdout_false_alarms_not_higher": crit3, "qualifies": bool(crit1 and crit2 and crit3)}
    qualified = [n for n in ELIGIBLE if decision[n]["qualifies"]]
    winner = min(qualified, key=lambda n: commercial[n]["all"]["rate"]) if qualified else None
    log("decision: " + json.dumps({n: d["qualifies"] for n, d in decision.items()}) + " -> "
        + (winner or "production stays"))

    results = {"seed": SEED, "commercial_counts": counts, "info": info, "commercial": commercial,
               "holdouts": hold, "tests_vs_P0": tests, "decision": decision, "winner": winner,
               "runtime_seconds": round(time.time() - started, 1)}
    f = open(os.path.join(REPORTS, "commercial_fix.json"), "w", encoding="utf-8")
    json.dump(results, f, indent=2, default=float)
    f.close()
    write_markdown(results)
    log("Done in " + str(results["runtime_seconds"]) + "s -> reports/COMMERCIAL_FIX.md")
    return 0


NAMES = {"P0": "P0: production", "P1": "P1: no Kaggle", "P2": "P2: Kaggle legit only",
         "P3": "P3: + promotions (diagnostic)"}


def pct(m):
    return "-" if m["rate"] is None else "%.1f%% (%.1f–%.1f)" % (100 * m["rate"], 100 * m["ci"][0], 100 * m["ci"][1])


def write_markdown(r):
    L = ["# Fixing the commercial false alarms", ""]
    L.append("Generated by `experiments_commercial_fix.py`. Protocol, written before the first run: "
             "[`COMMERCIAL_FIX_PROTOCOL.md`](COMMERCIAL_FIX_PROTOCOL.md). The commercial set is private "
             "(see [`COMMERCIAL.md`](COMMERCIAL.md)); this report holds counts only.")
    L.append("")
    L.append("## Decision")
    L.append("")
    if r["winner"]:
        L.append("**" + NAMES[r["winner"]] + " qualifies under the rule and replaces production.** The commercial "
                 "set was used to find the problem, so a fresh commercial test is still needed before calling "
                 "it fixed.")
    else:
        L.append("**No eligible candidate qualifies; production stays as it is.**")
    L.append("")
    L.append("| candidate | English promotions lower (p < 0.05) | 2025 recall not lower | hold-out false alarms "
             "not higher | qualifies |")
    L.append("|---|---|---|---|---|")
    yn = lambda b: "yes" if b else "**no**"
    for n, d in r["decision"].items():
        L.append("| %s | %s | %s | %s | %s |" % (NAMES[n], yn(d["commercial_english_lower"]),
                                                 yn(d["recall_2025_not_lower"]),
                                                 yn(d["holdout_false_alarms_not_higher"]),
                                                 "**yes**" if d["qualifies"] else "no"))
    L.append("")
    L.append("## Commercial false alarms")
    L.append("")
    L.append("P3's promotions are scored by models that never saw their sender (5 folds by sender domain).")
    L.append("")
    L.append("| candidate | training emails | English | not English | all | McNemar vs P0 (English) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for n, c in r["commercial"].items():
        mc = c.get("mcnemar_English")
        L.append("| %s | %d | %s | %s | %s | %s |" % (NAMES[n], r["info"][n]["n_train"], pct(c["English"]),
                                                     pct(c["not English"]), pct(c["all"]),
                                                     "-" if not mc else "%.3g" % mc["p_value"]))
    L.append("")
    for key, title in [("A", "Hold-out A: Nazario 2025 + Apache 2025"), ("B", "Hold-out B: Nazario 2025 + Ubuntu 2025"),
                       ("C", "Hold-out C: Phishing Pot 2026 + OSGeo 2025 (recall reported, not decisive)")]:
        L.append("## " + title)
        L.append("")
        L.append(format_table_ci([(NAMES[n], h[key]) for n, h in r["holdouts"].items()]))
        L.append("")
    L.append("## McNemar against P0 on the hold-outs")
    L.append("")
    L.append("| candidate | set | part | only candidate correct | only P0 correct | p |")
    L.append("|---|---|---|---:|---:|---:|")
    for n, t in r["tests_vs_P0"].items():
        for key, parts in t.items():
            for part, x in parts.items():
                L.append("| %s | %s | %s | %d | %d | %.3g |" % (NAMES[n], key, part, x["only_a_correct"],
                                                              x["only_b_correct"], x["p_value"]))
    L.append("")
    f = open(os.path.join(REPORTS, "COMMERCIAL_FIX.md"), "w", encoding="utf-8")
    f.write("\n".join(L))
    f.close()


if __name__ == "__main__":
    sys.exit(main())
