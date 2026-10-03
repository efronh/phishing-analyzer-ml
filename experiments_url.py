#!/usr/bin/env python3
# URL feature deneyi: v3 (üretim) ile v4 (v3 + 8 URL feature'ı) aynı prosedürle karşılaştırılır.
# Karar kuralı sonuçlardan önce yazıldı: reports/URL_FEATURES_PROTOCOL.md
#   1. Gruplu 5-fold CV'de (aynı fold'lar) eşleştirilmiş fold F1 farkının ortalaması > standart hatası
#   2. A, B, C'nin hiçbirinde meşru maillerde yanlış alarm anlamlı artmıyor (McNemar, p < 0.05)
# Lockbox'a dokunulmuyor.
#
# Kullanım: python prepare_data.py && python experiments_url.py

import json
import os
import sys
import time

import numpy as np

from evaluation import compute_metrics, format_table_ci, load_csv, mcnemar_exact, near_duplicate_groups
from features import URL_FEATURE_NAMES, _url_features
from ml_model import (PlattCalibrator, build_pipeline, cv_oof_scores, f1_curve, fold_f1_curves,
                      select_threshold, THRESHOLD_GRID)
from train import DEFAULT_DATA

DATA = os.path.join("data", "processed")
REPORTS = "reports"
SEED = 42
FOLDS = 5
VERSIONS = [("v3", 3), ("v4", 4)]


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + msg, flush=True)


def load_training():
    texts, labels, sources = [], [], []
    for path in DEFAULT_DATA:
        X, y, s = load_csv(path)
        texts.extend(X)
        labels.append(y)
        name = os.path.basename(path)[:-4]
        # kaynak bazında feature oranı tablosu için; nazario_sa içinde SpamAssassin ayrı
        sources.extend(src.split("_")[0] if name == "nazario_sa" else name for src in s)
    return texts, np.concatenate(labels), sources


def load_holdouts():
    Xa, ya, _ = load_csv(os.path.join(DATA, "holdout_2025.csv"))
    Xb_legit, _, _ = load_csv(os.path.join(DATA, "holdout_b_2025.csv"))
    Xc, yc, _ = load_csv(os.path.join(DATA, "holdout_c.csv"))
    phish = [x for x, l in zip(Xa, ya) if l == 1]
    Xb = phish + Xb_legit
    yb = np.array([1] * len(phish) + [0] * len(Xb_legit))
    return {"A": (Xa, ya), "B": (Xb, yb), "C": (Xc, yc)}


def feature_rates(texts, labels, sources):
    """Eğitim kaynaklarında her URL feature'ının sıfırdan farklı olma oranı (linkli maillerde)."""
    groups = {}
    for t, l, s in zip(texts, labels, sources):
        f = _url_features(t)
        if not any(f):
            continue
        key = s + (" (phishing)" if l == 1 else " (legit)")
        groups.setdefault(key, []).append([1.0 if v > 0 else 0.0 for v in f])
    return {k: {"n_with_url": len(v), "rates": dict(zip(URL_FEATURE_NAMES,
                                                         np.round(np.mean(v, axis=0), 4).tolist()))}
            for k, v in sorted(groups.items())}


def main():
    for p in DEFAULT_DATA + [os.path.join(DATA, f) for f in
                             ["holdout_2025.csv", "holdout_b_2025.csv", "holdout_c.csv"]]:
        if not os.path.exists(p):
            print("Missing " + p + " - run download_data.sh and prepare_data.py first.", file=sys.stderr)
            return 1
    started = time.time()
    texts, y, sources = load_training()
    holdouts = load_holdouts()
    log("training emails: " + str(len(texts)) + ", near-duplicate groups...")
    groups = near_duplicate_groups(texts)

    res = {}
    for name, fv in VERSIONS:
        log("CV " + name)
        oof, fold = cv_oof_scores(texts, y, FOLDS, SEED, n_jobs=FOLDS, groups=groups, feature_version=fv)
        cal = PlattCalibrator().fit(oof, y)
        scores = cal.transform(oof)
        th, info = select_threshold(fold_f1_curves(y, scores, fold), "plateau")
        k = int(np.argmin(np.abs(THRESHOLD_GRID - th)))
        fold_f1 = [float(f1_curve(y[fold == f], scores[fold == f])[k]) for f in range(FOLDS)]
        log("  " + name + ": threshold " + str(th) + ", fold F1 " + str(np.round(fold_f1, 4).tolist()))
        log("refit " + name)
        pipe = build_pipeline(feature_version=fv).fit(texts, y)
        hold = {}
        for key, (X, yy) in holdouts.items():
            p = cal.transform(pipe.predict_proba(X)[:, 1])
            hold[key] = {"metrics": compute_metrics(yy, p, th), "pred": p >= th}
        res[name] = {"threshold": th, "threshold_cv": info, "fold_f1": fold_f1,
                     "cv_metrics": compute_metrics(y, scores, th), "holdout": hold}

    # 1. birincil: eşleştirilmiş fold F1 farkı
    diff = np.array(res["v4"]["fold_f1"]) - np.array(res["v3"]["fold_f1"])
    mean_diff = float(diff.mean())
    se_diff = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    primary = mean_diff > se_diff

    # 2. koruma: meşru maillerde yanlış alarm (sadece label 0), ve tam set için bilgi amaçlı McNemar
    guard = {}
    significance = []
    for key, (X, yy) in holdouts.items():
        legit = yy == 0
        new, old = res["v4"]["holdout"][key]["pred"], res["v3"]["holdout"][key]["pred"]
        r = mcnemar_exact(yy[legit], new[legit], old[legit])
        # only_a_correct = sadece v4 doğru (v3 yanlış alarm), only_b_correct = sadece v3 doğru
        worse = r["only_b_correct"] > r["only_a_correct"] and r["p_value"] < 0.05
        guard[key] = {"only_v4_false_alarm": r["only_b_correct"], "only_v3_false_alarm": r["only_a_correct"],
                      "p_value": r["p_value"], "v4_significantly_worse": worse}
        full = mcnemar_exact(yy, new, old)
        significance.append({"set": key, "only_v4_correct": full["only_a_correct"],
                             "only_v3_correct": full["only_b_correct"], "p_value": full["p_value"]})
    guard_ok = not any(g["v4_significantly_worse"] for g in guard.values())
    adopt = bool(primary and guard_ok)
    log("primary: mean dF1 %.5f vs SE %.5f -> %s; guard %s -> %s"
        % (mean_diff, se_diff, primary, guard_ok, "ADOPT v4" if adopt else "keep v3"))

    results = {
        "seed": SEED, "folds": FOLDS, "n_train": len(texts),
        "decision": {"mean_fold_f1_diff": round(mean_diff, 5), "se": round(se_diff, 5),
                     "primary_passed": bool(primary), "guard": guard, "guard_passed": guard_ok,
                     "adopt_v4": adopt},
        "versions": {n: {"threshold": r["threshold"], "threshold_cv": r["threshold_cv"],
                         "fold_f1": [round(v, 5) for v in r["fold_f1"]], "cv_metrics": r["cv_metrics"],
                         "holdout": {k: v["metrics"] for k, v in r["holdout"].items()}}
                     for n, r in res.items()},
        "significance_full_sets": significance,
        "feature_rates_training": feature_rates(texts, y, sources),
    }
    results["runtime_seconds"] = round(time.time() - started, 1)
    os.makedirs(REPORTS, exist_ok=True)
    f = open(os.path.join(REPORTS, "url_features.json"), "w", encoding="utf-8")
    json.dump(results, f, indent=2, default=float)
    f.close()
    write_markdown(results)
    log("Done in " + str(results["runtime_seconds"]) + "s -> reports/URL_FEATURES.md")
    return 0


def write_markdown(r):
    d = r["decision"]
    v = r["versions"]
    L = ["# URL features", ""]
    L.append("Generated by `experiments_url.py`. Protocol, written before the first run: "
             "[`URL_FEATURES_PROTOCOL.md`](URL_FEATURES_PROTOCOL.md). v3 is the production feature set; "
             "v4 adds 8 bounded URL features. Both use the production procedure (grouped " + str(r["folds"])
             + "-fold CV on " + str(r["n_train"]) + " training emails, Platt calibration, plateau threshold, "
             "refit on all data).")
    L.append("")
    L.append("## Decision")
    L.append("")
    L.append("**" + ("v4 is adopted." if d["adopt_v4"] else "v3 stays in production.") + "**")
    L.append("")
    L.append("1. Primary, mean paired fold-F1 difference (v4 − v3): **%+.5f**, standard error %.5f → %s."
             % (d["mean_fold_f1_diff"], d["se"], "passed" if d["primary_passed"] else "not passed"))
    L.append("2. Guard, no significant rise in false alarms on A, B or C → %s."
             % ("passed" if d["guard_passed"] else "not passed"))
    L.append("")
    L.append("| set | false alarms only v4 makes | only v3 makes | McNemar p | v4 significantly worse |")
    L.append("|---|---:|---:|---:|---|")
    for k, g in d["guard"].items():
        L.append("| %s | %d | %d | %.3g | %s |" % (k, g["only_v4_false_alarm"], g["only_v3_false_alarm"],
                                                 g["p_value"], "yes" if g["v4_significantly_worse"] else "no"))
    L.append("")
    L.append("## Cross-validation (training data)")
    L.append("")
    L.append("| version | threshold | fold F1 | CV F1 (all folds) |")
    L.append("|---|---:|---|---:|")
    for n, x in v.items():
        L.append("| %s | %.3f | %s | %.4f |" % (n, x["threshold"], ", ".join("%.4f" % f for f in x["fold_f1"]),
                                              x["cv_metrics"]["f1"]))
    L.append("")
    for k, title in [("A", "Hold-out A: Nazario 2025 + Apache 2025"), ("B", "Hold-out B: Nazario 2025 + Ubuntu 2025"),
                     ("C", "Hold-out C: Phishing Pot 2026 + OSGeo 2025")]:
        L.append("## " + title)
        L.append("")
        L.append(format_table_ci([(n, x["holdout"][k]) for n, x in v.items()]))
        L.append("")
    L.append("Whole-set McNemar (recall and false alarms together):")
    L.append("")
    L.append("| set | only v4 correct | only v3 correct | p |")
    L.append("|---|---:|---:|---:|")
    for s in r["significance_full_sets"]:
        L.append("| %s | %d | %d | %.3g |" % (s["set"], s["only_v4_correct"], s["only_v3_correct"], s["p_value"]))
    L.append("")
    L.append("## How often each URL feature fires (training data, emails with a link)")
    L.append("")
    names = URL_FEATURE_NAMES
    L.append("| source | emails with a link | " + " | ".join("`" + n[4:] + "`" for n in names) + " |")
    L.append("|---|---:|" + "---:|" * len(names))
    for src, x in r["feature_rates_training"].items():
        L.append("| " + src + " | " + str(x["n_with_url"]) + " | "
                 + " | ".join("%.1f%%" % (100 * x["rates"][n]) for n in names) + " |")
    L.append("")
    L.append("`distinct_domains_log` fires on every email with a link; it is a count, not a flag.")
    L.append("")
    L.append("**Blind spot:** all legitimate test mail is from open-source mailing lists. Commercial mail "
             "uses click trackers (redirects, percent-encoding, deep subdomains) and free hosting much more "
             "often; the false-alarm cost of these features there is not measured.")
    L.append("")
    f = open(os.path.join(REPORTS, "URL_FEATURES.md"), "w", encoding="utf-8")
    f.write("\n".join(L))
    f.close()


if __name__ == "__main__":
    sys.exit(main())
