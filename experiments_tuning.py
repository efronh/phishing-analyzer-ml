#!/usr/bin/env python3
# Yöntem deneyleri:
#   1. Yakın kopyalar: aynı model rastgele split ve yakın kopya gruplarını bölmeyen split
#      ile eğitilir - in-domain test skoru ne kadar şişiyordu?
#   2. Hiperparametreler: C x max_features x karakter n-gram (12 kombinasyon), gruplu
#      5-fold CV ile (train+val üzerinde; test dilimine ve hold-out'lara dokunulmaz).
#
# Seçim kuralı (sonuçlara bakmadan önce sabitlendi): en yüksek ortalama CV F1'i veren
# kombinasyon seçilir; ama mevcut ayar en iyinin 1 standart hatası içindeyse mevcut ayar
# korunur (gürültü yüzünden ayar değiştirmiyoruz). Hold-out'lar sadece bilgi için ölçülür.
#
# Kullanım: python prepare_data.py && python experiments_tuning.py   (~25 dk)

import itertools
import json
import os
import sys
import time

import numpy as np

from evaluation import (
    best_f1_threshold,
    compute_metrics,
    format_table,
    format_table_ci,
    load_csv,
    mcnemar_exact,
    near_duplicate_groups,
)
from ml_model import (
    DEFAULT_C,
    DEFAULT_CHAR_NGRAM,
    DEFAULT_MAX_FEATURES,
    build_pipeline,
    cv_oof_scores,
    fit_with_cv_threshold,
    fold_f1_curves,
    pipeline_scores,
)
from train import DEFAULT_DATA, split_indices

DATA = os.path.join("data", "processed")
REPORTS = "reports"
SEED = 42
GRID = {
    "C": [1.0, 4.0, 16.0],
    "max_features": [50000, 200000],
    "char_ngram": [(3, 5), (2, 5)],
}
DEFAULT = {"C": DEFAULT_C, "max_features": DEFAULT_MAX_FEATURES, "char_ngram": tuple(DEFAULT_CHAR_NGRAM)}
COLUMNS = ["precision", "recall", "f1", "fpr", "roc_auc"]


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + msg, flush=True)


def config_name(cfg):
    return "C=%g, max_features=%dk, char=%d-%d" % (cfg["C"], cfg["max_features"] // 1000,
                                                    cfg["char_ngram"][0], cfg["char_ngram"][1])


def main():
    texts, labels = [], []
    for path in DEFAULT_DATA:
        X, y, _ = load_csv(path)
        texts.extend(X)
        labels.extend(y.tolist())
    y = np.array(labels)
    pick = lambda ids: [texts[i] for i in ids]
    started = time.time()

    # 1) rastgele vs gruplu split - aynı model, aynı yöntem (train'de eğit, val'de threshold)
    groups = near_duplicate_groups(texts)
    split_rows = []
    for name, g in [("random split", None), ("near-duplicate groups kept together", groups)]:
        tr, va, te = split_indices(y, SEED, g)
        pipe = build_pipeline().fit(pick(tr), y[tr])
        th = best_f1_threshold(y[va], pipeline_scores(pipe, pick(va)))
        m = compute_metrics(y[te], pipeline_scores(pipe, pick(te)), th)
        split_rows.append((name, m))
        log(name + ": test F1=" + str(m["f1"]) + " fpr=" + str(m["fpr"]))

    # 2) hiperparametreler - gruplu split'in train+val'i üzerinde gruplu 5-fold CV
    tr, va, te = split_indices(y, SEED, groups)
    trva = np.concatenate([tr, va])
    Xtv, ytv, gtv = pick(trva), y[trva], groups[trva]
    grid = []
    for C, mf, ng in itertools.product(GRID["C"], GRID["max_features"], GRID["char_ngram"]):
        cfg = {"C": C, "max_features": mf, "char_ngram": ng}
        t = time.time()
        oof, fold = cv_oof_scores(Xtv, ytv, folds=5, seed=SEED, n_jobs=5, groups=gtv, **cfg)
        curves = fold_f1_curves(ytv, oof, fold)
        best = int(np.argmax(curves.mean(axis=0)))
        fold_f1 = curves[:, best]
        row = {"config": cfg, "name": config_name(cfg),
               "cv_f1_mean": round(float(fold_f1.mean()), 4),
               "cv_f1_se": round(float(fold_f1.std(ddof=1) / np.sqrt(len(fold_f1))), 4),
               "minutes": round((time.time() - t) / 60, 1)}
        grid.append(row)
        log(row["name"] + ": CV F1 " + str(row["cv_f1_mean"]) + " ± " + str(row["cv_f1_se"]))

    best_row = max(grid, key=lambda r: r["cv_f1_mean"])
    default_row = [r for r in grid if r["config"] == DEFAULT][0]
    keep_default = default_row["cv_f1_mean"] >= best_row["cv_f1_mean"] - best_row["cv_f1_se"]
    selected = default_row if keep_default else best_row
    log("best: " + best_row["name"] + " | selected: " + selected["name"]
        + (" (default kept, within 1 SE)" if keep_default else ""))

    # 3) bilgi için: seçilen ayar mevcut ayardan farklıysa ikisini hold-out'larda karşılaştır
    holdout = {}
    if not keep_default:
        Xa, ya, _ = load_csv(os.path.join(DATA, "holdout_2025.csv"))
        Xb, _, _ = load_csv(os.path.join(DATA, "holdout_b_2025.csv"))
        Xc, yc, _ = load_csv(os.path.join(DATA, "holdout_c.csv"))
        Xb_full = [x for x, l in zip(Xa, ya) if l == 1] + Xb
        yb = np.array([1] * int(ya.sum()) + [0] * len(Xb))
        sets = {"A": (Xa, ya), "B": (Xb_full, yb), "C": (Xc, yc)}
        preds = {}
        for name, cfg in [("default", DEFAULT), ("selected", selected["config"])]:
            model = fit_with_cv_threshold(Xtv, ytv, folds=5, seed=SEED, n_jobs=5, groups=gtv, **cfg)
            holdout[name] = {"test": compute_metrics(y[te], model.predict_proba(pick(te)), model.threshold)}
            preds[name] = {}
            for key, (X, yy) in sets.items():
                s = model.predict_proba(X)
                holdout[name][key] = compute_metrics(yy, s, model.threshold)
                preds[name][key] = s >= model.threshold
        holdout["significance"] = [
            dict(set=key, **mcnemar_exact(yy, preds["selected"][key], preds["default"][key]))
            for key, (_, yy) in sets.items()]

    results = {
        "seed": SEED,
        "near_duplicate_groups": {
            "docs": len(texts),
            "groups": int(len(set(groups.tolist()))),
            "docs_in_multi_member_groups": int(sum(
                c for c in np.unique(groups, return_counts=True)[1] if c > 1)),
        },
        "split_comparison": dict(split_rows),
        "grid": grid,
        "best": best_row["name"],
        "selected": selected["name"],
        "selected_config": {**selected["config"], "char_ngram": list(selected["config"]["char_ngram"])},
        "default_kept": keep_default,
        "holdout_comparison": holdout,
        "runtime_minutes": round((time.time() - started) / 60, 1),
    }
    for r in results["grid"]:
        r["config"] = {**r["config"], "char_ngram": list(r["config"]["char_ngram"])}
    os.makedirs(REPORTS, exist_ok=True)
    f = open(os.path.join(REPORTS, "tuning.json"), "w", encoding="utf-8")
    json.dump(results, f, indent=2)
    f.close()
    write_markdown(results, split_rows)
    log("Done in " + str(results["runtime_minutes"]) + " min -> reports/TUNING.md")
    return 0


def write_markdown(r, split_rows):
    nd = r["near_duplicate_groups"]
    L = []
    L.append("# Near-duplicates and hyperparameters")
    L.append("")
    L.append("Generated by `experiments_tuning.py`. The hold-out sets are not used for any choice "
             "made here.")
    L.append("")
    L.append("## 1. How much did near-duplicates inflate the in-domain score?")
    L.append("")
    L.append("Emails with TF-IDF cosine similarity ≥ 0.8 are put in one group: replies quoting each "
             "other, or the same campaign with a new ID. **%d of %d training emails (%.0f%%)** have at "
             "least one near-copy. With a random split, copies of the same email can land in both "
             "train and test." % (nd["docs_in_multi_member_groups"], nd["docs"],
                                  100 * nd["docs_in_multi_member_groups"] / nd["docs"]))
    L.append("")
    L.append(format_table(split_rows, COLUMNS))
    L.append("")
    L.append("## 2. Hyperparameter search (grouped 5-fold CV on train + validation)")
    L.append("")
    L.append("Selection rule, fixed before running: take the highest mean CV F1, but keep the "
             "current setting if it is within one standard error of the best.")
    L.append("")
    L.append("| setting | CV F1 (mean ± SE) | minutes |")
    L.append("|---|---:|---:|")
    for g in sorted(r["grid"], key=lambda g: -g["cv_f1_mean"]):
        mark = " **(current)**" if g["name"] == r["selected"] and r["default_kept"] else ""
        if g["name"] == r["best"]:
            mark = mark + " ← best"
        L.append("| %s%s | %.4f ± %.4f | %.1f |" % (g["name"], mark, g["cv_f1_mean"], g["cv_f1_se"], g["minutes"]))
    L.append("")
    if r["default_kept"]:
        L.append("**Result: the current setting is kept.** It is within one standard error of the best, "
                 "so switching would be tuning to noise.")
    else:
        L.append("**Selected: " + r["selected"] + ".**")
        h = r["holdout_comparison"]
        L.append("")
        L.append("For information only (not used for selection), current vs selected on the hold-outs:")
        L.append("")
        for key in ["test", "A", "B", "C"]:
            L.append("**" + ("in-domain test" if key == "test" else "hold-out " + key) + "**")
            L.append("")
            L.append(format_table_ci([("current", h["default"][key]), ("selected", h["selected"][key])]))
            L.append("")
        L.append("| set | only selected correct | only current correct | McNemar p |")
        L.append("|---|---:|---:|---:|")
        for c in h["significance"]:
            L.append("| %s | %d | %d | %.3g |" % (c["set"], c["only_a_correct"], c["only_b_correct"], c["p_value"]))
    L.append("")
    f = open(os.path.join(REPORTS, "TUNING.md"), "w", encoding="utf-8")
    f.write("\n".join(L))
    f.close()


if __name__ == "__main__":
    sys.exit(main())
