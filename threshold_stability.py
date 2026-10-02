#!/usr/bin/env python3
# Threshold seçim yöntemlerini karşılaştırır. Üretim konfigürasyonu (train.py) 5 farklı
# seed'le eğitilir:
#   val       : train'de eğit, threshold'u tek validation diliminde (%15) seç
#   cv-max    : train+val üzerinde gruplu 5-fold CV, kalibre skorlarda ortalama F1'in zirvesi
#   cv-plateau: aynı CV, zirveye bitişik düz tepenin (zirve - 1 standart hata) ortası
# cv-max ve cv-plateau aynı fold eğitimlerini ve aynı yeniden eğitilmiş modeli paylaşır;
# aralarındaki fark sadece seçim kuralıdır.
# Başarı ölçütleri (önceden sabitlendi): plato kuralı threshold'un seed'ler arası
# standart sapmasını cv-max'a göre düşürmeli; hold-out sonuçlarının oynaklığı artmamalı.
# Hold-out setleri sadece ölçülür, hiçbir seçimde kullanılmaz. Model kaydetmez.
#
# Kullanım: python threshold_stability.py   (~15 dk)

import json
import os
import sys
import time

import numpy as np

from evaluation import compute_metrics, load_csv, near_duplicate_groups
from ml_model import (
    PRODUCTION_FEATURE_VERSION,
    PlattCalibrator,
    build_pipeline,
    cv_oof_scores,
    fold_f1_curves,
    pipeline_scores,
    select_threshold,
)
from train import DEFAULT_DATA, fit_model, split_indices

DATA = os.path.join("data", "processed")
REPORTS = "reports"
SEEDS = [42, 1, 2, 3, 4]
METHODS = ["val", "cv-max", "cv-plateau"]
KEYS = ["threshold", "test_f1", "nazario_2025_recall", "a_fpr", "b_fpr", "c_recall", "c_fpr"]
LABELS = {
    "threshold": "threshold",
    "test_f1": "in-domain test F1",
    "nazario_2025_recall": "Nazario 2025 recall",
    "a_fpr": "false alarms A (Apache)",
    "b_fpr": "false alarms B (Ubuntu)",
    "c_recall": "recall C (Phishing Pot)",
    "c_fpr": "false alarms C (OSGeo)",
}


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + msg, flush=True)


def evaluate(score, th, seed, test, holdouts):
    """score: metin listesi -> skor (üretimdeki gibi kalibre edilmiş olabilir)."""
    (Xt, yt), (Xa, ya), Xb, (Xc, yc) = test, holdouts[0], holdouts[1], holdouts[2]
    mt = compute_metrics(yt, score(Xt), th)
    ma = compute_metrics(ya, score(Xa), th)
    mc = compute_metrics(yc, score(Xc), th)
    return {
        "seed": seed,
        "threshold": round(th, 4),
        "test_f1": mt["f1"],
        "nazario_2025_recall": ma["recall"],
        "a_fpr": ma["fpr"],
        "b_fpr": round(float(np.mean(score(Xb) >= th)), 4),
        "c_recall": mc["recall"],
        "c_fpr": mc["fpr"],
    }


def main():
    texts = []
    labels = []
    for path in DEFAULT_DATA:
        X, y, _ = load_csv(path)
        texts.extend(X)
        labels.extend(y.tolist())
    y = np.array(labels)
    Xa, ya, _ = load_csv(os.path.join(DATA, "holdout_2025.csv"))
    Xb, _, _ = load_csv(os.path.join(DATA, "holdout_b_2025.csv"))
    Xc, yc, _ = load_csv(os.path.join(DATA, "holdout_c.csv"))
    holdouts = ((Xa, ya), Xb, (Xc, yc))
    pick = lambda ids: [texts[i] for i in ids]
    groups = near_duplicate_groups(texts)

    runs = {m: [] for m in METHODS}
    plateaus = []
    for seed in SEEDS:
        tr, va, te = split_indices(y, seed, groups)
        test = (pick(te), y[te])

        model, _ = fit_model(texts, y, tr, va, "val", seed=seed)
        runs["val"].append(evaluate(model.predict_proba, model.threshold, seed, test, holdouts))
        log("val " + str(runs["val"][-1]))

        trva = np.concatenate([tr, va])
        # üretimle aynı: gruplu CV, out-of-fold skorlarla Platt kalibrasyonu
        oof, fold = cv_oof_scores(pick(trva), y[trva], folds=5, seed=seed, n_jobs=5, groups=groups[trva],
                                  feature_version=PRODUCTION_FEATURE_VERSION)
        cal = PlattCalibrator().fit(oof, y[trva])
        curves = fold_f1_curves(y[trva], cal.transform(oof), fold)
        th_max, _ = select_threshold(curves, "max")
        th_plateau, info = select_threshold(curves, "plateau")
        plateaus.append(dict(info, seed=seed))
        pipe = build_pipeline(feature_version=PRODUCTION_FEATURE_VERSION).fit(pick(trva), y[trva])
        score = lambda X: cal.transform(pipeline_scores(pipe, X))
        for method, th in [("cv-max", th_max), ("cv-plateau", th_plateau)]:
            runs[method].append(evaluate(score, th, seed, test, holdouts))
            log(method + " " + str(runs[method][-1]))
        log("plateau " + str(info))

    summary = {}
    for method in METHODS:
        summary[method] = {}
        for key in KEYS:
            v = np.array([r[key] for r in runs[method]], dtype=float)
            summary[method][key] = {
                "mean": round(float(v.mean()), 4),
                "std": round(float(v.std()), 4),
                "min": round(float(v.min()), 4),
                "max": round(float(v.max()), 4),
            }

    os.makedirs(REPORTS, exist_ok=True)
    f = open(os.path.join(REPORTS, "threshold_stability.json"), "w", encoding="utf-8")
    json.dump({"data": DEFAULT_DATA, "seeds": SEEDS, "runs": runs, "plateaus": plateaus,
               "summary": summary}, f, indent=2)
    f.close()
    write_markdown(summary, plateaus)
    log("Done -> reports/THRESHOLD_STABILITY.md")
    return 0


def write_markdown(summary, plateaus):
    L = []
    L.append("# Threshold stability")
    L.append("")
    L.append("Generated by `threshold_stability.py`. The production configuration is trained with "
             + str(len(SEEDS)) + " seeds (different train/val/test splits and CV folds) and three ways "
             "of choosing the threshold:")
    L.append("")
    L.append("- **val:** train on 70%, pick the max-F1 threshold on one 15% validation split.")
    L.append("- **cv-max:** 5-fold cross-validation on the other 85%, then take the peak of the mean F1 "
             "curve. As in production, the folds keep near-duplicate groups together and the scores are "
             "Platt-calibrated first.")
    L.append("- **cv-plateau:** the same CV, then take the middle of the flat top around the peak. The "
             "flat top is every adjacent threshold whose mean F1 is within one standard error (across "
             "folds) of the peak.")
    L.append("")
    L.append("cv-max and cv-plateau share the same fold models and the same retrained model, so the only "
             "difference between them is the selection rule. The success criteria were set before "
             "running: the plateau rule should give a smaller threshold spread than cv-max, and the "
             "spread of the hold-out results should not grow. The hold-out sets are only measured here.")
    L.append("")
    head = "| metric |"
    sep = "|---|"
    for m in METHODS:
        head += " " + m + ": mean | " + m + ": std |"
        sep += "---:|---:|"
    L.append(head)
    L.append(sep)
    for key in KEYS:
        pct = key not in ("threshold", "test_f1")
        cells = []
        for m in METHODS:
            s = summary[m][key]
            if pct:
                cells += ["%.2f%%" % (100 * s["mean"]), "%.2f" % (100 * s["std"])]
            else:
                cells += ["%.4f" % s["mean"], "%.4f" % s["std"]]
        L.append("| " + LABELS[key] + " | " + " | ".join(cells) + " |")
    L.append("")
    L.append("Percent metrics: the std column is in percentage points.")
    L.append("")
    L.append("## Plateau per seed")
    L.append("")
    L.append("| seed | peak | plateau | middle (chosen) | peak F1 | F1 standard error |")
    L.append("|---:|---:|---|---:|---:|---:|")
    for p in plateaus:
        lo, hi = p["plateau"]
        L.append("| %d | %.3f | %.3f–%.3f | %.4f | %.4f | %.4f |" % (
            p["seed"], p["peak"], lo, hi, (lo + hi) / 2, p["peak_f1"], p["f1_se"]))
    L.append("")
    f = open(os.path.join(REPORTS, "THRESHOLD_STABILITY.md"), "w", encoding="utf-8")
    f.write("\n".join(L))
    f.close()


if __name__ == "__main__":
    sys.exit(main())
