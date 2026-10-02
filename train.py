#!/usr/bin/env python3
# CLI'da kullanılacak modeli eğitir: train/val/test split, threshold val'de seçilir,
# test sonuçları yazdırılır, model threshold + metadata ile birlikte kaydedilir.
# Deneyler/karşılaştırmalar için experiments.py'ye bakın.

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from analyzer import PhishingEmailAnalyzer
from evaluation import (
    best_f1_threshold,
    calibration_stats,
    compute_metrics,
    format_table,
    load_csv,
    near_duplicate_groups,
)
from ml_model import (
    CLASSIFIERS,
    PRODUCTION_FEATURE_VERSION,
    PhishingClassifier,
    build_pipeline,
    fit_with_cv_threshold,
)

DEFAULT_DATA = [
    os.path.join("data", "processed", "kaggle.csv"),
    os.path.join("data", "processed", "nazario_sa.csv"),
    # güncel meşru mailler - olmadan bol linkli duyurularda yanlış alarm %25'e çıkıyordu
    # (bkz. reports/FALSE_ALARMS.md)
    os.path.join("data", "processed", "modern_legit.csv"),
    # daha fazla gerçek phishing - recall'u geri kazandırdı (bkz. reports/RECALL.md)
    os.path.join("data", "processed", "nazario_older.csv"),
]


def split_indices(y, seed, groups=None):
    """70/15/15 train/val/test. "cv" yöntemleri train+val'i birlikte kullanır; test dilimi
    her yöntemde aynı kalır. groups verilirse (yakın kopya grupları) bir grubun bütün
    mailleri aynı dilime düşer: aynı kampanyanın varyantları hem train'de hem test'te
    olmaz (eğitim verisinin %27'sinin en az bir yakın kopyası var)."""
    idx = np.arange(len(y))
    if groups is None:
        tr, rest = train_test_split(idx, test_size=0.3, stratify=y, random_state=seed)
        va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
        return tr, va, te
    fold = np.empty(len(y), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    for k, (_, part) in enumerate(splitter.split(idx, y, groups)):
        fold[part] = k
    return idx[fold >= 6], idx[(fold >= 3) & (fold < 6)], idx[fold < 3]


def fit_model(texts, y, tr, va, method="plateau", seed=42, folds=5, jobs=None, use_rules=True,
              groups=None, feature_version=PRODUCTION_FEATURE_VERSION):
    """method="val"    : train'de eğit, threshold'u val diliminde seç (ilk yöntem).
    method="cv-max" : train+val üzerinde k-fold CV, ortalama F1'in zirvesi.
    method="plateau": aynı CV, zirvedeki düz tepenin (1 standart hata) ortası.
    CV yöntemlerinde skorlar Platt ile kalibre edilir, fold'lar yakın kopya gruplarını
    bölmez (groups verilirse) ve model sonra train+val'in tamamıyla yeniden eğitilir."""
    pick = lambda ids: [texts[i] for i in ids]
    if method == "val":
        model = PhishingClassifier(build_pipeline(use_rules=use_rules, feature_version=feature_version))
        model.fit(pick(tr), y[tr])
        model.threshold = best_f1_threshold(y[va], model.predict_proba(pick(va)))
        model.meta = {"threshold_method": "single validation split (max F1)"}
        return model, len(tr)
    trva = np.concatenate([tr, va])
    rule = "max" if method == "cv-max" else "plateau"
    model = fit_with_cv_threshold(pick(trva), y[trva], folds=folds, seed=seed, n_jobs=jobs,
                                  rule=rule, groups=None if groups is None else groups[trva],
                                  use_rules=use_rules, feature_version=feature_version)
    return model, len(trva)


def main():
    parser = argparse.ArgumentParser(description="Train the ML phishing classifier.")
    parser.add_argument("data", nargs="*", default=DEFAULT_DATA,
                        help="One or more CSV files (default: Kaggle + Nazario/SpamAssassin + modern legit "
                             "+ Nazario 2019-22)")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--classifier", default="logreg", choices=["logreg"],
                        help="Saved model needs probabilities, so only logreg for now "
                             "(see experiments.py for " + "/".join(CLASSIFIERS) + ")")
    parser.add_argument("--no-rules", action="store_true", help="Text features only")
    parser.add_argument("--out", default="model.joblib")
    parser.add_argument("--metrics-out", default="metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-method", default="plateau", choices=["plateau", "cv-max", "val"],
                        help="plateau: middle of the flat top of the CV F1 curve (default); "
                             "cv-max: its exact peak; val: one validation split")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=5, help="Parallel CV folds")
    parser.add_argument("--no-final", action="store_true",
                        help="Save the 85%% evaluation model instead of retraining on all data")
    parser.add_argument("--no-groups", action="store_true",
                        help="Random split instead of keeping near-duplicate groups together")
    args = parser.parse_args()

    texts = []
    labels = []
    for path in args.data:
        X, y, _ = load_csv(path, args.text_col, args.label_col)
        print("Loaded " + path + ": " + str(len(X)) + " emails (" + str(int(y.sum())) + " phishing)")
        texts.extend(X)
        labels.extend(y.tolist())
    y = np.array(labels)
    if len(texts) < 20 or len(set(labels)) < 2:
        print("Error: need at least 20 rows with both classes.", file=sys.stderr)
        return 1

    groups = None if args.no_groups else near_duplicate_groups(texts)
    tr, va, te = split_indices(y, args.seed, groups)
    pick = lambda ids: [texts[i] for i in ids]

    t = time.time()
    model, n_train = fit_model(texts, y, tr, va, args.threshold_method, args.seed, args.folds,
                               args.jobs, use_rules=not args.no_rules, groups=groups)
    method = model.meta["threshold_method"]
    print("Trained in " + str(round(time.time() - t, 1)) + "s on " + str(n_train) + " emails, threshold ("
          + method + ") = " + str(round(model.threshold, 3)))

    analyzer = PhishingEmailAnalyzer(resolve_dns=False)
    rs_va = np.array([analyzer.analyze(x).score for x in pick(va)], dtype=float)
    rs_te = np.array([analyzer.analyze(x).score for x in pick(te)], dtype=float)
    rows = [
        ("rule_based", compute_metrics(y[te], rs_te, best_f1_threshold(y[va], rs_va))),
        ("ml", compute_metrics(y[te], model.predict_proba(pick(te)), model.threshold)),
    ]
    print("")
    print("Test set: " + str(len(te)) + " emails")
    print(format_table(rows, ["precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"]))

    # kalibrasyon: ham logistic regression skoru vs Platt sonrası, aynı test diliminde
    calibration = {"raw": calibration_stats(y[te], model.pipeline.predict_proba(pick(te))[:, 1])}
    if model.calibrator is not None:
        calibration["platt"] = calibration_stats(y[te], model.predict_proba(pick(te)))
    print("")
    print("Calibration on test (lower is better): "
          + ", ".join(k + " ECE " + str(v["ece"]) + " / Brier " + str(v["brier"]) for k, v in calibration.items()))

    # Değerlendirme bitti: kaydedilecek model verinin tamamıyla (test dilimi dahil) yeniden
    # eğitilir. Test rakamları yukarıdaki %85'lik modele aittir; threshold zaten CV'den geliyor.
    evaluated_on = int(n_train)
    if not args.no_final and args.threshold_method != "val":
        t = time.time()
        all_idx = np.arange(len(texts))
        model, n_train = fit_model(texts, y, all_idx, np.array([], dtype=int), args.threshold_method,
                                   args.seed, args.folds, args.jobs, use_rules=not args.no_rules,
                                   groups=groups)
        print("")
        print("Final model retrained on all " + str(n_train) + " emails in "
              + str(round(time.time() - t, 1)) + "s, threshold = " + str(round(model.threshold, 3)))

    # fit_model'in eklediği bilgiyi (threshold yöntemi, plato aralığı) koru
    model.meta.update({
        "data": args.data,
        "n_train": int(n_train),
        "test_metrics_from_model_trained_on": evaluated_on,
        "threshold_method": method,
        "split": "random" if groups is None else "near-duplicate groups kept together",
        "classifier": args.classifier,
        "feature_version": PRODUCTION_FEATURE_VERSION,
        "rules": not args.no_rules,
        "test_metrics": rows[1][1],
        "test_calibration": calibration,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    model.save(args.out)
    f = open(args.metrics_out, "w", encoding="utf-8")
    json.dump({"metrics": dict(rows), "calibration": calibration}, f, indent=2)
    f.close()
    print("")
    print("Model saved to " + args.out + ", metrics to " + args.metrics_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
