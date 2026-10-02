#!/usr/bin/env python3
# Tüm deneyleri çalıştırır ve reports/ altına yazar:
#   1. Kaggle in-domain: rule-based vs farklı ML modelleri vs hybrid (elle / val'de ayarlı) + ablation
#   2. 5-fold cross-validation (ana model)
#   3. Cross-dataset: Kaggle'da eğitilen model gerçek phishing'de (Nazario) ne yapıyor?
#   4. Domain adaptation: biraz hedef-domain verisi eklemek ne kadar düzeltiyor?
#   5. Türkçe spam: İngilizce kuralların Türkçede çalışmadığını, ML'in çalıştığını göster
#   6. Hata analizi + modelin en güçlü feature'ları
#
# Kullanım: python prepare_data.py && python experiments.py

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
    compute_metrics,
    describe_email,
    format_table,
    load_csv,
    near_duplicate_groups,
    stratified_group_split,
)
from ml_model import ML_WEIGHT, PhishingClassifier, build_pipeline, pipeline_scores
from train import split_indices

DATA = os.path.join("data", "processed")
REPORTS = "reports"
COLUMNS = ["precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"]

# (isim, build_pipeline argümanları) - ablation + model karşılaştırması
MODEL_CONFIGS = [
    ("rules_only_lr", {"classifier": "logreg", "use_text": False, "use_rules": True}),
    ("text_nb", {"classifier": "nb", "use_text": True, "use_rules": False}),
    ("text_svm", {"classifier": "svm", "use_text": True, "use_rules": False}),
    ("text_lr", {"classifier": "logreg", "use_text": True, "use_rules": False}),
    ("text+rules_svm", {"classifier": "svm", "use_text": True, "use_rules": True}),
    ("text+rules_lr", {"classifier": "logreg", "use_text": True, "use_rules": True}),
]
MAIN_MODEL = "text+rules_lr"

_analyzer = PhishingEmailAnalyzer(resolve_dns=False)


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + msg, flush=True)


def rule_scores(texts):
    return np.array([_analyzer.analyze(t).score for t in texts], dtype=float)


def hybrid_scores(ml_prob, rules, weight=ML_WEIGHT):
    return weight * ml_prob * 100 + (1 - weight) * np.minimum(rules, 100)


def tune_hybrid_weight(ml_va, rs_va, yva):
    """ML ağırlığını validation F1'ine göre seç (0 = sadece kurallar, 1 = sadece ML).
    Eşitlikte daha yüksek ML ağırlığı. 0.6 elle seçilmişti; adil karşılaştırma için."""
    best_w, best_f1 = None, -1.0
    for w in np.round(np.linspace(0, 1, 21), 2):
        h = hybrid_scores(ml_va, rs_va, w)
        f1 = compute_metrics(yva, h, best_f1_threshold(yva, h))["f1"]
        if f1 >= best_f1:
            best_w, best_f1 = float(w), f1
    return best_w


def split3(texts, y, seed):
    """70/15/15 stratified train/val/test; yakın kopya grupları (aynı kampanyanın
    varyantları, birbirini alıntılayan cevaplar) bölünmez."""
    return split_indices(y, seed, near_duplicate_groups(texts))


def take(texts, idx):
    return [texts[i] for i in idx]


def eval_on(name, scores_val, y_val, scores_test, y_test, rows):
    th = best_f1_threshold(y_val, scores_val)
    m = compute_metrics(y_test, scores_test, th)
    rows.append((name, m))
    log("  " + name + ": F1=" + str(m["f1"]) + " ROC-AUC=" + str(m["roc_auc"]))
    return m


def experiment_in_domain(X, y, seed):
    log("Experiment 1: Kaggle in-domain")
    tr, va, te = split3(X, y, seed)
    Xtr, Xva, Xte = take(X, tr), take(X, va), take(X, te)
    ytr, yva, yte = y[tr], y[va], y[te]

    rows = []
    rs_va, rs_te = rule_scores(Xva), rule_scores(Xte)
    eval_on("rule_based (tuned threshold)", rs_va, yva, rs_te, yte, rows)
    # analyzer'ın kendi eşiği (HIGH = 50) - hiç ayar yapmadan
    rows.append(("rule_based (fixed >=50)", compute_metrics(yte, rs_te, 50)))

    models = {}
    for name, cfg in MODEL_CONFIGS:
        t = time.time()
        p = build_pipeline(**cfg).fit(Xtr, ytr)
        models[name] = p
        m = eval_on(name, pipeline_scores(p, Xva), yva, pipeline_scores(p, Xte), yte, rows)
        m["fit_seconds"] = round(time.time() - t, 1)

    main = models[MAIN_MODEL]
    ml_va, ml_te = pipeline_scores(main, Xva), pipeline_scores(main, Xte)
    eval_on("hybrid (w=0.6, hand-picked)", hybrid_scores(ml_va, rs_va), yva,
            hybrid_scores(ml_te, rs_te), yte, rows)
    w = tune_hybrid_weight(ml_va, rs_va, yva)
    eval_on("hybrid (w=" + str(w) + ", tuned on val)", hybrid_scores(ml_va, rs_va, w), yva,
            hybrid_scores(ml_te, rs_te, w), yte, rows)

    thresholds = {name: best_f1_threshold(yva, pipeline_scores(p, Xva)) for name, p in models.items()}
    thresholds["rule_based"] = best_f1_threshold(yva, rs_va)
    thresholds["hybrid"] = best_f1_threshold(yva, hybrid_scores(ml_va, rs_va))
    thresholds["hybrid_tuned"] = best_f1_threshold(yva, hybrid_scores(ml_va, rs_va, w))
    thresholds["hybrid_weight"] = w

    split = {"train": tr, "val": va, "test": te}
    return rows, models, thresholds, split


def experiment_cv(X, y, idx, seed, folds):
    log("Experiment 2: " + str(folds) + "-fold CV (" + MAIN_MODEL + ")")
    cfg = dict(MODEL_CONFIGS)[MAIN_MODEL]
    Xs, ys = take(X, idx), y[idx]
    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    groups = near_duplicate_groups(Xs)
    f1s = []
    aucs = []
    for k, (a, b) in enumerate(skf.split(Xs, ys, groups)):
        p = build_pipeline(**cfg).fit(take(Xs, a), ys[a])
        m = compute_metrics(ys[b], pipeline_scores(p, take(Xs, b)), 0.5)
        f1s.append(m["f1"])
        aucs.append(m["roc_auc"])
        log("  fold " + str(k + 1) + ": F1=" + str(m["f1"]))
    return {
        "folds": folds,
        "f1_mean": round(float(np.mean(f1s)), 4),
        "f1_std": round(float(np.std(f1s)), 4),
        "roc_auc_mean": round(float(np.mean(aucs)), 4),
        "roc_auc_std": round(float(np.std(aucs)), 4),
    }


def experiment_cross(models, thresholds, Xn, yn):
    log("Experiment 3: cross-dataset (train Kaggle -> test Nazario + SpamAssassin)")
    rows = []
    rs = rule_scores(Xn)
    rows.append(("rule_based (Kaggle threshold)", compute_metrics(yn, rs, thresholds["rule_based"])))
    rows.append(("rule_based (fixed >=50)", compute_metrics(yn, rs, 50)))
    for name, p in models.items():
        m = compute_metrics(yn, pipeline_scores(p, Xn), thresholds[name])
        rows.append((name, m))
        log("  " + name + ": F1=" + str(m["f1"]) + " recall=" + str(m["recall"]))
    ml = pipeline_scores(models[MAIN_MODEL], Xn)
    rows.append(("hybrid (w=0.6, hand-picked)", compute_metrics(yn, hybrid_scores(ml, rs), thresholds["hybrid"])))
    w = thresholds["hybrid_weight"]
    rows.append(("hybrid (w=" + str(w) + ", tuned on val)",
                 compute_metrics(yn, hybrid_scores(ml, rs, w), thresholds["hybrid_tuned"])))
    return rows


def experiment_adaptation(Xk_train, yk_train, Xn, yn, seed):
    """Nazario+SA'yı ikiye böl: yarısı eğitime eklenir, diğer yarısında test edilir."""
    log("Experiment 4: domain adaptation")
    cfg = dict(MODEL_CONFIGS)[MAIN_MODEL]
    # yarılar yakın kopya gruplarını bölmez: aynı kampanya iki yarıya düşmesin
    groups = near_duplicate_groups(Xn)
    a, b = stratified_group_split(yn, groups, 0.5, seed)
    # a'nın içinden threshold için küçük bir val ayır
    a_fit_local, a_val_local = stratified_group_split(yn[a], groups[a], 0.2, seed)
    a_fit, a_val = a[a_fit_local], a[a_val_local]
    Xa, ya = take(Xn, a_fit), yn[a_fit]
    Xv, yv = take(Xn, a_val), yn[a_val]
    Xb, yb = take(Xn, b), yn[b]

    rows = []
    setups = [
        ("Kaggle only", Xk_train, yk_train),
        ("Nazario+SA half only", Xa, ya),
        ("Kaggle + Nazario+SA half", list(Xk_train) + Xa, np.concatenate([yk_train, ya])),
    ]
    for name, Xt, yt in setups:
        p = build_pipeline(**cfg).fit(Xt, yt)
        th = best_f1_threshold(yv, pipeline_scores(p, Xv))
        m = compute_metrics(yb, pipeline_scores(p, Xb), th)
        rows.append((name, m))
        log("  " + name + ": F1=" + str(m["f1"]))
    rs_v, rs_b = rule_scores(Xv), rule_scores(Xb)
    rows.insert(0, ("rule_based", compute_metrics(yb, rs_b, best_f1_threshold(yv, rs_v))))
    return rows


def experiment_turkish(seed):
    log("Experiment 5: Turkish spam")
    Xtr, ytr, _ = load_csv(os.path.join(DATA, "turkish_train.csv"))
    Xte, yte, _ = load_csv(os.path.join(DATA, "turkish_eval.csv"))
    # threshold için eğitim setinin %20'si validation
    idx = np.arange(len(Xtr))
    a, v = train_test_split(idx, test_size=0.2, stratify=ytr, random_state=seed)
    Xa, ya, Xv, yv = take(Xtr, a), ytr[a], take(Xtr, v), ytr[v]

    rows = []
    rs_v, rs_te = rule_scores(Xv), rule_scores(Xte)
    rows.append(("rule_based (tuned threshold)",
                 compute_metrics(yte, rs_te, best_f1_threshold(yv, rs_v))))
    nonzero = float(np.mean(rs_te > 0))
    for name in ["text_nb", "text_lr", "text+rules_lr"]:
        p = build_pipeline(**dict(MODEL_CONFIGS)[name]).fit(Xa, ya)
        th = best_f1_threshold(yv, pipeline_scores(p, Xv))
        m = compute_metrics(yte, pipeline_scores(p, Xte), th)
        rows.append((name, m))
        log("  " + name + ": F1=" + str(m["f1"]))
    return rows, nonzero


def error_examples(texts, y, scores, threshold, kind, n=8):
    """En emin olunan yanlışlar: FP = legit ama yüksek skor, FN = phishing ama düşük skor."""
    scores = np.asarray(scores)
    if kind == "fp":
        idx = [i for i in np.argsort(scores)[::-1] if y[i] == 0 and scores[i] >= threshold]
    else:
        idx = [i for i in np.argsort(scores) if y[i] == 1 and scores[i] < threshold]
    out = []
    for i in idx[:n]:
        out.append({"score": round(float(scores[i]), 4), "text": describe_email(texts[i])})
    return out


def main():
    parser = argparse.ArgumentParser(description="Run all phishing classifier experiments.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    for f in ["kaggle.csv", "nazario_sa.csv", "turkish_train.csv", "turkish_eval.csv"]:
        if not os.path.exists(os.path.join(DATA, f)):
            print("Missing " + f + " - run prepare_data.py first.", file=sys.stderr)
            return 1
    os.makedirs(REPORTS, exist_ok=True)
    started = time.time()

    Xk, yk, _ = load_csv(os.path.join(DATA, "kaggle.csv"))
    Xn, yn, _ = load_csv(os.path.join(DATA, "nazario_sa.csv"))

    in_rows, models, thresholds, split = experiment_in_domain(Xk, yk, args.seed)

    cv = None
    if not args.skip_cv:
        cv = experiment_cv(Xk, yk, np.concatenate([split["train"], split["val"]]), args.seed, args.cv)

    cross_rows = experiment_cross(models, thresholds, Xn, yn)
    adapt_rows = experiment_adaptation(take(Xk, split["train"]), yk[split["train"]], Xn, yn, args.seed)
    tr_rows, tr_rule_coverage = experiment_turkish(args.seed)

    # hata analizi + feature önemleri (ana model)
    main = PhishingClassifier(models[MAIN_MODEL], thresholds[MAIN_MODEL])
    Xte, yte = take(Xk, split["test"]), yk[split["test"]]
    te_scores = main.predict_proba(Xte)
    n_scores = main.predict_proba(Xn)
    errors = {
        "kaggle_test_false_positives": error_examples(Xte, yte, te_scores, main.threshold, "fp"),
        "kaggle_test_false_negatives": error_examples(Xte, yte, te_scores, main.threshold, "fn"),
        "nazario_missed_phishing": error_examples(Xn, yn, n_scores, main.threshold, "fn"),
        "spamassassin_false_alarms": error_examples(Xn, yn, n_scores, main.threshold, "fp"),
    }
    top_phish, top_legit = main.top_features(20)

    results = {
        "seed": args.seed,
        "datasets": {
            "kaggle": {"n": len(Xk), "positive": int(yk.sum())},
            "nazario_sa": {"n": len(Xn), "positive": int(yn.sum())},
        },
        "in_domain": dict(in_rows),
        "hybrid_weight_tuned": thresholds["hybrid_weight"],
        "cross_validation": cv,
        "cross_dataset": dict(cross_rows),
        "domain_adaptation": dict(adapt_rows),
        "turkish": dict(tr_rows),
        "turkish_rule_coverage": round(tr_rule_coverage, 4),
        "top_features": {"phishing": top_phish, "legit": top_legit},
        "errors": errors,
        "runtime_seconds": round(time.time() - started, 1),
    }
    f = open(os.path.join(REPORTS, "results.json"), "w", encoding="utf-8")
    json.dump(results, f, indent=2, ensure_ascii=False)
    f.close()

    write_markdown(results, in_rows, cross_rows, adapt_rows, tr_rows)
    log("Done in " + str(results["runtime_seconds"]) + "s -> reports/results.json, reports/RESULTS.md")
    return 0


def write_markdown(r, in_rows, cross_rows, adapt_rows, tr_rows):
    L = []
    L.append("# Experiment results")
    L.append("")
    L.append("Generated by `experiments.py` (seed " + str(r["seed"]) + "). Thresholds are always "
             "picked on a validation split (max F1) and never on the test data.")
    L.append("")
    L.append("## 1. In-domain: Kaggle Phishing Email dataset")
    d = r["datasets"]["kaggle"]
    L.append("")
    L.append(str(d["n"]) + " emails after cleaning and dedup (" + str(d["positive"])
             + " phishing). Split: 70% train, 15% val, 15% test.")
    L.append("")
    L.append(format_table(in_rows, COLUMNS))
    L.append("")
    L.append("Hybrid = w·ML + (1−w)·rule score. w=0.6 was picked by hand; the tuned row picks w on "
             "validation F1 over 0, 0.05, …, 1 (ties go to the larger w). Tuned w = **"
             + str(r["hybrid_weight_tuned"]) + "**.")
    if r["cross_validation"]:
        cv = r["cross_validation"]
        L.append("")
        L.append("**" + str(cv["folds"]) + "-fold CV** (" + MAIN_MODEL + ", train+val): F1 = "
                 + "%.4f ± %.4f" % (cv["f1_mean"], cv["f1_std"]) + ", ROC-AUC = "
                 + "%.4f ± %.4f" % (cv["roc_auc_mean"], cv["roc_auc_std"]))
    L.append("")
    L.append("## 2. Cross-dataset: trained on Kaggle, tested on Nazario + SpamAssassin")
    d = r["datasets"]["nazario_sa"]
    L.append("")
    L.append(str(d["n"]) + " emails: " + str(d["positive"]) + " real phishing (Nazario 2023–2024) + "
             + str(d["n"] - d["positive"]) + " legitimate (SpamAssassin ham). The model has never seen "
             "either source. Thresholds come from the Kaggle validation set.")
    L.append("")
    L.append(format_table(cross_rows, COLUMNS))
    L.append("")
    L.append("## 3. Domain adaptation")
    L.append("")
    L.append("Half of Nazario+SA is held out as the test set. The other half is used for training "
             "and threshold selection.")
    L.append("")
    L.append(format_table(adapt_rows, COLUMNS))
    L.append("")
    L.append("## 4. Turkish spam (anilguven/turkish_spam_email)")
    L.append("")
    L.append("The rule engine's keywords are English, so it gives a non-zero score to only "
             + "%.1f%%" % (100 * r["turkish_rule_coverage"]) + " of the Turkish test emails.")
    L.append("")
    L.append(format_table(tr_rows, COLUMNS))
    L.append("")
    L.append("## 5. What the main model learned (" + MAIN_MODEL + ", top coefficients)")
    L.append("")
    L.append("| phishing → | weight | ← legit | weight |")
    L.append("|---|---:|---|---:|")
    for (pn, pw), (ln, lw) in zip(r["top_features"]["phishing"], r["top_features"]["legit"]):
        L.append("| `" + pn + "` | %.2f | `" % pw + ln + "` | %.2f |" % lw)
    L.append("")
    L.append("## 6. Error analysis")
    for key, items in r["errors"].items():
        L.append("")
        L.append("### " + key.replace("_", " "))
        L.append("")
        if not items:
            L.append("(none)")
        for e in items:
            L.append("- **" + "%.3f" % e["score"] + "** " + e["text"].replace("|", "\\|"))
    L.append("")
    f = open(os.path.join(REPORTS, "RESULTS.md"), "w", encoding="utf-8")
    f.write("\n".join(L))
    f.close()


if __name__ == "__main__":
    sys.exit(main())
