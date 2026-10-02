#!/usr/bin/env python3
# Yanlış alarm azaltma deneyi. Model versiyonları:
#   v1          : eski feature'lar (ham link sayısı vs.), Kaggle + Nazario/SA
#   v2          : sınırlı (bounded) feature'lar, aynı veri
#   v2+val      : v2 modeli, ama threshold modern meşru maillerle de seçiliyor
#                 (kazanç eğitimden mi, sadece threshold kalibrasyonundan mı?)
#   v3          : sınırlı feature'lar + modern meşru mailler eğitimde (Python/Fedora/GNU 2024)
#
# Her versiyon için iki çalışma noktası, ikisi de SADECE validation'da seçilir:
#   max_f1  : validation F1'i maksimum
#   fpr<=1% : validation'da yanlış alarm oranı en fazla %1
#
# Test setleri:
#   A: Nazario 2025 phishing + Apache listeleri 2025 (teşhiste kullanıldı, tam kör değil)
#   B: Nazario 2025 phishing + Ubuntu listeleri 2025 (kör - bu deneyden önce hiç bakılmadı)
#
# Kullanım: python prepare_data.py && python experiments_false_alarms.py

import json
import os
import sys
import time

import numpy as np

from evaluation import (
    best_f1_threshold,
    compute_metrics,
    format_table,
    group_split_single_class,
    load_csv,
    mcnemar_exact,
    near_duplicate_groups,
    threshold_for_fpr,
)
from experiments import split3, take
from ml_model import build_pipeline, pipeline_scores

DATA = os.path.join("data", "processed")
REPORTS = "reports"
SEED = 42
TARGET_FPR = 0.01
COLUMNS = ["precision", "recall", "f1", "fpr", "roc_auc"]


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + msg, flush=True)


def per_source_fpr(scores, threshold, y, sources):
    out = {}
    for src in sorted(set(sources)):
        idx = [i for i, s in enumerate(sources) if s == src and y[i] == 0]
        if idx:
            out[src] = round(float(np.mean(np.asarray(scores)[idx] >= threshold)), 4)
    return out


def main():
    needed = ["kaggle.csv", "nazario_sa.csv", "modern_legit.csv", "holdout_2025.csv", "holdout_b_2025.csv"]
    for f in needed:
        if not os.path.exists(os.path.join(DATA, f)):
            print("Missing " + f + " - run download_data.sh and prepare_data.py first.", file=sys.stderr)
            return 1
    os.makedirs(REPORTS, exist_ok=True)
    started = time.time()

    Xk, yk, _ = load_csv(os.path.join(DATA, "kaggle.csv"))
    Xn, yn, _ = load_csv(os.path.join(DATA, "nazario_sa.csv"))
    Xm, ym, _ = load_csv(os.path.join(DATA, "modern_legit.csv"))
    Xa, ya, sa = load_csv(os.path.join(DATA, "holdout_2025.csv"))
    Xb_legit, _, sb_legit = load_csv(os.path.join(DATA, "holdout_b_2025.csv"))

    # B = Nazario 2025 phishing (A'dakiyle aynı) + Ubuntu meşru mailleri
    phish_2025 = [x for x, l in zip(Xa, ya) if l == 1]
    Xb = phish_2025 + Xb_legit
    yb = np.array([1] * len(phish_2025) + [0] * len(Xb_legit))
    sb = ["nazario_2025"] * len(phish_2025) + list(sb_legit)

    # aynı split her versiyonda: temel veri ve modern veri ayrı ayrı 70/15/15
    X_base = Xk + Xn
    y_base = np.concatenate([yk, yn])
    tr, va, te = split3(X_base, y_base, SEED)
    # modern meşru maillerin yarısı birbirini alıntılayan cevaplar: yakın kopyalar aynı dilimde
    m_tr, m_va, m_te = group_split_single_class(near_duplicate_groups(Xm), SEED)

    base_train = (take(X_base, tr), y_base[tr])
    base_val = (take(X_base, va), y_base[va])
    base_test = (take(X_base, te), y_base[te])
    mod_train = (take(Xm, m_tr), ym[m_tr])
    mod_val = (take(Xm, m_va), ym[m_va])
    mod_test = (take(Xm, m_te), ym[m_te])

    def concat(a, b):
        return a[0] + b[0], np.concatenate([a[1], b[1]])

    base_plus_mod_val = concat(base_val, mod_val)

    # (isim, feature_version, eğitim verisi, threshold seçilen validation seti)
    versions = [
        ("v1", 1, base_train, base_val),
        ("v2", 2, base_train, base_val),
        ("v2+val", 2, base_train, base_plus_mod_val),
        ("v3", 2, concat(base_train, mod_train), base_plus_mod_val),
    ]

    fitted = {}
    rows = {"in_domain": [], "modern_test_fpr": {}, "A": [], "B": []}
    preds = {}
    by_source = {"A": {}, "B": {}}
    thresholds = {}
    for name, fv, train, val in versions:
        if name == "v2+val":
            # aynı v2 modeli, sadece threshold farklı veriyle seçiliyor
            pipe = fitted["v2"]
        else:
            log("training " + name)
            pipe = build_pipeline(feature_version=fv).fit(train[0], train[1])
            fitted[name] = pipe
        val_scores = pipeline_scores(pipe, val[0])
        ops = {
            "max_f1": best_f1_threshold(val[1], val_scores),
            "fpr<=1%": threshold_for_fpr(val[1], val_scores, TARGET_FPR),
        }
        thresholds[name] = ops

        s_base = pipeline_scores(pipe, base_test[0])
        s_mod = pipeline_scores(pipe, mod_test[0])
        s_a = pipeline_scores(pipe, Xa)
        s_b = pipeline_scores(pipe, Xb)
        for op, th in ops.items():
            label = name + " @ " + op
            rows["in_domain"].append((label, compute_metrics(base_test[1], s_base, th)))
            rows["modern_test_fpr"][label] = round(float(np.mean(s_mod >= th)), 4)
            preds[label] = {"A": s_a >= th, "B": s_b >= th}
            ma = compute_metrics(ya, s_a, th)
            mb = compute_metrics(yb, s_b, th)
            rows["A"].append((label, ma))
            rows["B"].append((label, mb))
            by_source["A"][label] = per_source_fpr(s_a, th, ya, sa)
            by_source["B"][label] = per_source_fpr(s_b, th, yb, sb)
            log("  " + label + ": A fpr=" + str(ma["fpr"]) + " recall=" + str(ma["recall"])
                + " | B fpr=" + str(mb["fpr"]))

    # farklar gürültü mü? (max_f1 çalışma noktasında, aynı maillerde eşleştirilmiş)
    significance = []
    for new, old in [("v2", "v1"), ("v2+val", "v2"), ("v3", "v2+val"), ("v3", "v1")]:
        for key, yy in [("A", ya), ("B", yb)]:
            r = mcnemar_exact(yy, preds[new + " @ max_f1"][key], preds[old + " @ max_f1"][key])
            significance.append({"new": new, "old": old, "set": key, **r})

    results = {
        "seed": SEED,
        "target_fpr": TARGET_FPR,
        "sizes": {
            "train_base": len(tr), "train_modern": len(m_tr),
            "val_base": len(va), "val_modern": len(m_va),
            "test_base": len(te), "test_modern": len(m_te),
            "holdout_a": len(Xa), "holdout_b": len(Xb),
        },
        "thresholds": thresholds,
        "in_domain": dict(rows["in_domain"]),
        "modern_test_fpr": rows["modern_test_fpr"],
        "holdout_a": dict(rows["A"]),
        "holdout_b": dict(rows["B"]),
        "fpr_by_source": by_source,
        "significance": significance,
        "runtime_seconds": round(time.time() - started, 1),
    }
    f = open(os.path.join(REPORTS, "false_alarms.json"), "w", encoding="utf-8")
    json.dump(results, f, indent=2)
    f.close()
    write_markdown(results, rows, by_source)
    log("Done in " + str(results["runtime_seconds"]) + "s -> reports/FALSE_ALARMS.md")
    return 0


def source_table(by_source):
    labels = list(by_source.keys())
    sources = sorted(set(s for v in by_source.values() for s in v))
    L = ["| model | " + " | ".join(s.replace("apache_", "").replace("ubuntu-", "") for s in sources) + " |"]
    L.append("|---|" + "---:|" * len(sources))
    for label in labels:
        L.append("| " + label + " | " + " | ".join(
            "%.1f%%" % (100 * by_source[label][s]) for s in sources) + " |")
    return "\n".join(L)


def write_markdown(r, rows, by_source):
    s = r["sizes"]
    L = []
    L.append("# Reducing false alarms")
    L.append("")
    L.append("Generated by `experiments_false_alarms.py`. All thresholds are chosen on validation data. "
             "Neither hold-out set is used for training or tuning.")
    L.append("")
    L.append("| version | rule features | training data | threshold picked on |")
    L.append("|---|---|---|---|")
    L.append("| v1 | raw counts (link count, summed URL score…) | Kaggle + Nazario/SA | Kaggle + Nazario/SA val |")
    L.append("| v2 | bounded (ratios, max, log, clipped) | Kaggle + Nazario/SA | Kaggle + Nazario/SA val |")
    L.append("| v2+val | bounded | Kaggle + Nazario/SA | + modern legit val |")
    L.append("| v3 | bounded | + " + str(s["train_modern"]) + " modern legit (Python/Fedora/GNU 2024) | + modern legit val |")
    L.append("")
    L.append("Operating points: **max_f1** maximizes F1 on validation. **fpr<=1%** is the lowest "
             "threshold whose validation false-alarm rate is at most 1%.")
    L.append("")
    L.append("## Hold-out B: blind (Nazario 2025 phishing + Ubuntu 2025 lists)")
    L.append("")
    L.append("Never looked at before this experiment. The legitimate emails come from an organization "
             "that appears nowhere in training.")
    L.append("")
    L.append(format_table(rows["B"], COLUMNS))
    L.append("")
    L.append("False-alarm rate per list:")
    L.append("")
    L.append(source_table(by_source["B"]))
    L.append("")
    L.append("## Hold-out A (Nazario 2025 phishing + Apache 2025 lists)")
    L.append("")
    L.append("Not fully blind: the v1 errors on this set were used to diagnose the problem.")
    L.append("")
    L.append(format_table(rows["A"], COLUMNS))
    L.append("")
    L.append("False-alarm rate per list:")
    L.append("")
    L.append(source_table(by_source["A"]))
    L.append("")
    L.append("## Are the differences real? (McNemar, paired, exact, max_f1 operating point)")
    L.append("")
    L.append("| comparison | set | only new correct | only old correct | p-value |")
    L.append("|---|---|---:|---:|---:|")
    for c in r["significance"]:
        L.append("| %s vs %s | %s | %d | %d | %.3g |" % (c["new"], c["old"], c["set"],
                 c["only_a_correct"], c["only_b_correct"], c["p_value"]))
    L.append("")
    L.append("## No regression on the original data")
    L.append("")
    L.append("Kaggle + Nazario/SA test split (" + str(s["test_base"]) + " emails):")
    L.append("")
    L.append(format_table(rows["in_domain"], COLUMNS))
    L.append("")
    L.append("False-alarm rate on the modern-legit test split (" + str(s["test_modern"])
             + " Python/Fedora/GNU 2024 emails, not trained on):")
    L.append("")
    L.append("| model | false alarms |")
    L.append("|---|---:|")
    for label, v in r["modern_test_fpr"].items():
        L.append("| " + label + " | %.1f%% |" % (100 * v))
    L.append("")
    f = open(os.path.join(REPORTS, "FALSE_ALARMS.md"), "w", encoding="utf-8")
    f.write("\n".join(L))
    f.close()


if __name__ == "__main__":
    sys.exit(main())
