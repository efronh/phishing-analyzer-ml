#!/usr/bin/env python3
# Recall'u geri kazanma deneyi. Kaçırılan 2025 phishing'lerinin çoğu linksiz, kısa,
# ödeme/fatura/ek dosya bahaneli mailler. Versiyonlar:
#   v3         : şu anki model (sınırlı feature'lar + modern meşru mailler)
#   v3+lure    : v3 + "tuzak" feature'ları (feature_version=3) - veri eklemeden
#   v4         : v3 + Nazario 2019-2022 phishing eğitimde
#   v4+lure    : ikisi birden
#
# Kurallar (sonuçlara bakmadan önce sabitlendi):
#   - Threshold her versiyonda aynı birleşik validation setinde max F1 ile seçilir.
#   - Üretim modeli = validation F1'i en yüksek versiyon. Test setleri seçimde kullanılmaz.
#
# Test setleri:
#   A: Nazario 2025 phishing + Apache 2025 (kaçırılan örnekleri teşhiste incelendi)
#   B: Nazario 2025 phishing + Ubuntu 2025
#   C: Phishing Pot 2026 (İngilizce) + OSGeo 2025 - KÖR, bu deneyden önce hiç bakılmadı
#
# Kullanım: python prepare_data.py && python experiments_recall.py

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
)
from experiments import split3, take
from ml_model import build_pipeline, pipeline_scores

DATA = os.path.join("data", "processed")
REPORTS = "reports"
SEED = 42
COLUMNS = ["precision", "recall", "f1", "fpr", "roc_auc"]


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + msg, flush=True)


def split_simple(texts, seed):
    """Tek sınıflı setler için gruplu 70/15/15: yakın kopyalar aynı dilimde kalır."""
    return group_split_single_class(near_duplicate_groups(texts), seed)


def main():
    needed = ["kaggle.csv", "nazario_sa.csv", "modern_legit.csv", "nazario_older.csv",
              "holdout_2025.csv", "holdout_b_2025.csv", "holdout_c.csv"]
    for f in needed:
        if not os.path.exists(os.path.join(DATA, f)):
            print("Missing " + f + " - run download_data.sh and prepare_data.py first.", file=sys.stderr)
            return 1
    os.makedirs(REPORTS, exist_ok=True)
    started = time.time()

    Xk, yk, _ = load_csv(os.path.join(DATA, "kaggle.csv"))
    Xn, yn, _ = load_csv(os.path.join(DATA, "nazario_sa.csv"))
    Xm, ym, _ = load_csv(os.path.join(DATA, "modern_legit.csv"))
    Xo, yo, _ = load_csv(os.path.join(DATA, "nazario_older.csv"))
    Xa, ya, _ = load_csv(os.path.join(DATA, "holdout_2025.csv"))
    Xb_legit, _, _ = load_csv(os.path.join(DATA, "holdout_b_2025.csv"))
    Xc, yc, _ = load_csv(os.path.join(DATA, "holdout_c.csv"))

    phish_2025 = [x for x, l in zip(Xa, ya) if l == 1]
    Xb = phish_2025 + Xb_legit
    yb = np.array([1] * len(phish_2025) + [0] * len(Xb_legit))

    X_base = Xk + Xn
    y_base = np.concatenate([yk, yn])
    tr, va, te = split3(X_base, y_base, SEED)
    m_tr, m_va, m_te = split_simple(Xm, SEED)
    o_tr, o_va, o_te = split_simple(Xo, SEED)

    def part(X, y, idx):
        return take(X, idx), y[idx]

    def concat(*parts):
        return sum((p[0] for p in parts), []), np.concatenate([p[1] for p in parts])

    base_tr, base_va, base_te = part(X_base, y_base, tr), part(X_base, y_base, va), part(X_base, y_base, te)
    mod_tr, mod_va, mod_te = part(Xm, ym, m_tr), part(Xm, ym, m_va), part(Xm, ym, m_te)
    old_tr, old_va, old_te = part(Xo, yo, o_tr), part(Xo, yo, o_va), part(Xo, yo, o_te)

    val = concat(base_va, mod_va, old_va)
    test_in = concat(base_te, mod_te, old_te)

    versions = [
        ("v3", 2, concat(base_tr, mod_tr)),
        ("v3+lure", 3, concat(base_tr, mod_tr)),
        ("v4", 2, concat(base_tr, mod_tr, old_tr)),
        ("v4+lure", 3, concat(base_tr, mod_tr, old_tr)),
    ]

    rows = {"val": [], "in_domain": [], "A": [], "B": [], "C": []}
    preds = {}
    thresholds = {}
    for name, fv, train in versions:
        log("training " + name)
        pipe = build_pipeline(feature_version=fv).fit(train[0], train[1])
        s_val = pipeline_scores(pipe, val[0])
        th = best_f1_threshold(val[1], s_val)
        thresholds[name] = th
        rows["val"].append((name, compute_metrics(val[1], s_val, th)))
        rows["in_domain"].append((name, compute_metrics(test_in[1], pipeline_scores(pipe, test_in[0]), th)))
        s_a, s_b, s_c = pipeline_scores(pipe, Xa), pipeline_scores(pipe, Xb), pipeline_scores(pipe, Xc)
        preds[name] = {"A": s_a >= th, "B": s_b >= th, "C": s_c >= th}
        rows["A"].append((name, compute_metrics(ya, s_a, th)))
        rows["B"].append((name, compute_metrics(yb, s_b, th)))
        mc = compute_metrics(yc, s_c, th)
        rows["C"].append((name, mc))
        log("  " + name + ": val F1=" + str(rows["val"][-1][1]["f1"])
            + " | C recall=" + str(mc["recall"]) + " fpr=" + str(mc["fpr"]))

    # önceden sabitlenen kural: validation F1'i en yüksek olan seçilir
    selected = max(rows["val"], key=lambda r: r[1]["f1"])[0]

    # farklar gürültü mü? aynı maillerde eşleştirilmiş McNemar testi
    labels = {"A": ya, "B": yb, "C": yc}
    significance = []
    for new, old in [("v3+lure", "v3"), ("v4", "v3"), ("v4+lure", "v4")]:
        for key in ["A", "B", "C"]:
            r = mcnemar_exact(labels[key], preds[new][key], preds[old][key])
            significance.append({"new": new, "old": old, "set": key, **r})
    log("selected by validation F1: " + selected)

    results = {
        "seed": SEED,
        "selected": selected,
        "thresholds": thresholds,
        "sizes": {
            "train_base": len(tr), "train_modern": len(m_tr), "train_older_phishing": len(o_tr),
            "val": len(val[1]), "test_in_domain": len(test_in[1]),
            "holdout_a": len(Xa), "holdout_b": len(Xb), "holdout_c": len(Xc),
            "holdout_c_phishing": int(yc.sum()),
        },
    }
    for key in rows:
        results[key] = dict(rows[key])
    results["significance"] = significance
    results["runtime_seconds"] = round(time.time() - started, 1)

    f = open(os.path.join(REPORTS, "recall.json"), "w", encoding="utf-8")
    json.dump(results, f, indent=2)
    f.close()
    write_markdown(results, rows)
    log("Done in " + str(results["runtime_seconds"]) + "s -> reports/RECALL.md")
    return 0


def write_markdown(r, rows):
    s = r["sizes"]
    L = []
    L.append("# Recovering recall")
    L.append("")
    L.append("Generated by `experiments_recall.py`. Every version's threshold is chosen on the same "
             "validation set (max F1). The production version is the one with the best **validation** "
             "F1. This rule was fixed before any test result was seen.")
    L.append("")
    L.append("| version | features | extra training data |")
    L.append("|---|---|---|")
    L.append("| v3 | bounded rule features | – (current model) |")
    L.append("| v3+lure | + lure features (payment / attachment pretext, generic greeting, urgency, "
             "credential request, short body) | – |")
    L.append("| v4 | bounded rule features | + " + str(s["train_older_phishing"]) + " Nazario 2019–2022 phishing |")
    L.append("| v4+lure | + lure features | + " + str(s["train_older_phishing"]) + " Nazario 2019–2022 phishing |")
    L.append("")
    L.append("**Selected by validation F1: " + r["selected"] + "**")
    L.append("")
    L.append("## Hold-out C: blind (Phishing Pot 2026 + OSGeo 2025 lists)")
    L.append("")
    L.append(str(s["holdout_c_phishing"]) + " English phishing emails from Phishing Pot (honeypot samples, "
             "2026) and " + str(s["holdout_c"] - s["holdout_c_phishing"]) + " legitimate emails from "
             "OSGeo mailing lists (gdal-dev, qgis-user, qgis-developer, 2025). Neither source appears in "
             "training, and no result on this set was seen before the selection rule was fixed.")
    L.append("")
    L.append(format_table(rows["C"], COLUMNS))
    L.append("")
    L.append("## Hold-out A: Nazario 2025 phishing + Apache 2025 lists")
    L.append("")
    L.append("The emails this set's previous model missed were inspected to choose the lure features, "
             "so this set is not blind for v3+lure / v4+lure.")
    L.append("")
    L.append(format_table(rows["A"], COLUMNS))
    L.append("")
    L.append("## Hold-out B: Nazario 2025 phishing + Ubuntu 2025 lists")
    L.append("")
    L.append(format_table(rows["B"], COLUMNS))
    L.append("")
    L.append("## Are the differences real? (McNemar, paired, exact)")
    L.append("")
    L.append("Each comparison counts the emails that only one of the two versions classifies "
             "correctly. A p-value above 0.05 means the difference could easily be noise.")
    L.append("")
    L.append("| comparison | set | only new correct | only old correct | p-value |")
    L.append("|---|---|---:|---:|---:|")
    for c in r["significance"]:
        L.append("| %s vs %s | %s | %d | %d | %.3g |" % (c["new"], c["old"], c["set"],
                 c["only_a_correct"], c["only_b_correct"], c["p_value"]))
    L.append("")
    L.append("## Validation and in-domain test")
    L.append("")
    L.append("Validation (" + str(s["val"]) + " emails, used for thresholds and selection):")
    L.append("")
    L.append(format_table(rows["val"], COLUMNS))
    L.append("")
    L.append("In-domain test (" + str(s["test_in_domain"]) + " emails: Kaggle, Nazario/SA, modern legit, "
             "Nazario 2019–22):")
    L.append("")
    L.append(format_table(rows["in_domain"], COLUMNS))
    L.append("")
    f = open(os.path.join(REPORTS, "RECALL.md"), "w", encoding="utf-8")
    f.write("\n".join(L))
    f.close()


if __name__ == "__main__":
    sys.exit(main())
