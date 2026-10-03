#!/usr/bin/env python3
# Kayıtlı modeli (train.py çıktısı) eğitimde HİÇ kullanılmamış 2025 verisiyle test eder:
#   A: Nazario 2025 phishing + Apache mail listeleri 2025 (meşru)
#   B: Nazario 2025 phishing + Ubuntu mail listeleri 2025 (meşru)
#   C: Phishing Pot 2026 (İngilizce phishing) + OSGeo mail listeleri 2025 (meşru)
# Threshold dahil hiçbir şey bu veriye göre ayarlanmıyor - model olduğu gibi kullanılıyor.
#
# Kullanım: python prepare_data.py && python train.py && python evaluate_holdout.py

import argparse
import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from analyzer import PhishingEmailAnalyzer
from evaluation import (compute_metrics, describe_email, format_prevalence_table, format_table,
                        format_table_ci, load_csv, prevalence_rows)
from ml_model import PhishingClassifier

DATA = os.path.join("data", "processed")
REPORTS = "reports"
COLUMNS = ["precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"]

# analyzer'ın "HIGH" eşiği - faz 1 bu eşikle phishing diyordu
RULE_THRESHOLD = 50
# eğitimdeki bir phishing'e bu kadar benzeyen mail "aynı kampanyanın tekrarı" sayılıyor
NEAR_DUP_SIMILARITY = 0.8


def near_duplicate_mask(holdout_texts, train_texts):
    """Her hold-out maili için eğitimdeki en benzer maile cosine benzerliği."""
    vec = TfidfVectorizer(min_df=1, sublinear_tf=True).fit(train_texts + holdout_texts)
    sims = cosine_similarity(vec.transform(holdout_texts), vec.transform(train_texts))
    best = sims.max(axis=1)
    return best >= NEAR_DUP_SIMILARITY, best


def examples(texts, scores, idx, n=8):
    out = []
    for i in idx[:n]:
        out.append({"score": round(float(scores[i]), 4), "text": describe_email(texts[i])})
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved model on the 2025 hold-out set.")
    parser.add_argument("--model", default="model.joblib")
    args = parser.parse_args()

    path = os.path.join(DATA, "holdout_2025.csv")
    if not os.path.exists(path) or not os.path.exists(args.model):
        print("Need " + path + " and " + args.model + " - run prepare_data.py and train.py first.",
              file=sys.stderr)
        return 1

    model = PhishingClassifier.load(args.model)
    X, y, sources = load_csv(path)
    sources = np.array(sources)
    print("Hold-out: " + str(len(X)) + " emails (" + str(int(y.sum())) + " phishing)")
    print("Model: trained " + str(model.meta.get("trained_at")) + " on " + str(model.meta.get("data"))
          + ", threshold " + str(round(model.threshold, 3)))

    analyzer = PhishingEmailAnalyzer(resolve_dns=False)
    rules = np.array([analyzer.analyze(t).score for t in X], dtype=float)
    ml = model.predict_proba(X)

    # CLI'ın kararları: model varsa threshold, yoksa kural riski (HIGH/CRITICAL = skor >= 50)
    overall = [
        ("phase 1: rule_based (score >= 50)", compute_metrics(y, rules, RULE_THRESHOLD)),
        ("phase 2: ml (saved threshold)", compute_metrics(y, ml, model.threshold)),
    ]
    # gerçekçi phishing paylarında precision için her setin (y, skor) çifti
    prevalence_sets = [("A", y.copy(), ml.copy())]

    # B: aynı Nazario 2025 phishing + Ubuntu meşru mailleri (farklı kurum)
    overall_b = []
    path_b = os.path.join(DATA, "holdout_b_2025.csv")
    if os.path.exists(path_b):
        Xb, _, sb = load_csv(path_b)
        rules_b = np.array([analyzer.analyze(t).score for t in Xb], dtype=float)
        ml_b = model.predict_proba(Xb)
        p_idx = y == 1
        yb = np.concatenate([np.ones(int(p_idx.sum()), dtype=int), np.zeros(len(Xb), dtype=int)])
        overall_b = [
            ("phase 1: rule_based (score >= 50)",
             compute_metrics(yb, np.concatenate([rules[p_idx], rules_b]), RULE_THRESHOLD)),
            ("phase 2: ml (saved threshold)",
             compute_metrics(yb, np.concatenate([ml[p_idx], ml_b]), model.threshold)),
        ]
        prevalence_sets.append(("B", yb, np.concatenate([ml[p_idx], ml_b])))
        # tablo ve hata örnekleri için B'nin meşru maillerini de ekle (C ayrı tutuluyor)
        X = X + Xb
        y = np.concatenate([y, np.zeros(len(Xb), dtype=int)])
        sources = np.concatenate([sources, np.array(sb)])
        rules = np.concatenate([rules, rules_b])
        ml = np.concatenate([ml, ml_b])

    # kaynak bazında: phishing'de recall, meşru listelerde false positive oranı
    per_source = []
    for src in sorted(set(sources.tolist())):
        m = sources == src
        pred = ml[m] >= model.threshold
        rule_pred = rules[m] >= RULE_THRESHOLD
        label = int(y[m][0])
        per_source.append({
            "source": src,
            "n": int(m.sum()),
            "label": "phishing" if label == 1 else "legit",
            "ml_flagged": round(float(pred.mean()), 4),
            "rules_flagged": round(float(rule_pred.mean()), 4),
        })

    # C: tamamen ayrı kaynaklar (Phishing Pot + OSGeo)
    overall_c = []
    c_info = {}
    path_c = os.path.join(DATA, "holdout_c.csv")
    if os.path.exists(path_c):
        Xc, yc, sc = load_csv(path_c)
        rules_c = np.array([analyzer.analyze(t).score for t in Xc], dtype=float)
        ml_c = model.predict_proba(Xc)
        overall_c = [
            ("phase 1: rule_based (score >= 50)", compute_metrics(yc, rules_c, RULE_THRESHOLD)),
            ("phase 2: ml (saved threshold)", compute_metrics(yc, ml_c, model.threshold)),
        ]
        prevalence_sets.append(("C", yc, ml_c))
        sc = np.array(sc)
        for src in sorted(set(sc.tolist())):
            m = sc == src
            per_source_c = {
                "source": src,
                "n": int(m.sum()),
                "label": "phishing" if int(yc[m][0]) == 1 else "legit",
                "ml_flagged": round(float((ml_c[m] >= model.threshold).mean()), 4),
                "rules_flagged": round(float((rules_c[m] >= RULE_THRESHOLD).mean()), 4),
            }
            c_info.setdefault("per_source", []).append(per_source_c)
        c_missed = [i for i in np.argsort(ml_c) if yc[i] == 1 and ml_c[i] < model.threshold]
        c_false = [i for i in np.argsort(ml_c)[::-1] if yc[i] == 0 and ml_c[i] >= model.threshold]
        c_info["missed_phishing"] = examples(Xc, ml_c, c_missed)
        c_info["false_alarms"] = examples(Xc, ml_c, c_false)
        c_info["n"] = len(Xc)
        c_info["positive"] = int(yc.sum())

    # near-duplicate analizi: eğitimdeki Nazario 2023/24 kampanyalarının tekrarı mı?
    Xtrain, ytrain, _ = load_csv(os.path.join(DATA, "nazario_sa.csv"))
    train_phish = [t for t, l in zip(Xtrain, ytrain) if l == 1]
    phish_idx = np.where(y == 1)[0]
    dup, best_sim = near_duplicate_mask([X[i] for i in phish_idx], train_phish)
    novel_idx = phish_idx[~dup]
    dup_idx = phish_idx[dup]
    near_dup = {
        "similarity_threshold": NEAR_DUP_SIMILARITY,
        "near_duplicates": int(dup.sum()),
        "novel": int((~dup).sum()),
        "ml_recall_near_duplicates": round(float((ml[dup_idx] >= model.threshold).mean()), 4) if len(dup_idx) else None,
        "ml_recall_novel": round(float((ml[novel_idx] >= model.threshold).mean()), 4),
        "rules_recall_novel": round(float((rules[novel_idx] >= RULE_THRESHOLD).mean()), 4),
    }
    # novel phishing + A'nın meşru mailleri (B ayrı raporlanıyor)
    legit_idx = np.where((y == 0) & np.char.startswith(sources.astype(str), "apache_"))[0]
    strict_idx = np.concatenate([novel_idx, legit_idx])
    strict = [
        ("phase 1: rule_based", compute_metrics(y[strict_idx], rules[strict_idx], RULE_THRESHOLD)),
        ("phase 2: ml", compute_metrics(y[strict_idx], ml[strict_idx], model.threshold)),
    ]

    missed = [i for i in np.argsort(ml) if y[i] == 1 and ml[i] < model.threshold]
    false_alarms = [i for i in np.argsort(ml)[::-1] if y[i] == 0 and ml[i] >= model.threshold]

    results = {
        "model": {"path": args.model, "threshold": model.threshold, "meta": model.meta},
        "n": len(X),
        "positive": int(y.sum()),
        "legit_a": int(len(legit_idx)),
        "legit_b": int(len(X) - int(y.sum()) - len(legit_idx)),
        "overall": dict(overall),
        "overall_b": dict(overall_b),
        "overall_c": dict(overall_c),
        "holdout_c": c_info,
        "per_source": per_source,
        "near_duplicate_analysis": near_dup,
        "novel_only": dict(strict),
        "prevalence": [{"set": name, "test_share": round(float(ys.mean()), 4),
                        "test_pr_auc": compute_metrics(ys, ss, model.threshold)["pr_auc"],
                        "rows": prevalence_rows(ys, ss, model.threshold)}
                       for name, ys, ss in prevalence_sets],
        "errors": {
            "missed_phishing": examples(X, ml, missed),
            "false_alarms": examples(X, ml, false_alarms),
        },
    }

    os.makedirs(REPORTS, exist_ok=True)
    f = open(os.path.join(REPORTS, "holdout_2025.json"), "w", encoding="utf-8")
    json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    f.close()
    write_markdown(results, overall, strict, overall_b, overall_c)

    print("")
    print("Hold-out A (Apache legit):")
    print(format_table(overall, COLUMNS))
    if overall_b:
        print("")
        print("Hold-out B (Ubuntu legit):")
        print(format_table(overall_b, COLUMNS))
    if overall_c:
        print("")
        print("Hold-out C (Phishing Pot + OSGeo):")
        print(format_table(overall_c, COLUMNS))
    print("")
    print("Near-duplicates of training phishing: " + str(near_dup["near_duplicates"]) + " / "
          + str(len(phish_idx)) + ". Without them:")
    print(format_table(strict, COLUMNS))
    print("")
    print("Report: reports/HOLDOUT_2025.md")
    return 0


def write_markdown(r, overall, strict, overall_b, overall_c):
    nd = r["near_duplicate_analysis"]
    L = []
    L.append("# Hold-out test: 2025 data the model never saw")
    L.append("")
    L.append("Generated by `evaluate_holdout.py`. The saved model (`train.py`, trained on "
             + ", ".join("`" + os.path.basename(d) + "`" for d in r["model"]["meta"].get("data", []))
             + ") is used exactly as shipped. Nothing, including the threshold, is tuned on this data.")
    L.append("")
    L.append("- **Phishing:** " + str(r["positive"]) + " emails from the Nazario corpus, 2025. That year "
             "is not in the training data.")
    L.append("- **Legitimate, set A:** " + str(r["legit_a"]) + " emails from 8 Apache Software "
             "Foundation mailing lists (Feb, May and Aug 2025): user questions, developer discussion, "
             "release announcements.")
    if overall_b:
        L.append("- **Legitimate, set B:** " + str(r["legit_b"]) + " emails from Ubuntu mailing lists "
                 "(Jan, Apr, Jul and Oct 2025): security notices, user questions, announcements.")
    L.append("- Emails that appear word-for-word in the training data are removed.")
    if overall_c:
        c = r["holdout_c"]
        L.append("- **Set C, separate sources:** " + str(c["positive"]) + " English phishing emails from "
                 "Phishing Pot (honeypot samples, 2026, CC BY-NC 4.0) and " + str(c["n"] - c["positive"])
                 + " legitimate emails from OSGeo mailing lists (2025).")
    L.append("")
    L.append("## Hold-out A: phishing + Apache lists")
    L.append("")
    L.append(format_table_ci(overall))
    L.append("")
    if overall_b:
        L.append("## Hold-out B: phishing + Ubuntu lists")
        L.append("")
        L.append(format_table_ci(overall_b))
        L.append("")
    if overall_c:
        L.append("## Hold-out C: Phishing Pot + OSGeo lists")
        L.append("")
        L.append(format_table_ci(overall_c))
        L.append("")
        L.append("| source | n | true label | flagged by ML | flagged by rules |")
        L.append("|---|---:|---|---:|---:|")
        for s in r["holdout_c"]["per_source"]:
            L.append("| " + s["source"] + " | " + str(s["n"]) + " | " + s["label"] + " | "
                     + "%.1f%%" % (100 * s["ml_flagged"]) + " | " + "%.1f%%" % (100 * s["rules_flagged"]) + " |")
        L.append("")
        for key, title in [("missed_phishing", "Missed phishing in set C"),
                           ("false_alarms", "False alarms in set C")]:
            L.append("### " + title)
            L.append("")
            items = r["holdout_c"][key]
            if not items:
                L.append("(none)")
            for e in items:
                L.append("- **" + "%.3f" % e["score"] + "** " + e["text"].replace("|", "\\|"))
            L.append("")
    L.append("## Precision in a real inbox")
    L.append("")
    L.append("The hold-outs are 22–33% phishing; a real inbox after the provider's filters is far "
             "below that. Recall and the false-alarm rate do not depend on the phishing share, but "
             "precision and PR-AUC do. Precision below is what the measured recall and false-alarm "
             "rate (same threshold) imply at each share: TPR·π / (TPR·π + FPR·(1−π)). The 95% "
             "interval samples recall and FPR from their Jeffreys posteriors. PR-AUC re-weights "
             "the legitimate emails to the target share, so it ranks the same scores in that inbox.")
    L.append("")
    L.append(format_prevalence_table([(p["set"], p["test_share"], p["test_pr_auc"], p["rows"])
                                      for p in r["prevalence"]]))
    L.append("")
    L.append("Caveats: this assumes the hold-out's legitimate mail is representative of the inbox's "
             "legitimate mail (it is all mailing lists), and the false-alarm rate rests on 3–22 "
             "errors, so the intervals are wide.")
    L.append("")
    L.append("## Excluding repeats of known campaigns")
    L.append("")
    L.append(str(nd["near_duplicates"]) + " of the " + str(r["positive"]) + " phishing emails are "
             "near-copies (TF-IDF cosine ≥ " + str(nd["similarity_threshold"]) + ") of 2023–24 "
             "phishing used in training. Same campaign, new ID. ML recall on those: "
             + "%.1f%%" % (100 * (nd["ml_recall_near_duplicates"] or 0)) + ". On the "
             + str(nd["novel"]) + " genuinely new ones: **" + "%.1f%%" % (100 * nd["ml_recall_novel"])
             + "** (rules: " + "%.1f%%" % (100 * nd["rules_recall_novel"]) + ").")
    L.append("")
    L.append("New phishing only, plus the Apache legitimate emails:")
    L.append("")
    L.append(format_table_ci(strict))
    L.append("")
    L.append("## By source")
    L.append("")
    L.append("| source | n | true label | flagged by ML | flagged by rules |")
    L.append("|---|---:|---|---:|---:|")
    for s in r["per_source"]:
        L.append("| " + s["source"] + " | " + str(s["n"]) + " | " + s["label"] + " | "
                 + "%.1f%%" % (100 * s["ml_flagged"]) + " | " + "%.1f%%" % (100 * s["rules_flagged"]) + " |")
    L.append("")
    L.append("For phishing sources, *flagged* is the recall. For legitimate sources, it is the false-alarm rate.")
    for key, title in [("missed_phishing", "Missed phishing (lowest ML score)"),
                       ("false_alarms", "False alarms (highest ML score)")]:
        L.append("")
        L.append("## " + title)
        L.append("")
        items = r["errors"][key]
        if not items:
            L.append("(none)")
        for e in items:
            L.append("- **" + "%.3f" % e["score"] + "** " + e["text"].replace("|", "\\|"))
    L.append("")
    f = open(os.path.join(REPORTS, "HOLDOUT_2025.md"), "w", encoding="utf-8")
    f.write("\n".join(L))
    f.close()


if __name__ == "__main__":
    sys.exit(main())
