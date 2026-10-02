#!/usr/bin/env python3
# Kilitli (lockbox) test: kayıtlı üretim modelini, geliştirme boyunca hiç bakılmamış veride
# TEK BİR KEZ ölçer. Protokol: reports/LOCKBOX_PROTOCOL.md (ölçümden önce yazıldı).
# reports/LOCKBOX.md varsa çalışmayı reddeder (--force ile zorlanırsa README'de belirtilmeli).
#
# Kullanım: ./download_lockbox.sh && python evaluate_lockbox.py

import argparse
import hashlib
import json
import os
import sys

import numpy as np

from analyzer import PhishingEmailAnalyzer
from email_parsing import message_to_text, parse_bytes
from evaluation import compute_metrics, describe_email, format_table_ci, load_csv, near_duplicate_groups
from ml_model import PhishingClassifier
from prepare_data import POT_TOKEN_REGEX, dedup, looks_english, read_mbox_dir
from train import DEFAULT_DATA

RAW = os.path.join("data", "raw", "lockbox")
REPORT = os.path.join("reports", "LOCKBOX.md")
RULE_THRESHOLD = 50


def load_lockbox():
    rows = []
    folder = os.path.join(RAW, "phishing_pot")
    total = 0
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".eml"):
            continue
        total = total + 1
        f = open(os.path.join(folder, fname), "rb")
        raw = f.read()
        f.close()
        try:
            text = POT_TOKEN_REGEX.sub("", message_to_text(parse_bytes(raw)))
        except Exception:
            continue
        if len(text) >= 40 and looks_english(text):
            rows.append({"text": text, "label": 1, "source": "phishing_pot_older"})
    english = len(rows)
    for source, items in sorted(read_mbox_dir(os.path.join(RAW, "legit2026"), 0).items()):
        rows.extend(items)
    return dedup(rows), {"pot_samples": total, "pot_english": english}


def main():
    parser = argparse.ArgumentParser(description="One-time lockbox evaluation of the saved model.")
    parser.add_argument("--model", default="model.joblib")
    parser.add_argument("--force", action="store_true", help="Run again (must be disclosed)")
    args = parser.parse_args()
    if os.path.exists(REPORT) and not args.force:
        print("Lockbox already evaluated (" + REPORT + "). It is meant to be run once; "
              "see reports/LOCKBOX_PROTOCOL.md.", file=sys.stderr)
        return 1

    rows, counts = load_lockbox()

    # eğitimdeki bir maile yakın kopya olanlar çıkarılır (eğitim verisi değişmez)
    train_texts = []
    for path in DEFAULT_DATA:
        X, _, _ = load_csv(path)
        train_texts.extend(X)
    groups = near_duplicate_groups(train_texts + [r["text"] for r in rows])
    train_groups = set(groups[:len(train_texts)].tolist())
    kept = [r for r, g in zip(rows, groups[len(train_texts):]) if g not in train_groups]
    counts["dropped_near_copies_of_training"] = len(rows) - len(kept)

    texts = [r["text"] for r in kept]
    y = np.array([r["label"] for r in kept])
    sources = np.array([r["source"] for r in kept])

    f = open(args.model, "rb")
    model_sha = hashlib.sha256(f.read()).hexdigest()
    f.close()
    model = PhishingClassifier.load(args.model)
    analyzer = PhishingEmailAnalyzer(resolve_dns=False)
    ml = model.predict_proba(texts)
    rules = np.array([analyzer.analyze(t).score for t in texts], dtype=float)

    overall = [
        ("phase 1: rule_based (score >= 50)", compute_metrics(y, rules, RULE_THRESHOLD)),
        ("phase 2: ml (saved threshold)", compute_metrics(y, ml, model.threshold)),
    ]
    per_source = []
    for src in sorted(set(sources.tolist())):
        m = sources == src
        per_source.append({
            "source": src, "n": int(m.sum()), "label": "phishing" if y[m][0] == 1 else "legit",
            "ml_flagged": round(float((ml[m] >= model.threshold).mean()), 4),
            "rules_flagged": round(float((rules[m] >= RULE_THRESHOLD).mean()), 4),
        })
    missed = [i for i in np.argsort(ml) if y[i] == 1 and ml[i] < model.threshold][:8]
    alarms = [i for i in np.argsort(ml)[::-1] if y[i] == 0 and ml[i] >= model.threshold][:8]

    results = {
        "model": {"path": args.model, "sha256": model_sha, "threshold": model.threshold,
                  "trained_at": model.meta.get("trained_at"), "n_train": model.meta.get("n_train")},
        "counts": dict(counts, phishing=int(y.sum()), legit=int((y == 0).sum())),
        "overall": dict(overall),
        "per_source": per_source,
        "missed_phishing": [{"score": round(float(ml[i]), 4), "text": describe_email(texts[i])} for i in missed],
        "false_alarms": [{"score": round(float(ml[i]), 4), "text": describe_email(texts[i])} for i in alarms],
        "forced_rerun": bool(args.force),
    }
    os.makedirs("reports", exist_ok=True)
    f = open(os.path.join("reports", "lockbox.json"), "w", encoding="utf-8")
    json.dump(results, f, indent=2)
    f.close()
    write_markdown(results, overall)
    print(format_table_ci(overall))
    print("Report: " + REPORT)
    return 0


def write_markdown(r, overall):
    c = r["counts"]
    L = []
    L.append("# Lockbox test: one-time evaluation")
    L.append("")
    L.append("Protocol (written before this run): [`LOCKBOX_PROTOCOL.md`](LOCKBOX_PROTOCOL.md). "
             "This is the only evaluation of the saved model on this data"
             + (". **This was a forced rerun.**" if r["forced_rerun"] else "."))
    L.append("")
    L.append("- Model: `" + r["model"]["path"] + "`, sha256 `" + r["model"]["sha256"][:16] + "…`, trained "
             + str(r["model"]["trained_at"]) + " on " + str(r["model"]["n_train"]) + " emails, threshold "
             + "%.3f" % r["model"]["threshold"] + ".")
    L.append("- Phishing: %d of %d older Phishing Pot samples are English. Legitimate: ISC, Samba and Mailman "
             "mailing lists (January, April and July 2026)." % (c["pot_english"], c["pot_samples"]))
    L.append("- %d emails were dropped as near-copies of training emails. Left: **%d phishing, %d "
             "legitimate**." % (c["dropped_near_copies_of_training"], c["phishing"], c["legit"]))
    L.append("")
    L.append(format_table_ci(overall))
    L.append("")
    L.append("| source | n | true label | flagged by ML | flagged by rules |")
    L.append("|---|---:|---|---:|---:|")
    for s in r["per_source"]:
        L.append("| %s | %d | %s | %.1f%% | %.1f%% |" % (s["source"], s["n"], s["label"],
                                                         100 * s["ml_flagged"], 100 * s["rules_flagged"]))
    for key, title in [("missed_phishing", "Missed phishing (lowest score)"),
                       ("false_alarms", "False alarms (highest score)")]:
        L.append("")
        L.append("## " + title)
        L.append("")
        if not r[key]:
            L.append("(none)")
        for e in r[key]:
            L.append("- **%.3f** %s" % (e["score"], e["text"].replace("|", "\\|")))
    L.append("")
    f = open(REPORT, "w", encoding="utf-8")
    f.write("\n".join(L))
    f.close()


if __name__ == "__main__":
    sys.exit(main())
