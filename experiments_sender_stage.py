#!/usr/bin/env python3
# Doğrulanmış gönderen için ikinci aşama: gönderen doğrulanmış ve şüpheli değilse, modelin
# phishing demesi için daha yüksek bir eşik (t_v) gerekir. Model değişmez; sadece karar katmanı.
# Protokol (sonuçlardan önce yazıldı): reports/SENDER_STAGE_PROTOCOL.md
# Gizlilik: reklamlar için sadece toplam oranlar.
#
# Kullanım: python experiments_sender_stage.py

import glob
import json
import mailbox
import os
import sys
import time

import numpy as np

from email_parsing import message_to_text, parse_bytes
from evaluation import load_csv, mcnemar_exact, near_duplicate_groups, wilson_ci
from experiments_commercial import load_promo, load_redactor, promo_mbox_paths
from header_signals import from_host, header_signals, sender_relief, suspicious_sender, verified_sender
from ml_model import PRODUCTION_FEATURE_VERSION, PhishingClassifier, PlattCalibrator, cv_oof_scores
from prepare_data import POT_TOKEN_REGEX, RECIPIENT_REGEX, text_key
from train import DEFAULT_DATA

RAW = os.path.join("data", "raw")
DATA = os.path.join("data", "processed")
REPORTS = "reports"
SEED = 42
PERCENTILE = 5
SENSITIVITY = [0.9, 0.95, 0.99, 0.999]
NEVER = 1.01  # "doğrulanmış ve şüpheli değilse hiç alarm verme"


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + msg, flush=True)


def relief_by_key(messages, clean):
    """Ham mailleri, prepare_data.py'nin ürettiği metnin anahtarıyla eşler: key -> (doğrulanmış, şüpheli).
    Aynı metin birden çok kez geçerse ilki (dedup'un tuttuğu) kullanılır."""
    out = {}
    for msg in messages:
        try:
            text = clean(message_to_text(msg))
        except Exception:
            continue
        key = text_key(text)
        if key not in out:
            out[key] = sender_relief(msg, text)
    return out


def mbox(path):
    return mailbox.mbox(path, factory=lambda fp: parse_bytes(fp.read()))


def eml_dir(folder):
    for path in sorted(glob.glob(os.path.join(folder, "*.eml"))):
        f = open(path, "rb")
        raw = f.read()
        f.close()
        yield parse_bytes(raw)


def strip_recipient(text):
    return RECIPIENT_REGEX.sub("", text)


def strip_pot(text):
    return POT_TOKEN_REGEX.sub("", text)


def flags_for(texts, table):
    """Her metin için (doğrulanmış, şüpheli); eşleşmeyen satırlar doğrulanmamış sayılır (rahatlatma yok)."""
    out = [table.get(text_key(t)) for t in texts]
    matched = sum(1 for x in out if x is not None)
    return [x if x is not None else (False, False) for x in out], matched


def decide(p, flags, t, t_v):
    bar = np.array([t_v if (v and not s) else t for v, s in flags])
    return p >= bar


def rate(k, n):
    return {"k": int(k), "n": int(n), "rate": round(k / n, 4) if n else None, "ci": wilson_ci(int(k), int(n))}


def main():
    if not promo_mbox_paths():
        print("No finished .mbox export in data/raw/own_promo/", file=sys.stderr)
        return 1
    started = time.time()
    model = PhishingClassifier.load("model.joblib")
    t = float(model.threshold)

    # ---- 1. t_v: eğitim verisinin out-of-fold skorlarından
    texts, labels = [], []
    for p in DEFAULT_DATA:
        X, y, _ = load_csv(p)
        texts.extend(X)
        labels.append(y)
    labels = np.concatenate(labels)
    log("out-of-fold scores on " + str(len(texts)) + " training emails")
    oof, _ = cv_oof_scores(texts, labels, 5, SEED, 5, near_duplicate_groups(texts),
                           feature_version=PRODUCTION_FEATURE_VERSION)
    oof = PlattCalibrator().fit(oof, labels).transform(oof)
    train_table = relief_by_key((m for y in range(2019, 2025) for m in mbox(os.path.join(RAW, "nazario-%d.mbox" % y))),
                                strip_recipient)
    phish_idx = np.where(labels == 1)[0]
    tr_flags = [train_table.get(text_key(texts[i])) for i in phish_idx]
    eligible = [oof[i] for i, f in zip(phish_idx, tr_flags) if f is not None and f[0] and not f[1]]
    t_v = max(t, float(np.percentile(eligible, PERCENTILE)))
    log("t = %.3f, t_v = %.4f (from %d verified, not suspicious training phishing)" % (t, t_v, len(eligible)))

    # ---- 2. değerlendirme setleri
    Xc, yc, _ = load_csv(os.path.join(DATA, "holdout_c.csv"))
    c_phish = [x for x, l in zip(Xc, yc) if l == 1]
    Xa, ya, _ = load_csv(os.path.join(DATA, "holdout_2025.csv"))
    n_phish = [x for x, l in zip(Xa, ya) if l == 1]
    c_flags, c_matched = flags_for(c_phish, relief_by_key(eml_dir(os.path.join(RAW, "phishing_pot")), strip_pot))
    n_flags, n_matched = flags_for(n_phish, relief_by_key(mbox(os.path.join(RAW, "nazario-2025.mbox")), strip_recipient))

    redactor, _ = load_redactor()

    # load_promo ham maili tutmuyor: başlık sinyalleri ve From host'u yüklerken alınır,
    # ücretsiz hosting kontrolü maskelenmiş metinle yapılır (maskeleme linklere dokunmuyor)
    rows, _ = load_promo(redactor, extra=lambda msg: (header_signals(msg), from_host(msg)))
    promo = [r["text"] for r in rows]
    english = np.array([r["english"] for r in rows])
    p_flags = [(verified_sender(sig), suspicious_sender(host, r["text"], sig)) for r in rows for sig, host in [r["extra"]]]

    sets = {
        "hold-out C phishing (Phishing Pot)": (c_phish, c_flags, 1, c_matched),
        "Nazario 2025 phishing": (n_phish, n_flags, 1, n_matched),
        "promotions, English": ([x for x, e in zip(promo, english) if e], [f for f, e in zip(p_flags, english) if e], 0, None),
        "promotions, all": (promo, p_flags, 0, None),
    }
    scores = {name: model.predict_proba(X) for name, (X, *_rest) in sets.items()}

    results = {"t": t, "t_v": t_v, "n_eligible_training_phishing": len(eligible), "sets": {}, "sensitivity": {}}
    for name, (X, flags, label, matched) in sets.items():
        p = scores[name]
        base = p >= t
        stage = decide(p, flags, t, t_v)
        y = np.full(len(X), label)
        verified = sum(1 for v, _ in flags if v)
        relieved = sum(1 for v, s in flags if v and not s)
        flagged = lambda pred: int(pred.sum())
        mc = mcnemar_exact(y, stage, base)
        results["sets"][name] = {
            "n": len(X), "matched_to_raw": matched, "verified": rate(verified, len(X)),
            "verified_not_suspicious": rate(relieved, len(X)),
            "production_flagged": rate(flagged(base), len(X)), "stage_flagged": rate(flagged(stage), len(X)),
            "mcnemar": mc,
            # phishing setinde: only_b_correct = sadece üretimin yakaladığı (kaybedilen phishing)
            # reklam setinde: only_a_correct = ikinci aşamanın kaldırdığı yanlış alarm
        }
        results["sensitivity"][name] = {("never alert" if tv == NEVER else "%g" % tv):
                                        round(float(decide(p, flags, t, max(t, tv)).mean()), 4)
                                        for tv in SENSITIVITY + [NEVER]}

    def lost_recall(name):
        m = results["sets"][name]["mcnemar"]
        return m["only_b_correct"] > m["only_a_correct"] and m["p_value"] < 0.05

    fa = results["sets"]["promotions, English"]["mcnemar"]
    crit = {"c_recall_not_lower": not lost_recall("hold-out C phishing (Phishing Pot)"),
            "nazario_2025_recall_not_lower": not lost_recall("Nazario 2025 phishing"),
            "english_promotions_lower": fa["only_a_correct"] > fa["only_b_correct"] and fa["p_value"] < 0.05}
    results["decision"] = dict(crit, qualifies=all(crit.values()))
    results["runtime_seconds"] = round(time.time() - started, 1)
    log("decision: " + json.dumps(results["decision"]))

    f = open(os.path.join(REPORTS, "sender_stage.json"), "w", encoding="utf-8")
    json.dump(results, f, indent=2, default=float)
    f.close()
    write_markdown(results)
    log("Done in " + str(results["runtime_seconds"]) + "s -> reports/SENDER_STAGE.md")
    return 0


def pct(x):
    return "-" if x["rate"] is None else "%.1f%% (%.1f–%.1f)" % (100 * x["rate"], 100 * x["ci"][0], 100 * x["ci"][1])


def write_markdown(r):
    L = ["# Verified-sender second stage", ""]
    L.append("Generated by `experiments_sender_stage.py`. Protocol, written before the first run: "
             "[`SENDER_STAGE_PROTOCOL.md`](SENDER_STAGE_PROTOCOL.md). The production model is unchanged; for a "
             "verified, not suspicious sender the bar rises from t = %.3f to **t_v = %.4f**, chosen from training "
             "out-of-fold scores (5th percentile of %d verified, not suspicious training phishing emails)."
             % (r["t"], r["t_v"], r["n_eligible_training_phishing"]))
    L.append("")
    d = r["decision"]
    L.append("## Decision")
    L.append("")
    L.append("**" + ("The second stage qualifies." if d["qualifies"] else "The second stage does not qualify.") + "**")
    L.append("")
    yn = lambda b: "yes" if b else "**no**"
    L.append("- Hold-out C recall not significantly lower: " + yn(d["c_recall_not_lower"]))
    L.append("- Nazario 2025 recall not significantly lower: " + yn(d["nazario_2025_recall_not_lower"]))
    L.append("- English promotions false alarms significantly lower: " + yn(d["english_promotions_lower"])
             + " (not blind: this set was used to find the problem)")
    L.append("")
    L.append("## Results")
    L.append("")
    L.append("For phishing sets *flagged* is the recall; for promotions it is the false-alarm rate.")
    L.append("")
    L.append("| set | emails | matched to raw | verified sender | verified, not suspicious | flagged: production | "
             "flagged: + second stage | McNemar p |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, x in r["sets"].items():
        L.append("| %s | %d | %s | %s | %s | %s | %s | %.3g |" % (
            name, x["n"], "-" if x["matched_to_raw"] is None else str(x["matched_to_raw"]), pct(x["verified"]),
            pct(x["verified_not_suspicious"]), pct(x["production_flagged"]), pct(x["stage_flagged"]),
            x["mcnemar"]["p_value"]))
    L.append("")
    L.append("## Sensitivity (descriptive, does not decide)")
    L.append("")
    keys = list(next(iter(r["sensitivity"].values())).keys())
    L.append("| set | " + " | ".join("t_v = " + k if k != "never alert" else k for k in keys) + " |")
    L.append("|---|" + "---:|" * len(keys))
    for name, row in r["sensitivity"].items():
        L.append("| " + name + " | " + " | ".join("%.1f%%" % (100 * row[k]) for k in keys) + " |")
    L.append("")
    L.append("Emails without headers (every mailing-list hold-out) are never verified, so their false alarms are "
             "unchanged by construction.")
    L.append("")
    f = open(os.path.join(REPORTS, "SENDER_STAGE.md"), "w", encoding="utf-8")
    f.write("\n".join(L))
    f.close()


if __name__ == "__main__":
    sys.exit(main())
