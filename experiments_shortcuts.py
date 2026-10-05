#!/usr/bin/env python3
# Kaynak kısayollarını kaldırma (katman 1): alıntı/yanıt yapısı, tarihler ve format. Model
# eğitimde "yanıt = meşru", "2002 = meşru", "boşluklu noktalama = meşru (Enron)" gibi kaynak
# işaretleri öğrenmişti; S bunları bütün kaynaklarda aynı kuralla siler.
# Protokol (sonuçlardan önce yazıldı): reports/SHORTCUTS_PROTOCOL.md
# Gizlilik: reklamlar (varsa) için sadece toplam oranlar.
#
# Kullanım: python experiments_shortcuts.py

import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from evaluation import load_csv, mcnemar_exact, near_duplicate_groups, wilson_ci
from experiments_commercial import load_promo, load_redactor, promo_mbox_paths
from experiments_llm import load_zenodo
from experiments_url import load_holdouts
from ml_model import (PRODUCTION_FEATURE_VERSION, PhishingClassifier, PlattCalibrator, build_pipeline,
                      cv_oof_scores, f1_curve, fold_f1_curves, select_threshold)
from train import DEFAULT_DATA

REPORTS = "reports"
SEED = 42
BOOT_SEED = 0
N_BOOT = 2000
TOP_K = 30

# P0 = kayıtlı üretim modeli; diğerleri aynı ayarlarla eğitilir, sadece temizleme değişir
VARIANTS = {
    "P0": (),
    "S": ("reply", "dates", "format"),
    "S without reply removal": ("dates", "format"),
    "S without date masking": ("reply", "format"),
    "S without format normalization": ("reply", "dates"),
}
CANDIDATE = "S"

# Yanıt zinciri şablonları: protokolde sabitlendi, her phishing mailinin sonuna eklenir
BENIGN = "Hi, thanks for sending this over. I have shared it with the team and we will review it this week."
TEMPLATES = {
    "T1": ("\n\nOn Tue, Feb 11, 2025 at 3:42 PM Laura Chen <laura.chen@northwind-consulting.com> wrote:\n"
           "> Hi, thanks for sending this over.\n"
           "> I have shared it with the team and we will review it this week.\n"
           "> Best regards, Laura\n"),
    "T2": ("\n\nOn Feb 11, 2025, at 15:42, Laura Chen\n"
           "<laura.chen@northwind-consulting.com> wrote:\n\n"
           "> Hi, thanks for sending this over.\n"
           "> I have shared it with the team and we will review it this week.\n"),
    "T3": ("\n\n> Hi, thanks for sending this over.\n"
           "> I have shared it with the team and we will review it this week.\n"
           "> Best regards, Laura\n"),
    "T4": ("\n\n-----Original Message-----\nFrom: Laura Chen\nSent: Tuesday, February 11, 2025 3:42 PM\n"
           "To: Accounts Team\nSubject: RE: Q1 budget review\n\n" + BENIGN + "\n"),
    "T5": "\n\n" + BENIGN + "\n",
    "T6": "\n\nThanks,\nLaura\n",
}
TARGETED = ("T1", "T2", "T3")
TEMPLATE_SETS = ("Nazario 2025", "hold-out C")


def corpus_of(source, label):
    """Kaynak denetimi için eğitim satırının derlemi."""
    if source == "kaggle":
        return "Kaggle phishing" if label == 1 else "Kaggle legitimate"
    if source.startswith("spamassassin"):
        return "SpamAssassin ham"
    if source.startswith("nazario"):
        return "Nazario 2019–2024"
    return "Mailing lists 2024"


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + msg, flush=True)


def rate(pred):
    k, n = int(np.sum(pred)), len(pred)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci": wilson_ci(k, n)}


def train_variant(texts, labels, groups, clean):
    """Üretimle aynı yol: gruplu 5-fold CV, Platt, plato eşiği, sonra bütün veriyle yeniden eğitim.
    Döner: (model, fold F1'leri kendi eşiğinde, eşik)."""
    oof, fold = cv_oof_scores(texts, labels, 5, SEED, 5, groups,
                              feature_version=PRODUCTION_FEATURE_VERSION, clean=clean)
    calibrator = PlattCalibrator().fit(oof, labels)
    scores = calibrator.transform(oof)
    threshold, info = select_threshold(fold_f1_curves(labels, scores, fold), "plateau")
    fold_f1 = [float(f1_curve(labels[fold == k], scores[fold == k], np.array([threshold]))[0])
               for k in np.unique(fold)]
    meta = {"phishing_share_in_training": round(float(labels.mean()), 4), "threshold_cv": info, "clean": list(clean)}
    return calibrator, threshold, fold_f1, meta


def auc_with_ci(pos, neg):
    rng = np.random.default_rng(BOOT_SEED)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    point = roc_auc_score(y, np.r_[pos, neg])
    boot = []
    for _ in range(N_BOOT):
        p = pos[rng.integers(0, len(pos), len(pos))]
        q = neg[rng.integers(0, len(neg), len(neg))]
        boot.append(roc_auc_score(y, np.r_[p, q]))
    return {"auc": float(point), "ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])]}


def source_audit(model, texts, corpora, features):
    """Her feature için: onu içeren eğitim maillerinin en sık geldiği derlem ve payı.
    Sadece kelime/karakter feature'ları (kural feature'ları ölçeklenmiş, "içerir" anlamı yok)."""
    if "clean" in model.pipeline.named_steps:
        texts = model.pipeline.named_steps["clean"].transform(texts)
    union = model.pipeline.named_steps["features"]
    names = list(union.get_feature_names_out())
    index = {n: i for i, n in enumerate(names)}
    wanted = [n for n, _ in features if not n.startswith("rules__")]
    blocks = {}
    for part_name, part in union.transformer_list:
        if part_name in ("word", "char"):
            blocks[part_name] = part.transform(texts).tocsc()
    offsets, start = {}, 0
    for part_name, part in union.transformer_list:
        offsets[part_name] = start
        start = start + len(part.get_feature_names_out())
    corpora = np.array(corpora)
    out = {}
    for n in wanted:
        part_name = n.split("__", 1)[0]
        col = blocks[part_name][:, index[n] - offsets[part_name]]
        rows = col.nonzero()[0]
        if len(rows) == 0:
            continue
        values, counts = np.unique(corpora[rows], return_counts=True)
        top = int(np.argmax(counts))
        out[n] = {"emails": int(len(rows)), "top_corpus": str(values[top]), "top_share": float(counts[top] / len(rows))}
    return out


def mcnemar_vs(y, pred, base):
    """Varyant (a) ile P0 (b) aynı maillerde."""
    return mcnemar_exact(y, pred, base)


def clean_json(x):
    if isinstance(x, dict):
        return {k: clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float("%.6g" % x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def main():
    started = time.time()
    saved = PhishingClassifier.load("model.joblib")

    texts, labels, corpora = [], [], []
    for p in DEFAULT_DATA:
        X, y, src = load_csv(p)
        texts.extend(X)
        labels.append(y)
        corpora.extend(corpus_of(s, int(l)) for s, l in zip(src, y))
    labels = np.concatenate(labels)
    groups = near_duplicate_groups(texts)

    # ---- 1. varyantlar
    models, cv = {}, {}
    for name, clean in VARIANTS.items():
        log("training " + name)
        calibrator, threshold, fold_f1, meta = train_variant(texts, labels, groups, clean)
        cv[name] = {"threshold": threshold, "fold_f1": fold_f1}
        if name == "P0":
            if abs(threshold - saved.threshold) > 1e-9:
                print("STOP: recomputed P0 threshold %.4f differs from the saved %.4f." % (threshold, saved.threshold),
                      file=sys.stderr)
                return 1
            models[name] = saved
        else:
            pipe = build_pipeline(feature_version=PRODUCTION_FEATURE_VERSION, clean=clean).fit(texts, labels)
            models[name] = PhishingClassifier(pipe, threshold, meta, calibrator)
        log("%s: threshold %.4f, fold F1 %s" % (name, threshold, ", ".join("%.4f" % f for f in fold_f1)))
    base_f1 = np.array(cv["P0"]["fold_f1"])
    for name in VARIANTS:
        d = np.array(cv[name]["fold_f1"]) - base_f1
        cv[name]["mean_f1"] = float(np.mean(cv[name]["fold_f1"]))
        cv[name]["paired_difference"] = float(d.mean())
        cv[name]["paired_se"] = float(d.std(ddof=1) / np.sqrt(len(d))) if name != "P0" else 0.0

    # ---- 2. değerlendirme setleri
    holdouts = {n: (X, np.asarray(y)) for n, (X, y) in load_holdouts().items()}
    phish_sets = {
        "Nazario 2025": [x for x, l in zip(*holdouts["A"]) if l == 1],
        "hold-out C": [x for x, l in zip(*holdouts["C"]) if l == 1],
    }
    zenodo = [r["text"] for r in load_zenodo()]
    promo = None
    if promo_mbox_paths():
        redactor, _ = load_redactor()
        rows, _ = load_promo(redactor)
        promo = ([r["text"] for r in rows], np.array([r["english"] for r in rows]))

    preds, scores = {}, {}
    for name, model in models.items():
        log("scoring " + name)
        p, s = {}, {}
        for set_name, (X, y) in holdouts.items():
            s[set_name] = model.predict_proba(X)
        for set_name, X in phish_sets.items():
            for t_name, t in [("none", "")] + list(TEMPLATES.items()):
                s[(set_name, t_name)] = model.predict_proba([x + t for x in X]) if t else None
        s["zenodo"] = model.predict_proba(zenodo)
        if promo is not None:
            s["promo"] = model.predict_proba(promo[0])
        for k, v in s.items():
            if v is not None:
                p[k] = v >= model.threshold
        scores[name], preds[name] = s, p

    # ---- 3. sonuçlar
    results = {}
    for name in VARIANTS:
        P, S = preds[name], scores[name]
        r = {"cv": cv[name], "holdouts": {}, "templates": {}}
        for set_name, (X, y) in holdouts.items():
            r["holdouts"][set_name] = {
                "recall": rate(P[set_name][y == 1]), "false_alarms": rate(P[set_name][y == 0]),
                "vs_P0_phishing": mcnemar_vs(y[y == 1], P[set_name][y == 1], preds["P0"][set_name][y == 1]),
                "vs_P0_legit": mcnemar_vs(y[y == 0], P[set_name][y == 0], preds["P0"][set_name][y == 0])}
        for set_name, X in phish_sets.items():
            n = len(X)
            ones = np.ones(n, dtype=int)
            r["templates"][set_name] = {}
            for t_name in ["none"] + list(TEMPLATES):
                key = (set_name, t_name)
                pred = P[key] if t_name != "none" else (P["A"][holdouts["A"][1] == 1] if set_name == "Nazario 2025"
                                                         else P["C"][holdouts["C"][1] == 1])
                ref = preds["P0"][key] if t_name != "none" else (
                    preds["P0"]["A"][holdouts["A"][1] == 1] if set_name == "Nazario 2025"
                    else preds["P0"]["C"][holdouts["C"][1] == 1])
                r["templates"][set_name][t_name] = {"recall": rate(pred), "vs_P0": mcnemar_vs(ones, pred, ref)}
        r["zenodo"] = {"recall": rate(P["zenodo"]),
                       "vs_P0": mcnemar_vs(np.ones(len(zenodo), dtype=int), P["zenodo"], preds["P0"]["zenodo"])}
        if promo is not None:
            eng = promo[1]
            r["promotions"] = {"English": rate(P["promo"][eng]), "all": rate(P["promo"])}
            naz = S["A"][holdouts["A"][1] == 1]
            pot = S["C"][holdouts["C"][1] == 1]
            r["same_genre_auc"] = {"English promotions vs Nazario 2025": auc_with_ci(naz, S["promo"][eng]),
                                   "English promotions vs hold-out C": auc_with_ci(pot, S["promo"][eng])}
        results[name] = r

    # ---- 4. en güçlü feature'lar ve kaynak denetimi (P0 ve S)
    audit = {}
    for name in ("P0", CANDIDATE):
        log("source audit " + name)
        phishing, legit = models[name].top_features(TOP_K)
        a = source_audit(models[name], texts, corpora, phishing + legit)
        audit[name] = {"phishing": [[n, w, a.get(n)] for n, w in phishing],
                       "legitimate": [[n, w, a.get(n)] for n, w in legit]}

    # ---- 5. karar (protokoldeki kural)
    c = results[CANDIDATE]
    sig = lambda t, better: t["p_value"] < 0.05 and ((t["only_a_correct"] > t["only_b_correct"]) == better)
    checks = {}
    for set_name in TEMPLATE_SETS:
        for t_name in TARGETED:
            checks["robust %s %s" % (t_name, set_name)] = sig(c["templates"][set_name][t_name]["vs_P0"], True)
        checks["recall not lower, " + set_name] = not sig(c["templates"][set_name]["none"]["vs_P0"], False)
    for set_name in ("A", "B", "C"):
        checks["false alarms not higher, " + set_name] = not sig(c["holdouts"][set_name]["vs_P0_legit"], False)
    checks["Zenodo recall not lower"] = not sig(c["zenodo"]["vs_P0"], False)
    decision = {"checks": checks, "qualifies": all(checks.values())}
    log("decision: S " + ("qualifies" if decision["qualifies"] else "does not qualify"))

    out = clean_json({"variants": {k: list(v) for k, v in VARIANTS.items()}, "candidate": CANDIDATE,
                      "templates": TEMPLATES, "decision": decision, "results": results, "audit": audit,
                      "runtime_s": round(time.time() - started)})
    f = open(os.path.join(REPORTS, "shortcuts.json"), "w", encoding="utf-8")
    json.dump(out, f, indent=2, ensure_ascii=False)
    f.close()
    write_markdown(out)
    log("done in %ds -> reports/SHORTCUTS.md" % (time.time() - started))
    return 0


# ---- rapor

def pct(x, digits=1):
    if x is None or x.get("rate") is None:
        return "-"
    return "%.*f%% (%.*f–%.*f)" % (digits, 100 * x["rate"], digits, 100 * x["ci"][0], digits, 100 * x["ci"][1])


def mc(t):
    return "%d / %d, p = %.3g" % (t["only_a_correct"], t["only_b_correct"], t["p_value"])


def write_markdown(r):
    res = r["results"]
    names = list(r["variants"])
    L = ["# Removing source shortcuts (layer 1)", ""]
    L.append("Generated by `experiments_shortcuts.py`. Protocol, written before the first run: "
             "[`SHORTCUTS_PROTOCOL.md`](SHORTCUTS_PROTOCOL.md). S removes reply structure, masks dates and "
             "normalizes the text format; the three other variants each leave one of those out (descriptive).")
    L.append("")
    L.append("McNemar columns read \"only the variant right / only P0 right, p\".")
    L.append("")
    d = r["decision"]
    L.append("## Decision")
    L.append("")
    L.append("**S " + ("qualifies and replaces production." if d["qualifies"] else "does not qualify; nothing ships.")
             + "**")
    L.append("")
    L.append("| condition | holds |")
    L.append("|---|---|")
    for k, v in d["checks"].items():
        L.append("| %s | %s |" % (k, "yes" if v else "**no**"))
    L.append("")

    L.append("## Reply-chain test")
    L.append("")
    L.append("Recall on phishing with a template appended. T1–T3 are removed by S's cleaning by construction; "
             "T4–T6 test what the model learned.")
    L.append("")
    for set_name in TEMPLATE_SETS:
        L.append("### " + set_name)
        L.append("")
        L.append("| template | " + " | ".join(names) + " |")
        L.append("|---|" + "---:|" * len(names))
        for t_name in ["none"] + list(r["templates"]):
            L.append("| %s | %s |" % ("unmodified" if t_name == "none" else t_name, " | ".join(
                pct(res[n]["templates"][set_name][t_name]["recall"]) for n in names)))
        L.append("")
        L.append("S against P0: " + "; ".join(
            "%s %s" % (t, mc(res[CANDIDATE]["templates"][set_name][t]["vs_P0"]))
            for t in ["none"] + list(r["templates"])) + ".")
        L.append("")

    L.append("## Hold-outs")
    L.append("")
    L.append("| variant | set | recall | false alarms | McNemar phishing | McNemar legitimate |")
    L.append("|---|---|---:|---:|---:|---:|")
    for n in names:
        for set_name in ("A", "B", "C"):
            h = res[n]["holdouts"][set_name]
            L.append("| %s | %s | %s | %s | %s | %s |" % (n, set_name, pct(h["recall"]), pct(h["false_alarms"]),
                                                     "-" if n == "P0" else mc(h["vs_P0_phishing"]),
                                                     "-" if n == "P0" else mc(h["vs_P0_legit"])))
    L.append("")

    L.append("## In-domain CV, LLM-written phishing, promotions")
    L.append("")
    L.append("| variant | threshold | mean fold F1 | paired change (± SE) | Zenodo recall | English promotions "
             "flagged | all promotions flagged |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for n in names:
        c = res[n]["cv"]
        promo = res[n].get("promotions")
        L.append("| %s | %.4f | %.4f | %s | %s | %s | %s |" % (
            n, c["threshold"], c["mean_f1"],
            "-" if n == "P0" else "%+.4f ± %.4f" % (c["paired_difference"], c["paired_se"]),
            pct(res[n]["zenodo"]["recall"]), pct(promo["English"]) if promo else "-",
            pct(promo["all"]) if promo else "-"))
    L.append("")
    if "same_genre_auc" in res["P0"]:
        L.append("Same-genre ROC-AUC (1.0 = perfect separation, 0.5 = chance):")
        L.append("")
        keys = list(res["P0"]["same_genre_auc"])
        L.append("| variant | " + " | ".join(keys) + " |")
        L.append("|---|" + "---:|" * len(keys))
        for n in names:
            L.append("| %s | %s |" % (n, " | ".join(
                "%.3f (%.3f–%.3f)" % (res[n]["same_genre_auc"][k]["auc"], *res[n]["same_genre_auc"][k]["ci"])
                for k in keys)))
        L.append("")

    L.append("## Strongest features and where they come from")
    L.append("")
    L.append("For each word or character feature: the training corpus most of the emails containing it come from, "
             "and that share. Rule features are not audited.")
    L.append("")
    for name, a in r["audit"].items():
        for side in ("phishing", "legitimate"):
            L.append("### %s, toward %s" % (name, side))
            L.append("")
            L.append("| feature | weight | emails | top corpus (share) |")
            L.append("|---|---:|---:|---|")
            for feat, w, au in a[side]:
                L.append("| `%s` | %+.2f | %s | %s |" % (
                    feat.replace("|", "\\|"), w, "-" if au is None else au["emails"],
                    "-" if au is None else "%s (%.0f%%)" % (au["top_corpus"], 100 * au["top_share"])))
            L.append("")
    f = open(os.path.join(REPORTS, "SHORTCUTS.md"), "w", encoding="utf-8")
    f.write("\n".join(L))
    f.close()


if __name__ == "__main__":
    sys.exit(main())
