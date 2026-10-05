#!/usr/bin/env python3
# Maliyete ve phishing payına göre karar politikası. Model değişmez, sadece karar katmanı:
#   P0      üretim eşiği (0.519, eğitim karışımında F1'e göre)
#   P1      prior shift: olasılığı gerçek paya göre yeniden oku, maliyeti en düşük eşik (aday)
#   P2      eğitim out-of-fold skorlarında maliyeti doğrudan en düşük tutan eşik (teşhis)
#   oracle  hold-out'un kendisinde en düşük maliyet (sadece referans, tanımı gereği iyimser)
# Ayrıca analist inceleme bandı (Chow kuralı) ve risk-coverage eğrisi.
# Protokol (sonuçlardan önce yazıldı): reports/DECISION_POLICY_PROTOCOL.md
# Gizlilik: reklamlar (varsa) için sadece toplam oranlar.
#
# Kullanım: python experiments_decision.py

import json
import os
import sys
import time

import numpy as np

from evaluation import (REAL_PREVALENCES, calibration_stats, cost_threshold, expected_cost, load_csv,
                        min_cost_threshold, near_duplicate_groups, policy_rates, review_band,
                        shift_probability, wilson_ci)
from experiments_commercial import load_promo, load_redactor, promo_mbox_paths
from experiments_url import load_holdouts
from ml_model import (PRODUCTION_FEATURE_VERSION, PhishingClassifier, PlattCalibrator, cv_oof_scores,
                      fold_f1_curves, select_threshold)
from train import DEFAULT_DATA

REPORTS = "reports"
SEED = 42
BOOT_SEED = 0
N_BOOT = 2000
COST_RATIOS = (1, 10, 100, 1000)
PREVALENCES = REAL_PREVALENCES
PRIMARY = (100, 0.01)
REVIEW_COSTS = (0.05, 0.1, 0.25, 0.5)
PRIMARY_REVIEW = 0.25
N_CURVE = 50
HOLDOUTS = ("A", "B", "C")
RATE_KEYS = ("fnr", "fpr", "review_phishing", "review_legit")


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + msg, flush=True)


def cell_key(r, pi):
    return "r=%g, pi=%g" % (r, pi)


def boot_rates(y, scores, policies, chunk=100):
    """Her (lo, hi) politikası için N_BOOT yeniden örneklemede sınıf bazındaki oranlar.
    Phishing ve meşru ayrı ayrı yeniden örneklenir; bütün politikalar aynı örnekleri görür,
    yani politikalar arası farklar eşleştirilmiş."""
    rng = np.random.default_rng(BOOT_SEED)
    y = np.asarray(y).astype(bool)
    s1, s0 = scores[y], scores[~y]
    out = {pol: {k: np.empty(N_BOOT) for k in RATE_KEYS} for pol in policies}
    for start in range(0, N_BOOT, chunk):
        b1 = s1[rng.integers(0, len(s1), (chunk, len(s1)))]
        b0 = s0[rng.integers(0, len(s0), (chunk, len(s0)))]
        part = slice(start, start + chunk)
        for lo, hi in policies:
            f1, f0 = b1 >= hi, b0 >= hi
            r1, r0 = (b1 > lo) & ~f1, (b0 > lo) & ~f0
            o = out[(lo, hi)]
            o["fnr"][part] = (~f1 & ~r1).mean(axis=1)
            o["fpr"][part] = f0.mean(axis=1)
            o["review_phishing"][part] = r1.mean(axis=1)
            o["review_legit"][part] = r0.mean(axis=1)
    return out


def interval(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def precision(rates, pi):
    """Payı pi olan kutuda alarmların gerçek phishing olan kısmı (tek eşikli politika)."""
    tp = (1 - rates["fnr"] - rates["review_phishing"]) * pi
    fp = rates["fpr"] * (1 - pi)
    return np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-300), np.nan)


def evaluate(y, scores, band, boot, r, pi, c=0.0):
    """Bir politikanın bir setteki sonucu; boot None ise (oracle) aralıksız."""
    point = policy_rates(y, scores, *band)
    out = {"band": list(band), "fnr": point["fnr"], "fpr": point["fpr"],
           "review_rate": pi * point["review_phishing"] + (1 - pi) * point["review_legit"],
           "cost": 1000 * expected_cost(point, r, pi, c)}
    if band[0] == band[1]:
        out["precision"] = float(precision(point, pi))
    if boot is not None:
        b = boot[band]
        out["fnr_ci"] = interval(b["fnr"])
        out["fpr_ci"] = interval(b["fpr"])
        out["cost_ci"] = interval(1000 * expected_cost(b, r, pi, c))
        if band[0] == band[1]:
            out["precision_ci"] = interval(precision(b, pi)[~np.isnan(precision(b, pi))])
        out["_cost_boot"] = 1000 * expected_cost(b, r, pi, c)
    return out


def paired_difference(a, b):
    """a - b, maliyet olarak (1000 mail başına), eşleştirilmiş bootstrap aralığıyla."""
    return {"difference": a["cost"] - b["cost"], "ci": interval(a["_cost_boot"] - b["_cost_boot"])}


def reliability(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": [float(edges[b]), float(edges[b + 1])], "n": int(m.sum()),
                         "mean_predicted": float(p[m].mean()), "observed": float(y[m].mean())})
    return rows


def flagged_share(mask):
    k, n = int(mask.sum()), len(mask)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci": wilson_ci(k, n)}


def clean(x):
    """JSON için: bootstrap dizilerini at, ondalıkları kısalt."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float("%.6g" % x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def main():
    started = time.time()
    model = PhishingClassifier.load("model.joblib")
    t0 = float(model.threshold)
    pi_t = float(model.meta["phishing_share_in_training"])

    # ---- 1. eğitim out-of-fold skorları, üretimdekiyle aynı ayarlar (fit_with_cv_threshold)
    texts, labels = [], []
    for p in DEFAULT_DATA:
        X, y, _ = load_csv(p)
        texts.extend(X)
        labels.append(y)
    labels = np.concatenate(labels)
    log("out-of-fold scores on " + str(len(texts)) + " training emails")
    oof, fold = cv_oof_scores(texts, labels, 5, SEED, 5, near_duplicate_groups(texts),
                              feature_version=PRODUCTION_FEATURE_VERSION)
    calibrator = PlattCalibrator().fit(oof, labels)
    train_scores = calibrator.transform(oof)
    t_check, _ = select_threshold(fold_f1_curves(labels, train_scores, fold), "plateau")
    sanity = {"saved_threshold": t0, "recomputed_threshold": t_check,
              "saved_train_share": pi_t, "recomputed_train_share": float(labels.mean()),
              "saved_platt": [float(model.calibrator.lr.coef_[0][0]), float(model.calibrator.lr.intercept_[0])],
              "recomputed_platt": [float(calibrator.lr.coef_[0][0]), float(calibrator.lr.intercept_[0])],
              "training_oof_calibration": calibration_stats(labels, train_scores)}
    log("sanity: " + json.dumps(clean(sanity)))
    if abs(t_check - t0) > 1e-9:
        print("STOP: the recomputed threshold (%.4f) differs from the saved one (%.4f). "
              "The protocol says to investigate before going on." % (t_check, t0), file=sys.stderr)
        return 1

    # ---- 2. hold-out'lar, kaydedilmiş modelle
    sets = {"training": (labels, train_scores)}
    for name, (X, y) in load_holdouts().items():
        sets[name] = (np.asarray(y), model.predict_proba(X))
        log("hold-out %s: %d emails, %d phishing" % (name, len(X), int(np.sum(y))))

    # ---- 3. eşikler: hepsi sadece eğitim verisinden
    cells = [(r, pi) for pi in PREVALENCES for r in COST_RATIOS]
    thresholds = {(r, pi): {"P0": t0, "P1": cost_threshold(r, pi, pi_t),
                            "P2": min_cost_threshold(labels, train_scores, r, pi)[0]} for r, pi in cells}
    r_p, pi_p = PRIMARY
    bands = {c: review_band(r_p, c, pi_p, pi_t) for c in REVIEW_COSTS}
    p1_primary = thresholds[PRIMARY]["P1"]
    policies = sorted({(t, t) for th in thresholds.values() for t in th.values()}
                      | {b for b in bands.values() if b is not None})

    results = {}
    for name, (y, s) in sets.items():
        log("bootstrap on " + name)
        boot = boot_rates(y, s, policies)
        res = {"n_phishing": int(y.sum()), "n_legit": int(len(y) - y.sum()), "cells": {}}
        for r, pi in cells:
            th = dict(thresholds[(r, pi)])
            row = {pol: evaluate(y, s, (t, t), boot, r, pi) for pol, t in th.items()}
            if name != "training":
                t_or = min_cost_threshold(y, s, r, pi)[0]
                row["oracle"] = evaluate(y, s, (t_or, t_or), None, r, pi)
                row["regret"] = {pol: row[pol]["cost"] - row["oracle"]["cost"] for pol in ("P0", "P1", "P2")}
            row["P1_minus_P0"] = paired_difference(row["P1"], row["P0"])
            row["trivial_cost"] = 1000 * min(pi * r, 1 - pi)
            res["cells"][cell_key(r, pi)] = row

        # inceleme bandı, ana hücrede
        p1 = res["cells"][cell_key(*PRIMARY)]["P1"]
        review = {}
        for c, band in bands.items():
            if band is None:
                continue
            e = evaluate(y, s, band, boot, r_p, pi_p, c)
            e["minus_P1"] = paired_difference(e, p1)
            review["c=%g" % c] = e
        res["review"] = review
        curve = []
        for c in np.linspace(0, r_p / (1 + r_p), N_CURVE):
            band = review_band(r_p, c, pi_p, pi_t) or (p1_primary, p1_primary)
            rt = policy_rates(y, s, *band)
            rate = pi_p * rt["review_phishing"] + (1 - pi_p) * rt["review_legit"]
            auto = pi_p * r_p * rt["fnr"] + (1 - pi_p) * rt["fpr"]
            curve.append({"c": float(c), "band": list(band), "coverage": 1 - rate,
                          "auto_cost_per_1000_decided": 1000 * auto / (1 - rate) if rate < 1 else None,
                          "total_cost": 1000 * (auto + c * rate)})
        res["risk_coverage"] = curve

        if name != "training":
            share = float(y.mean())
            reread = shift_probability(s, share, pi_t)
            res["calibration"] = {"share": share, "as_is": calibration_stats(y, s),
                                  "reread": calibration_stats(y, reread), "reliability": reliability(y, reread)}
        results[name] = res

    # ---- 4. karar (protokoldeki kural)
    prim = cell_key(*PRIMARY)
    p1_checks, rev_checks = {}, {}
    for name in HOLDOUTS:
        row = results[name]["cells"][prim]
        d = row["P1_minus_P0"]
        p1_checks[name] = {"lower": row["P1"]["cost"] < row["P0"]["cost"], "significantly_higher": d["ci"][0] > 0}
        rv = results[name]["review"]["c=%g" % PRIMARY_REVIEW]
        rev_checks[name] = {"lower": rv["cost"] < row["P1"]["cost"],
                            "significantly_higher": rv["minus_P1"]["ci"][0] > 0}

    def passes(checks):
        return sum(c["lower"] for c in checks.values()) >= 2 and not any(c["significantly_higher"]
                                                                        for c in checks.values())

    p1_ok = passes(p1_checks)
    decision = {"P1": {"checks": p1_checks, "qualifies": p1_ok},
                "review": {"checks": rev_checks, "conditions_hold": passes(rev_checks),
                           "qualifies": p1_ok and passes(rev_checks)}}
    log("decision: P1 %s, review band %s" % ("qualifies" if p1_ok else "does not qualify",
                                            "qualifies" if decision["review"]["qualifies"] else "does not qualify"))

    # ---- 5. reklamlar (özel veri, sadece betimleyici)
    commercial = None
    if promo_mbox_paths():
        redactor, _ = load_redactor()
        promo, _ = load_promo(redactor)
        ps = model.predict_proba([row["text"] for row in promo])
        english = np.array([row["english"] for row in promo])
        commercial = {}
        for r, pi in cells:
            band = review_band(r, PRIMARY_REVIEW, pi, pi_t)
            t1 = thresholds[(r, pi)]["P1"]
            entry = {}
            for subset, m in (("English", english), ("all", np.ones(len(ps), dtype=bool))):
                x = ps[m]
                entry[subset] = {"P0_flagged": flagged_share(x >= t0), "P1_flagged": flagged_share(x >= t1),
                                 "band_flagged": flagged_share(x >= band[1]) if band else None,
                                 "band_review": flagged_share((x > band[0]) & (x < band[1])) if band else None}
            commercial[cell_key(r, pi)] = entry
        log("commercial set: %d promotions, %d English" % (len(ps), int(english.sum())))

    out = {"sanity": sanity, "train_share": pi_t, "primary": {"cost_ratio": r_p, "prevalence": pi_p},
           "primary_review_cost": PRIMARY_REVIEW,
           "thresholds": {cell_key(r, pi): th for (r, pi), th in thresholds.items()},
           "review_bands": {"c=%g" % c: b for c, b in bands.items()},
           "decision": decision, "sets": results, "commercial": commercial,
           "runtime_s": round(time.time() - started)}
    out = clean(out)
    f = open(os.path.join(REPORTS, "decision_policy.json"), "w", encoding="utf-8")
    json.dump(out, f, indent=2)
    f.close()
    write_markdown(out)
    log("done in %ds -> reports/DECISION_POLICY.md" % (time.time() - started))
    return 0


# ---- rapor

def pct(x, ci=None, digits=1):
    if x is None:
        return "-"
    s = "%.*f%%" % (digits, 100 * x)
    if ci is not None:
        s = s + " (%.*f–%.*f)" % (digits, 100 * ci[0], digits, 100 * ci[1])
    return s


def cost(e, with_ci=True):
    if e is None:
        return "-"
    s = "%.1f" % e["cost"]
    if with_ci and e.get("cost_ci"):
        s = s + " (%.1f–%.1f)" % tuple(e["cost_ci"])
    return s


def share(x):
    return "-" if x is None else "%.1f%%" % (100 * x["rate"])


def write_markdown(r):
    sets = r["sets"]
    prim = cell_key(r["primary"]["cost_ratio"], r["primary"]["prevalence"])
    yn = lambda b: "yes" if b else "**no**"
    L = ["# Decision policy under cost and prevalence", ""]
    L.append("Generated by `experiments_decision.py`. Protocol, written before the first run: "
             "[`DECISION_POLICY_PROTOCOL.md`](DECISION_POLICY_PROTOCOL.md). The model is not changed, only "
             "the threshold and an optional review band.")
    L.append("")
    L.append("Costs are in false-alarm units: a missed phishing email costs `r`, a review costs `c`. "
             "**Cost** is the expected cost per 1,000 emails at phishing share `π`, with 95% bootstrap "
             "intervals. P0 = production (0.519), P1 = prior shift (the candidate), P2 = lowest cost on "
             "training out-of-fold scores, oracle = lowest cost on the hold-out itself (optimistic).")
    L.append("")

    d = r["decision"]
    L.append("## Decision")
    L.append("")
    L.append("**P1 " + ("qualifies" if d["P1"]["qualifies"] else "does not qualify") + "; the review band "
             + ("qualifies" if d["review"]["qualifies"] else "does not qualify") + ".** Primary cell: `r = 100`, "
             "`π = 1%%`; review cost `c = %g`." % r["primary_review_cost"])
    L.append("")
    L.append("| set | P0 cost | P1 cost | P1 − P0 (95% int.) | P1 lower | P1 not significantly higher |")
    L.append("|---|---:|---:|---:|---|---|")
    for name in HOLDOUTS:
        row = sets[name]["cells"][prim]
        df = row["P1_minus_P0"]
        L.append("| %s | %s | %s | %+.1f (%+.1f to %+.1f) | %s | %s |" % (
            name, cost(row["P0"]), cost(row["P1"]), df["difference"], df["ci"][0], df["ci"][1],
            yn(d["P1"]["checks"][name]["lower"]), yn(not d["P1"]["checks"][name]["significantly_higher"])))
    L.append("")
    L.append("| set | P1 cost | P1 + review cost | difference (95% int.) | lower | not significantly higher |")
    L.append("|---|---:|---:|---:|---|---|")
    key = "c=%g" % r["primary_review_cost"]
    for name in HOLDOUTS:
        rv = sets[name]["review"][key]
        df = rv["minus_P1"]
        L.append("| %s | %s | %s | %+.1f (%+.1f to %+.1f) | %s | %s |" % (
            name, cost(sets[name]["cells"][prim]["P1"]), cost(rv), df["difference"], df["ci"][0], df["ci"][1],
            yn(d["review"]["checks"][name]["lower"]),
            yn(not d["review"]["checks"][name]["significantly_higher"])))
    L.append("")

    L.append("## Thresholds")
    L.append("")
    L.append("| `r` | `π` | P1 (formula) | P2 (training) | oracle A | oracle B | oracle C |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|")
    for k, th in r["thresholds"].items():
        rr, pi = k.replace("r=", "").replace("pi=", "").split(", ")
        L.append("| %s | %s%% | %.4f | %.4f | %s |" % (
            rr, "%g" % (100 * float(pi)), th["P1"], th["P2"],
            " | ".join("%.4f" % sets[n]["cells"][k]["oracle"]["band"][0] for n in HOLDOUTS)))
    L.append("")

    L.append("## Expected cost per 1,000 emails")
    L.append("")
    for pi in PREVALENCES:
        L.append("### %g%% phishing" % (100 * pi))
        L.append("")
        L.append("| `r` | policy | training | A | B | C |")
        L.append("|---:|---|---:|---:|---:|---:|")
        for rr in COST_RATIOS:
            k = cell_key(rr, pi)
            for pol in ("P0", "P1", "P2", "oracle"):
                cells = [cost(sets[n]["cells"][k].get(pol)) for n in ("training",) + HOLDOUTS]
                L.append("| %g | %s | %s |" % (rr, pol, " | ".join(cells)))
            L.append("| %g | best trivial | %s |" % (rr, " | ".join(
                ["%.1f" % sets["training"]["cells"][k]["trivial_cost"]] * 4)))
        L.append("")

    L.append("## Primary cell in detail")
    L.append("")
    L.append("| set | policy | threshold | missed phishing (FNR) | false alarms (FPR) | precision at 1% | cost |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for name in ("training",) + HOLDOUTS:
        row = sets[name]["cells"][prim]
        for pol in ("P0", "P1", "P2", "oracle"):
            e = row.get(pol)
            if e is None:
                continue
            L.append("| %s | %s | %.4f | %s | %s | %s | %s |" % (
                name, pol, e["band"][0], pct(e["fnr"], e.get("fnr_ci")), pct(e["fpr"], e.get("fpr_ci"), 2),
                pct(e.get("precision"), e.get("precision_ci")), cost(e)))
    L.append("")

    L.append("## Regret against the oracle")
    L.append("")
    L.append("Extra cost per 1,000 emails over the hold-out's own best threshold.")
    L.append("")
    L.append("| `r` | `π` | A: P0 / P1 | B: P0 / P1 | C: P0 / P1 |")
    L.append("|---:|---:|---:|---:|---:|")
    for rr, pi in [(rr, pi) for pi in PREVALENCES for rr in COST_RATIOS]:
        k = cell_key(rr, pi)
        L.append("| %g | %g%% | %s |" % (rr, 100 * pi, " | ".join(
            "%.1f / %.1f" % (sets[n]["cells"][k]["regret"]["P0"], sets[n]["cells"][k]["regret"]["P1"])
            for n in HOLDOUTS)))
    L.append("")

    L.append("## Calibration transfer")
    L.append("")
    L.append("ECE of the probability on each hold-out, as is (it assumes %.1f%% phishing) and re-read at the "
             "hold-out's own share. Training out-of-fold ECE: %.4f." % (
                 100 * r["train_share"], r["sanity"]["training_oof_calibration"]["ece"]))
    L.append("")
    L.append("| set | phishing share | ECE as is | ECE re-read |")
    L.append("|---|---:|---:|---:|")
    for name in HOLDOUTS:
        c = sets[name]["calibration"]
        L.append("| %s | %.1f%% | %.4f | %.4f |" % (name, 100 * c["share"], c["as_is"]["ece"], c["reread"]["ece"]))
    L.append("")
    for name in HOLDOUTS:
        L.append("Reliability, hold-out %s (re-read):" % name)
        L.append("")
        L.append("| bin | emails | mean predicted | observed phishing |")
        L.append("|---|---:|---:|---:|")
        for b in sets[name]["calibration"]["reliability"]:
            L.append("| %.1f–%.1f | %d | %.3f | %.3f |" % (b["bin"][0], b["bin"][1], b["n"], b["mean_predicted"],
                                                       b["observed"]))
        L.append("")

    L.append("## Review band at the primary cell")
    L.append("")
    L.append("Review rate is the share of all mail, at 1% phishing, that goes to an analyst.")
    L.append("")
    L.append("| `c` | band | set | review rate | missed phishing | false alarms | cost incl. review |")
    L.append("|---:|---|---|---:|---:|---:|---:|")
    for key, band in r["review_bands"].items():
        if band is None:
            continue
        for name in ("training",) + HOLDOUTS:
            e = sets[name]["review"][key]
            L.append("| %s | %.3f–%.4f | %s | %s | %s | %s | %s |" % (
                key[2:], band[0], band[1], name, pct(e["review_rate"]), pct(e["fnr"], e.get("fnr_ci")),
                pct(e["fpr"], e.get("fpr_ci"), 2), cost(e)))
    L.append("")
    L.append("Where no legitimate email is flagged, the bootstrap interval is 0–0 by construction. The Wilson upper "
             "bounds for 0 false alarms are " + ", ".join(
                 "%.2f%% (%s, %d emails)" % (100 * wilson_ci(0, sets[n]["n_legit"])[1], n, sets[n]["n_legit"])
                 for n in HOLDOUTS) + ".")
    L.append("")
    L.append("Risk–coverage (every fifth point of %d; all of them in the JSON). Coverage = share of mail decided "
             "without an analyst; auto cost = cost per 1,000 of those emails." % N_CURVE)
    L.append("")
    L.append("| `c` | " + " | ".join("%s: coverage / auto cost / total cost" % n for n in HOLDOUTS) + " |")
    L.append("|---:|" + "---:|" * len(HOLDOUTS))
    n_pts = len(sets["A"]["risk_coverage"])
    for i in list(range(0, n_pts, 5)) + [n_pts - 1]:
        cells = []
        for name in HOLDOUTS:
            p = sets[name]["risk_coverage"][i]
            auto = "-" if p["auto_cost_per_1000_decided"] is None else "%.1f" % p["auto_cost_per_1000_decided"]
            cells.append("%s / %s / %.1f" % (pct(p["coverage"]), auto, p["total_cost"]))
        L.append("| %.3f | %s |" % (sets["A"]["risk_coverage"][i]["c"], " | ".join(cells)))
    L.append("")

    if r["commercial"]:
        L.append("## Commercial mail (private, descriptive)")
        L.append("")
        L.append("Every one of these emails is legitimate, so every flag is a false alarm. With the review band "
                 "(`c = %g`), \"flagged / review\" splits the emails that are not passed." % r["primary_review_cost"])
        L.append("")
        L.append("| `r` | `π` | English: P0 | English: P1 | English: band flagged / review | all: P0 | all: P1 | "
                 "all: band flagged / review |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k, e in r["commercial"].items():
            rr, pi = k.replace("r=", "").replace("pi=", "").split(", ")
            cells = []
            for subset in ("English", "all"):
                x = e[subset]
                cells.extend([share(x["P0_flagged"]), share(x["P1_flagged"]),
                              share(x["band_flagged"]) + " / " + share(x["band_review"])])
            L.append("| %s | %g%% | %s |" % (rr, 100 * float(pi), " | ".join(cells)))
        L.append("")

    s = r["sanity"]
    L.append("## Sanity check")
    L.append("")
    L.append("- Threshold recomputed from the out-of-fold scores: %.4f (saved: %.4f)." % (
        s["recomputed_threshold"], s["saved_threshold"]))
    L.append("- Platt slope and intercept: recomputed %.4f, %.4f; saved %.4f, %.4f." % tuple(
        s["recomputed_platt"] + s["saved_platt"]))
    L.append("- Phishing share in training: %.4f." % s["recomputed_train_share"])
    L.append("")
    f = open(os.path.join(REPORTS, "DECISION_POLICY.md"), "w", encoding="utf-8")
    f.write("\n".join(L))
    f.close()


if __name__ == "__main__":
    sys.exit(main())
