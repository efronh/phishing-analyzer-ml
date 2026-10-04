#!/usr/bin/env python3
# Ticari meşru mail testi (yazarın kendi gelen kutusundaki reklam mailleri, çoğu Türkçe).
# Protokol, sonuçlardan önce yazıldı: reports/COMMERCIAL_PROTOCOL.md
#
# Gizlilik: mailler data/raw/own_promo/ altında kalır (git dışı) ve bu script hiçbir mailin
# gövdesini, konusunu ya da gönderenini yazdırmaz / rapora koymaz - sadece toplam sayılar.
# redact.txt'deki kelimeler (yazarın adı, adresi) metin modele girmeden silinir.
#
# Ölçülenler:
#   1. HTML ve URL sinyallerinin ne sıklıkta tetiklendiği: ticari mail vs eğitimdeki phishing
#      (Nazario 2019-2024) vs eğitimdeki meşru mail (SpamAssassin). Dilden bağımsız.
#   2. Üretim modelinin (v3) ve URL feature'lı v4'ün bu maillerde yanlış alarm oranı,
#      İngilizce ve İngilizce olmayan ayrı (model sadece İngilizce eğitildi).
#
# Kullanım: python experiments_commercial.py

import base64
import json
import mailbox
import os
import random
import re
import sys
import time
from email.utils import getaddresses
from urllib.parse import quote

import numpy as np

from analyzer import URL_REGEX
from email_parsing import message_to_text, parse_bytes
from evaluation import load_csv, mcnemar_exact, near_duplicate_groups, wilson_ci
from features import URL_FEATURE_NAMES, _url_features
from html_signals import SIGNAL_NAMES, html_signals, sender_domain
from ml_model import PhishingClassifier, fit_with_cv_threshold
from prepare_data import RECIPIENT_REGEX, SA_HAM_DIRS, dedup, looks_english
from train import DEFAULT_DATA

RAW = os.path.join("data", "raw")
PROMO_DIR = os.path.join(RAW, "own_promo")
REPORTS = "reports"
SEED = 42
# tek bir marka seti domine etmesin
MAX_PER_SENDER = 30
# bir sinyal "umut verici" sayılır: phishing'in en az %5'inde ve ticari maildekinin en az 3 katı
MIN_PHISH_RATE = 0.05
MIN_RATIO = 3.0
TRAIN_PHISH_FILES = ["nazario-%d.mbox" % y for y in range(2019, 2025)]


def log(msg):
    print("[" + time.strftime("%H:%M:%S") + "] " + msg, flush=True)


def tr_variants(term):
    """Türkçe büyük/küçük harf: "efe" -> "EFE", "Efe"; "i" -> "İ", "ı" -> "I"."""
    up = term.replace("i", "İ").replace("ı", "I").upper()
    low = term.replace("İ", "i").replace("I", "ı").lower()
    return {term, term.lower(), term.upper(), term.title(), up, low, low[:1].replace("i", "İ").upper() + low[1:]}


def load_redactor():
    path = os.path.join(PROMO_DIR, "redact.txt")
    terms = []
    if os.path.exists(path):
        f = open(path, encoding="utf-8", errors="replace")
        terms = [line.strip() for line in f if len(line.strip()) >= 2]
        f.close()
    variants = sorted({v for t in terms for v in tr_variants(t)}, key=len, reverse=True)
    if not variants:
        return None, 0
    # kelime sınırı: "Efe" silinsin ama "Efes" kalsın
    pattern = r"(?<!\w)(" + "|".join(re.escape(v) for v in variants) + r")(?!\w)"
    return re.compile(pattern, re.IGNORECASE), len(terms)


# alıcı adresinin geçebileceği header'lar (Hide My Email adresleri dahil)
RECIPIENT_HEADERS = ["To", "Cc", "Delivered-To", "X-Original-To", "Original-Recipient"]
# metindeki kişisel numaralar - URL'lerin DIŞINDA (URL feature'ları bozulmasın)
PHONE_RE = re.compile(r"(?<!\d)(\+?90[\s.-]?)?\(?0?5\d{2}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\d)"
                      r"|\+\d{1,3}[\s.-]?\(?\d{2,4}\)?([\s.-]?\d{2,4}){2,4}")
CARD_RE = re.compile(r"(?<!\d)(\d{4}[ -]){3}\d{4}(?!\d)|\*{2,}[\s*]*\d{4}")
LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{10,}(?!\d)")


def recipient_variants(msg):
    """Bu mailin alıcı adres(ler)i ve linklerde geçebilecek kodlanmış halleri:
    URL-encoded, iki kez encoded, base64 (standart ve URL-safe), adresin @ öncesi kısmı."""
    out = set()
    values = []
    for h in RECIPIENT_HEADERS:
        values.extend(str(v) for v in (msg.get_all(h) or []))
    for _, addr in getaddresses(values):
        addr = addr.strip().strip("<>;")
        if "@" not in addr:
            continue
        out.update({addr, quote(addr, safe=""), quote(quote(addr, safe=""), safe="")})
        for enc in (base64.b64encode, base64.urlsafe_b64encode):
            out.add(enc(addr.encode()).decode().rstrip("="))
        local = addr.split("@", 1)[0]
        if len(local) >= 4:
            out.add(local)
    return out


def mask_personal(text, msg, redactor):
    """Metin modele girmeden önce: redact.txt kelimeleri, alıcı adres(ler)i (her kodlamada),
    URL dışındaki telefon / kart / uzun numaralar silinir."""
    if redactor is not None:
        text = redactor.sub("", text)
    variants = sorted(recipient_variants(msg), key=len, reverse=True)
    if variants:
        rx = re.compile(r"(?<![\w.%+-])(" + "|".join(re.escape(v) for v in variants) + r")(?![\w%-])",
                        re.IGNORECASE)
        text = rx.sub("", text)
    parts = []
    last = 0
    for m in URL_REGEX.finditer(text):
        parts.append(_mask_numbers(text[last:m.start()]))
        parts.append(m.group(0))
        last = m.end()
    parts.append(_mask_numbers(text[last:]))
    return "".join(parts)


def _mask_numbers(chunk):
    return LONG_NUMBER_RE.sub("", CARD_RE.sub("", PHONE_RE.sub("", chunk)))


def passes_filter(msg, require_dmarc=True):
    """Junk karışmış olabilir: sadece iCloud'un geldiği anda INBOX'a koyduğu ve DMARC'ı
    geçen mailler tutulur. İçeriğe bakmayan, önceden sabitlenen kural (protokol, ek 1).
    require_dmarc=False: DMARC'ın kendisini ölçen deney için (yoksa oran baştan %100 çıkar)."""
    folder = str(msg.get("X-Apple-Movetofolder", "") or "").strip().upper()
    if folder != "INBOX":
        return False
    if not require_dmarc:
        return True
    auth = " ".join(str(v) for v in (msg.get_all("Authentication-Results") or [])).lower()
    return re.search(r"dmarc=pass\b", auth) is not None


def promo_mbox_paths():
    out = []
    for name in sorted(os.listdir(PROMO_DIR)):
        p = os.path.join(PROMO_DIR, name)
        if name.endswith(".partial.mbox"):
            continue
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "mbox")):
            out.append(os.path.join(p, "mbox"))
        elif name.endswith(".mbox") and os.path.isfile(p):
            out.append(p)
    return out


def read_mbox_messages(path):
    for msg in mailbox.mbox(path, factory=lambda fp: parse_bytes(fp.read())):
        yield msg


def load_promo(redactor, require_dmarc=True, extra=None):
    """extra: msg -> değer; sonuç row["extra"]'ya konur (ör. başlık kontrolleri)."""
    rows = []
    filtered = 0
    for path in promo_mbox_paths():
        for msg in read_mbox_messages(path):
            if not passes_filter(msg, require_dmarc):
                filtered = filtered + 1
                continue
            try:
                text = mask_personal(message_to_text(msg), msg, redactor)
                sig = html_signals(msg)
            except Exception:
                continue
            if len(text) < 40:
                continue
            rows.append({"text": text, "label": 0, "source": "own_promo",
                         "sender": sender_domain(msg), "html": sig,
                         "extra": extra(msg) if extra is not None else None})
    n_raw = len(rows)
    rows = dedup(rows)
    by_sender = {}
    for r in rows:
        by_sender.setdefault(r["sender"], []).append(r)
    rng = random.Random(SEED)
    capped = []
    for s in sorted(by_sender):
        items = by_sender[s]
        if len(items) > MAX_PER_SENDER:
            items = rng.sample(items, MAX_PER_SENDER)
        capped.extend(items)
    for r in capped:
        r["english"] = looks_english(r["text"])
    return capped, {"dropped_by_junk_filter": filtered, "messages_read": n_raw, "after_dedup": len(rows), "senders": len(by_sender),
                    "after_sender_cap": len(capped)}


def load_reference():
    """Karşılaştırma için eğitim kaynakları (hold-out'lara dokunulmuyor)."""
    phish, ham = [], []
    for name in TRAIN_PHISH_FILES:
        for msg in read_mbox_messages(os.path.join(RAW, name)):
            try:
                text = RECIPIENT_REGEX.sub("", message_to_text(msg))
                phish.append({"text": text, "html": html_signals(msg)})
            except Exception:
                continue
    for d in SA_HAM_DIRS:
        folder = os.path.join(RAW, d)
        for fname in sorted(os.listdir(folder)):
            if fname == "cmds":
                continue
            f = open(os.path.join(folder, fname), "rb")
            raw = f.read()
            f.close()
            try:
                msg = parse_bytes(raw)
                ham.append({"text": message_to_text(msg), "html": html_signals(msg)})
            except Exception:
                continue
    return phish, ham


def rates(rows):
    n = len(rows)
    out = {"n": n}
    for s in SIGNAL_NAMES:
        k = sum(1 for r in rows if r["html"][s])
        out[s] = {"rate": round(k / n, 4) if n else None, "ci": wilson_ci(k, n)}
    url = [_url_features(r["text"]) for r in rows]
    for i, s in enumerate(URL_FEATURE_NAMES[:-1]):   # sonuncusu sayı, bayrak değil
        k = sum(1 for u in url if u[i] > 0)
        out["url_" + s[4:]] = {"rate": round(k / n, 4) if n else None, "ci": wilson_ci(k, n)}
    k = sum(1 for u in url if any(u))
    out["has_link"] = {"rate": round(k / n, 4) if n else None, "ci": wilson_ci(k, n)}
    return out


# teşhis listesinde bir kelime en az bu kadar mailde geçmeli: tek bir gönderen (en fazla 30 mail)
# İngilizce alt kümenin %18'ini oluşturabiliyor, %25 eşiği marka adlarının listeye girmesini önler
DIAG_MIN_DOC_FREQ = 0.25


def diagnose(model, groups):
    """Sonuç görüldükten SONRA eklenen keşif analizi (protokolde yok): kararı kelime, harf grubu
    ve kural blokları ne kadar itiyor; reklamlarda en çok iten feature'lar hangileri."""
    pipe = model.pipeline
    union = pipe.named_steps["features"]
    coef = pipe.named_steps["clf"].coef_.ravel()
    names = union.get_feature_names_out()
    blocks = {b: np.array([n.startswith(b + "__") for n in names]) for b in ["word", "char", "rules"]}
    out = {"groups": {}}
    for name, texts in groups.items():
        F = union.transform(texts)
        p = model.predict_proba(texts)
        row = {"n": len(texts), "flagged": round(float((p >= model.threshold).mean()), 4),
               "median_probability": round(float(np.median(p)), 4)}
        for b, mask in blocks.items():
            idx = np.where(mask)[0]
            row["mean_logit_" + b] = round(float(np.asarray(F[:, idx].multiply(coef[mask]).sum(axis=1)).mean()), 3)
        out["groups"][name] = row
    E = groups["commercial, English"]
    F = union.transform(E)
    mask = blocks["word"] | blocks["rules"]
    Fm = F[:, np.where(mask)[0]].tocsc()
    cm, nm = coef[mask], names[mask]
    df = np.asarray((Fm > 0).mean(axis=0)).ravel()
    mean_c = np.asarray(Fm.multiply(cm).mean(axis=0)).ravel()
    top = [i for i in np.argsort(-mean_c) if df[i] >= DIAG_MIN_DOC_FREQ][:12]
    out["top_pushes_english"] = [{"feature": str(nm[i]), "mean_contribution": round(float(mean_c[i]), 3),
                                  "share_of_emails": round(float(df[i]), 3)} for i in top]
    return out


def main():
    if not promo_mbox_paths():
        print("No finished .mbox export in " + PROMO_DIR, file=sys.stderr)
        return 1
    started = time.time()
    redactor, n_terms = load_redactor()
    log("redaction terms: " + str(n_terms))
    promo, counts = load_promo(redactor)
    log("commercial set: " + json.dumps(counts) + ", English: " + str(sum(r["english"] for r in promo)))
    phish, ham = load_reference()
    log("reference: " + str(len(phish)) + " training phishing, " + str(len(ham)) + " SpamAssassin ham")

    # 1. sinyal oranları
    groups = {"commercial (all)": promo,
              "commercial, HTML only": [r for r in promo if r["html"]["has_html"]],
              "training phishing (Nazario 2019-24)": phish,
              "training phishing, HTML only": [r for r in phish if r["html"]["has_html"]],
              "SpamAssassin ham": ham}
    signal_rates = {g: rates(rows) for g, rows in groups.items()}
    # umut verici mi? HTML sinyalleri HTML'li mailler içinde, URL sinyalleri bütün maillerde
    verdicts = {}
    for s in SIGNAL_NAMES[1:] + ["url_" + n[4:] for n in URL_FEATURE_NAMES[:-1]]:
        within = "HTML only" if s in SIGNAL_NAMES else None
        p = signal_rates["training phishing, HTML only" if within else "training phishing (Nazario 2019-24)"][s]["rate"]
        c_group = "commercial, HTML only" if within else "commercial (all)"
        c = signal_rates[c_group][s]["rate"]
        n_c = signal_rates[c_group]["n"]
        ratio = p / max(c, 1.0 / max(n_c, 1))
        verdicts[s] = {"phishing_rate": p, "commercial_rate": c, "ratio": round(ratio, 2),
                       "promising": bool(p >= MIN_PHISH_RATE and ratio >= MIN_RATIO)}

    # 2. modelin yanlış alarmları: v3 = kayıtlı üretim modeli, v4 = aynı prosedürle URL feature'lı
    log("loading production model (v3)")
    v3 = PhishingClassifier.load("model.joblib")
    log("training v4 with the production procedure")
    texts, labels = [], []
    for p in DEFAULT_DATA:
        X, y, _ = load_csv(p)
        texts.extend(X)
        labels.append(y)
    labels = np.concatenate(labels)
    v4 = fit_with_cv_threshold(texts, labels, folds=5, seed=SEED, n_jobs=5,
                               groups=near_duplicate_groups(texts), feature_version=4)
    X = [r["text"] for r in promo]
    english = np.array([r["english"] for r in promo])
    pred = {"v3": v3.predict_proba(X) >= v3.threshold, "v4": v4.predict_proba(X) >= v4.threshold}
    false_alarms = {}
    for subset, mask in [("English", english), ("not English", ~english), ("all", np.ones(len(X), bool))]:
        n = int(mask.sum())
        row = {"n": n}
        for name, p in pred.items():
            k = int(p[mask].sum())
            row[name] = {"false_alarms": k, "rate": round(k / n, 4) if n else None, "ci": wilson_ci(k, n)}
        mc = mcnemar_exact(np.zeros(n, int), pred["v4"][mask], pred["v3"][mask]) if n else None
        row["mcnemar_v4_vs_v3"] = mc
        false_alarms[subset] = row

    log("diagnosis (exploratory)")
    Xm, _, _ = load_csv(os.path.join("data", "processed", "modern_legit.csv"))
    Xk, yk, _ = load_csv(os.path.join("data", "processed", "kaggle.csv"))
    rng = random.Random(SEED)
    kaggle_phish = rng.sample([t for t, l in zip(Xk, yk) if l == 1], 1500)
    diagnosis = diagnose(v3, {
        "commercial, English": [t for t, e in zip(X, english) if e],
        "commercial, not English": [t for t, e in zip(X, english) if not e],
        "2024 mailing lists (training, legit)": Xm,
        "SpamAssassin ham (training, legit)": [r["text"] for r in ham],
        "Nazario 2019-24 (training, phishing)": [r["text"] for r in phish],
        "Kaggle 'phishing' (training, sample of 1500)": kaggle_phish,
    })

    results = {"counts": counts, "redaction_terms": n_terms, "max_per_sender": MAX_PER_SENDER,
               "english": int(english.sum()), "signal_rates": signal_rates, "verdicts": verdicts,
               "thresholds": {"v3": v3.threshold, "v4": v4.threshold}, "false_alarms": false_alarms,
               "diagnosis_exploratory": diagnosis,
               "runtime_seconds": round(time.time() - started, 1)}
    os.makedirs(REPORTS, exist_ok=True)
    f = open(os.path.join(REPORTS, "commercial.json"), "w", encoding="utf-8")
    json.dump(results, f, indent=2, default=float)
    f.close()
    write_markdown(results)
    log("Done in " + str(results["runtime_seconds"]) + "s -> reports/COMMERCIAL.md")
    return 0


def pct(x):
    return "-" if x is None else "%.1f%%" % (100 * x)


def write_markdown(r):
    c = r["counts"]
    L = ["# Commercial legitimate mail", ""]
    L.append("Generated by `experiments_commercial.py`. Protocol, written before the first run: "
             "[`COMMERCIAL_PROTOCOL.md`](COMMERCIAL_PROTOCOL.md).")
    L.append("")
    L.append("**Data:** promotional emails from the author's own inbox, mostly Turkish. They are **not "
             "published**, so these numbers cannot be reproduced from this repository. The report holds "
             "counts only: no subject lines, senders or bodies.")
    L.append("")
    L.append("- %d emails dropped by the junk filter (not routed to INBOX by iCloud, or no DMARC pass)."
             % c["dropped_by_junk_filter"])
    L.append("- %d emails read, %d after removing duplicates, %d sender domains, %d after keeping at most "
             "%d per sender. %d of them are English by the project's language rule."
             % (c["messages_read"], c["after_dedup"], c["senders"], c["after_sender_cap"],
                r["max_per_sender"], r["english"]))
    L.append("")
    L.append("## 1. How often each signal fires")
    L.append("")
    L.append("HTML signals are counted among emails that have HTML; URL signals among all emails. "
             "A signal is called **promising** if it fires on at least %d%% of training phishing and at "
             "least %g× as often as on commercial mail (rule fixed in advance). Comparison phishing is "
             "the training corpus (Nazario 2019–2024); no hold-out was used."
             % (int(100 * MIN_PHISH_RATE), MIN_RATIO))
    L.append("")
    sr = r["signal_rates"]
    cols = ["commercial (all)", "commercial, HTML only", "training phishing (Nazario 2019-24)",
            "training phishing, HTML only", "SpamAssassin ham"]
    L.append("| signal | " + " | ".join(cols) + " | phishing ÷ commercial | promising |")
    L.append("|---|" + "---:|" * len(cols) + "---:|---|")
    L.append("| emails | " + " | ".join(str(sr[g]["n"]) for g in cols) + " | | |")
    for s in ["has_html", "has_link"] + list(r["verdicts"]):
        v = r["verdicts"].get(s)
        L.append("| `" + s + "` | " + " | ".join(pct(sr[g][s]["rate"]) for g in cols) + " | "
                 + ("%.1f×" % v["ratio"] if v else "") + " | "
                 + (("**yes**" if v["promising"] else "no") if v else "") + " |")
    L.append("")
    L.append("## 2. False alarms of the model")
    L.append("")
    L.append("Naming: `v3`/`v4` in the JSON are **feature-set** numbers (`feature_version` 3 and 4), not the model "
             "versions of the README. Feature set 3 is the production model (README: v4+lure); feature set 4 adds "
             "the URL features.")
    L.append("")
    L.append("Every email here is legitimate, so each phishing verdict is a false alarm. The model was "
             "trained on English only: the non-English rows measure a language shift as much as "
             "commercial mail, and are not a fair false-alarm rate.")
    L.append("")
    L.append("| subset | emails | production (v4+lure) | production + URL features | McNemar p |")
    L.append("|---|---:|---:|---:|---:|")
    for subset, row in r["false_alarms"].items():
        cells = []
        for name in ["v3", "v4"]:
            m = row[name]
            cells.append("-" if m["rate"] is None else "%s (%.1f–%.1f)" % (pct(m["rate"]), 100 * m["ci"][0],
                                                                         100 * m["ci"][1]))
        p = row["mcnemar_v4_vs_v3"]["p_value"] if row["mcnemar_v4_vs_v3"] else None
        L.append("| %s | %d | %s | %s | %s |" % (subset, row["n"], cells[0], cells[1],
                                                 "-" if p is None else "%.3g" % p))
    L.append("")
    dg = r.get("diagnosis_exploratory")
    if dg:
        L.append("## 3. Why (exploratory, added after seeing the result)")
        L.append("")
        L.append("Not in the protocol: added once the false-alarm rate was known, to find its cause. The "
                 "production model's score is a sum of word, character n-gram and rule-feature contributions "
                 "(plus an intercept). Negative pushes toward legitimate, positive toward phishing.")
        L.append("")
        L.append("| group | emails | flagged | median P(phishing) | words | char n-grams | rule features |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for g, x in dg["groups"].items():
            L.append("| %s | %d | %s | %.3f | %+.2f | %+.2f | %+.2f |" % (
                g, x["n"], pct(x["flagged"]), x["median_probability"], x["mean_logit_word"],
                x["mean_logit_char"], x["mean_logit_rules"]))
        L.append("")
        L.append("Features that push English promotions toward phishing the most, among features present in at "
                 "least %d%% of them (no single sender can reach that share, so no brand name can appear):"
                 % int(100 * DIAG_MIN_DOC_FREQ))
        L.append("")
        L.append("| feature | mean contribution | share of emails |")
        L.append("|---|---:|---:|")
        for t in dg["top_pushes_english"]:
            L.append("| `%s` | %+.3f | %s |" % (t["feature"], t["mean_contribution"], pct(t["share_of_emails"])))
        L.append("")
    f = open(os.path.join(REPORTS, "COMMERCIAL.md"), "w", encoding="utf-8")
    f.write("\n".join(L))
    f.close()


if __name__ == "__main__":
    sys.exit(main())
