# train.py ve experiments.py'nin ortak kullandığı yardımcılar:
# dataset okuma, metrikler, threshold seçimi.

import csv
import math
import re
import sys

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

POSITIVE_LABELS = {"1", "phishing", "phishing email", "spam", "malicious", "true"}
NEGATIVE_LABELS = {"0", "legit", "legitimate", "ham", "safe email", "safe", "false"}


def load_csv(path, text_col="text", label_col="label"):
    csv.field_size_limit(sys.maxsize)
    texts = []
    labels = []
    sources = []
    f = open(path, "r", encoding="utf-8", errors="replace", newline="")
    for row in csv.DictReader(f):
        text = row.get(text_col) or ""
        raw = (row.get(label_col) or "").strip().lower()
        if text.strip() == "":
            continue
        if raw in POSITIVE_LABELS:
            labels.append(1)
        elif raw in NEGATIVE_LABELS:
            labels.append(0)
        else:
            continue
        texts.append(text)
        sources.append(row.get("source") or "")
    f.close()
    return texts, np.array(labels), sources


# Raporlar yayınlanıyor: hata örneklerinde mail listesi katılımcılarının adresleri ve
# isimleri geçiyordu. Gövdeyi hiç yazmıyoruz; konu satırında da adres/link maskeleniyor.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+(?:@| at )[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_SUBJECT_MAX = 100


def describe_email(text):
    """Rapor için kişisel veri içermeyen özet: konu (maskeli) + gövde uzunluğu + link sayısı."""
    subject = ""
    body = text
    if text.startswith("Subject:"):
        first, _, body = text.partition("\n\n")
        subject = first[len("Subject:"):].strip()
    subject = _URL_RE.sub("<link>", _EMAIL_RE.sub("<email>", subject))
    if len(subject) > _SUBJECT_MAX:
        subject = subject[:_SUBJECT_MAX].rstrip() + "…"
    links = len(_URL_RE.findall(body))
    return ("Subject: " + (subject or "(none)") + " [body: " + str(len(body.strip()))
            + " chars, " + str(links) + " links]")


def best_f1_threshold(y_true, scores):
    """Validation setinde F1'i maksimize eden threshold. Test setine dokunmuyoruz."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # son precision/recall noktasının threshold'u yok
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def threshold_for_fpr(y_true, scores, max_fpr):
    """Validation'da yanlış alarm oranı max_fpr'ı geçmeyen en düşük threshold
    (o kısıt altında en yüksek recall)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    legit = np.sort(scores[y_true == 0])[::-1]
    if len(legit) == 0:
        return float(scores.min())
    allowed = int(np.floor(max_fpr * len(legit)))
    if allowed >= len(legit):
        return float(scores.min())
    # en yüksek skorlu `allowed` meşru mail threshold'u geçebilir, sonraki geçemez
    return float(np.nextafter(legit[allowed], np.inf))


def wilson_ci(k, n, z=1.96):
    """k/n oranı için %95 Wilson güven aralığı. Hold-out setler birkaç yüz mail;
    %0.3 gibi bir yanlış alarm oranı aslında 3 mail demek."""
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def mcnemar_exact(y_true, pred_a, pred_b):
    """İki modeli aynı maillerde karşılaştırır (eşleştirilmiş, kesin binom testi).
    b: sadece A'nın doğru bildiği mailler, c: sadece B'nin. p küçükse fark gürültü değil."""
    y = np.asarray(y_true)
    ok_a = np.asarray(pred_a) == y
    ok_b = np.asarray(pred_b) == y
    b = int((ok_a & ~ok_b).sum())
    c = int((~ok_a & ok_b).sum())
    n = b + c
    if n == 0:
        return {"only_a_correct": b, "only_b_correct": c, "p_value": 1.0}
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return {"only_a_correct": b, "only_b_correct": c, "p_value": round(min(1.0, 2 * tail), 4)}


def calibration_stats(y_true, probs, bins=10):
    """Brier skoru ve ECE (expected calibration error, eşit genişlikte 10 kutu)."""
    y = np.asarray(y_true)
    p = np.asarray(probs, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece = ece + m.mean() * abs(p[m].mean() - y[m].mean())
    return {"brier": round(float(brier_score_loss(y, p)), 4), "ece": round(float(ece), 4)}


def format_rate(m, key):
    """"97.8% (96.0–98.8)" - oran ve %95 güven aralığı."""
    lo, hi = m.get(key + "_ci", [None, None])
    if lo is None:
        return "%.1f%%" % (100 * m[key])
    return "%.1f%% (%.1f–%.1f)" % (100 * m[key], 100 * lo, 100 * hi)


def compute_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "threshold": round(float(threshold), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "fpr": round(float(fp) / max(fp + tn, 1), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "recall_ci": wilson_ci(int(tp), int(tp + fn)),
        "fpr_ci": wilson_ci(int(fp), int(fp + tn)),
        "precision_ci": wilson_ci(int(tp), int(tp + fp)),
    }
    # tek sınıflı alt kümelerde AUC tanımsız
    if len(set(y_true.tolist())) == 2:
        m["roc_auc"] = round(roc_auc_score(y_true, scores), 4)
        m["pr_auc"] = round(average_precision_score(y_true, scores), 4)
    else:
        m["roc_auc"] = None
        m["pr_auc"] = None
    return m


def format_table(rows, columns):
    """rows: [(name, metrics_dict)] -> markdown tablo."""
    out = ["| model | " + " | ".join(columns) + " |"]
    out.append("|---|" + "---:|" * len(columns))
    for name, m in rows:
        cells = []
        for c in columns:
            v = m.get(c)
            cells.append("-" if v is None else ("%.4f" % v if isinstance(v, float) else str(v)))
        out.append("| " + name + " | " + " | ".join(cells) + " |")
    return "\n".join(out)


def format_table_ci(rows):
    """Hold-out raporları için: recall ve yanlış alarm %95 güven aralığıyla."""
    out = ["| model | precision | recall (95% CI) | F1 | false alarms (95% CI) | ROC-AUC |"]
    out.append("|---|---:|---:|---:|---:|---:|")
    for name, m in rows:
        auc = "-" if m.get("roc_auc") is None else "%.4f" % m["roc_auc"]
        out.append("| " + name + " | %.3f | " % m["precision"] + format_rate(m, "recall")
                   + " | %.3f | " % m["f1"] + format_rate(m, "fpr") + " | " + auc + " |")
    return "\n".join(out)


def near_duplicate_groups(texts, threshold=0.8, chunk=500):
    """TF-IDF cosine benzerliği >= threshold olan mailleri aynı gruba koyar (union-find).
    Aynı kampanyanın ID'si değişmiş varyantları hem train'de hem test'te olursa
    in-domain skor şişer - split'i bu gruplara göre yapınca bu sızıntı kapanır.
    Döner: her mail için grup numarası."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(texts)
    # sık kelimeler her çifti "benzer" yapmasın: stopword'ler ve çok yaygın terimler dışarıda
    X = TfidfVectorizer(sublinear_tf=True, stop_words="english", max_df=0.5, min_df=2,
                        dtype=np.float32).fit_transform(texts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    XT = X.T.tocsr()
    for start in range(0, n, chunk):
        sim = (X[start:start + chunk] @ XT).tocoo()
        rows = sim.row + start
        keep = (sim.data >= threshold) & (rows < sim.col)
        for i, j in zip(rows[keep], sim.col[keep]):
            a, b = find(int(i)), find(int(j))
            if a != b:
                parent[a] = b
    return np.array([find(i) for i in range(n)])


def stratified_group_split(y, groups, test_fraction, seed):
    """Sınıf oranını koruyarak (yaklaşık) ikiye böler; bir grup asla iki tarafa düşmez.
    Döner: (kalan_indeksler, test_indeksleri)"""
    from sklearn.model_selection import StratifiedGroupKFold

    idx = np.arange(len(y))
    splitter = StratifiedGroupKFold(n_splits=int(round(1 / test_fraction)), shuffle=True, random_state=seed)
    rest, test = next(splitter.split(idx, y, groups))
    return rest, test


def group_split_single_class(groups, seed, fractions=(0.7, 0.15, 0.15)):
    """Tek sınıflı setler için (modern meşru, eski Nazario) gruplu 70/15/15.
    Gruplar karıştırılıp mail sayısına göre sırayla dilimlere doldurulur."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    order = np.random.RandomState(seed).permutation(len(uniq))
    sizes = {g: c for g, c in zip(*np.unique(groups, return_counts=True))}
    limits = np.cumsum(fractions) * len(groups)
    part_of = {}
    filled = 0
    for g in uniq[order]:
        part_of[g] = int(np.searchsorted(limits, filled, side="right"))
        filled = filled + sizes[g]
    part = np.array([min(part_of[g], len(fractions) - 1) for g in groups])
    idx = np.arange(len(groups))
    return tuple(idx[part == k] for k in range(len(fractions)))
