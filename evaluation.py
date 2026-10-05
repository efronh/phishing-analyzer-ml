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


# gerçek gelen kutusunda (provider filtrelerinden sonra) phishing payı - test setleri %22-33
REAL_PREVALENCES = (0.001, 0.01, 0.05)


def precision_at_prevalence(confusion, prevalence, n_samples=20000, seed=0):
    """Test setindeki recall ve yanlış alarm oranı, phishing payı `prevalence` olan bir
    gelen kutusunda hangi precision'a karşılık gelir? (Bayes: TPR*pi / (TPR*pi + FPR*(1-pi)))
    Aralık: recall ve FPR için Jeffreys Beta posteriorundan örnekleme (0 yanlış alarmda da tanımlı)."""
    tp, fn, fp, tn = confusion["tp"], confusion["fn"], confusion["fp"], confusion["tn"]
    pi = prevalence

    def ppv(tpr, fpr):
        return tpr * pi / np.maximum(tpr * pi + fpr * (1 - pi), 1e-300)

    rng = np.random.default_rng(seed)
    tpr = rng.beta(tp + 0.5, fn + 0.5, n_samples)
    fpr = rng.beta(fp + 0.5, tn + 0.5, n_samples)
    lo, hi = np.percentile(ppv(tpr, fpr), [2.5, 97.5])
    point = ppv(tp / max(tp + fn, 1), fp / max(fp + tn, 1))
    return {"prevalence": pi, "precision": round(float(point), 4),
            "precision_ci": [round(float(lo), 4), round(float(hi), 4)]}


def pr_auc_at_prevalence(y_true, scores, prevalence):
    """PR-AUC de test setinin phishing payına bağlı. Meşru mailleri yeniden ağırlıklandırıp
    payı `prevalence`'a çekerek, aynı skorların o gelen kutusundaki PR-AUC'sini hesaplar."""
    y = np.asarray(y_true)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    w_neg = (1 - prevalence) / prevalence * pos / neg
    weights = np.where(y == 1, 1.0, w_neg)
    return round(float(average_precision_score(y, scores, sample_weight=weights)), 4)


def prevalence_rows(y_true, scores, threshold, prevalences=REAL_PREVALENCES):
    """Rapor için: her gerçekçi phishing payında precision (aralıklı) ve PR-AUC."""
    m = compute_metrics(y_true, scores, threshold)
    out = []
    for pi in prevalences:
        r = precision_at_prevalence(m["confusion"], pi)
        r["pr_auc"] = pr_auc_at_prevalence(y_true, scores, pi)
        out.append(r)
    return out


def format_prevalence_table(named_rows):
    """named_rows: [(set adı, test payı, test PR-AUC, prevalence_rows)] -> markdown."""
    pis = [r["prevalence"] for r in named_rows[0][3]]
    head = "| set | test mix: PR-AUC | " + " | ".join(
        "%g%% phishing: precision (95%% int.) / PR-AUC" % (100 * p) for p in pis) + " |"
    out = [head, "|---|---:|" + "---:|" * len(pis)]
    for name, share, test_pr_auc, rows in named_rows:
        cells = ["%.1f%% (%.1f–%.1f) / %s" % (100 * r["precision"], 100 * r["precision_ci"][0],
                                             100 * r["precision_ci"][1],
                                             "-" if r.get("pr_auc") is None else "%.3f" % r["pr_auc"])
                 for r in rows]
        out.append("| %s (%.0f%% phishing) | %.4f | " % (name, 100 * share, test_pr_auc)
                   + " | ".join(cells) + " |")
    return "\n".join(out)


# ---- maliyete ve phishing payına göre karar (bkz. reports/DECISION_POLICY_PROTOCOL.md)
# Maliyet birimi bir false alarm; r = kaçan phishing'in maliyeti / false alarm'ın maliyeti.

def _odds(p):
    return p / (1 - p)


def shift_probability(probs, prevalence, train_share):
    """Kalibre olasılık eğitimdeki phishing payını varsayar. Pay `prevalence` olunca aynı mailin
    olasılığı: odds, k = odds(prevalence) / odds(train_share) ile çarpılır. Sadece payın
    değiştiği, phishing ve meşru maillerin kendi içinde aynı kaldığı varsayımıyla doğru."""
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1 - 1e-12)
    o = _odds(p) * _odds(prevalence) / _odds(train_share)
    return o / (1 + o)


def cost_threshold(cost_ratio, prevalence, train_share):
    """Payı `prevalence` olan kutuda beklenen maliyeti en düşük tutan eşik, eğitim payındaki
    olasılık üzerinde: p_d > 1 / (1 + r)  <=>  p > 1 / (1 + r·k)."""
    k = _odds(prevalence) / _odds(train_share)
    return 1.0 / (1.0 + cost_ratio * k)


def review_band(cost_ratio, review_cost, prevalence, train_share):
    """Chow kuralı: geçir (maliyet r·p_d), işaretle (1 − p_d) veya analiste gönder (c; analist
    hep doğru bilir). İnceleme bandı c/r < p_d < 1 − c, sadece c < r/(1+r) iken var.
    Döner: eğitim payındaki olasılık üzerinde (alt, üst), band yoksa None."""
    r, c = cost_ratio, review_cost
    if c >= r / (1.0 + r):
        return None
    lo, hi = shift_probability([c / r, 1 - c], train_share, prevalence)
    return float(lo), float(hi)


def policy_rates(y_true, scores, lo, hi):
    """p >= hi işaretlenir, lo < p < hi incelemeye gider, gerisi geçer. Tek eşik: lo = hi = t.
    fnr = geçirilen phishing (incelemeye giden phishing yakalanmış sayılır)."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(scores, dtype=float)
    flag = p >= hi
    review = (p > lo) & ~flag
    return {"fnr": float((~flag & ~review)[y].mean()), "fpr": float(flag[~y].mean()),
            "review_phishing": float(review[y].mean()), "review_legit": float(review[~y].mean())}


def expected_cost(rates, cost_ratio, prevalence, review_cost=0.0):
    """Mail başına beklenen maliyet (false alarm biriminde), sınıf bazındaki oranlardan."""
    pi = prevalence
    return (pi * (cost_ratio * rates["fnr"] + review_cost * rates["review_phishing"])
            + (1 - pi) * (rates["fpr"] + review_cost * rates["review_legit"]))


_LOGIT_LIMIT = math.log(1e-4 / (1 - 1e-4))
COST_GRID = 1 / (1 + np.exp(-np.linspace(_LOGIT_LIMIT, -_LOGIT_LIMIT, 2001)))


def min_cost_threshold(y_true, scores, cost_ratio, prevalence, grid=COST_GRID):
    """Beklenen maliyeti verinin kendisinde en düşük tutan eşik. Olasılığın değerine değil,
    sadece sınıf bazındaki skor dağılımına güvenir. Maliyet basamaklı olduğu için en düşük
    maliyet çoğu zaman bir aralık: en uzun aralığın ortası (grid logit'te eşit aralıklı).
    Döner: (eşik, o eşikteki maliyet)."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(scores, dtype=float)
    pos = np.sort(p[y])
    neg = np.sort(p[~y])
    fnr = np.searchsorted(pos, grid, side="left") / len(pos)
    fpr = 1 - np.searchsorted(neg, grid, side="left") / len(neg)
    cost = prevalence * cost_ratio * fnr + (1 - prevalence) * fpr
    best = np.isclose(cost, cost.min(), rtol=1e-9, atol=1e-15)
    run_start, best_run = None, (0, 0)
    for i, b in enumerate(np.append(best, False)):
        if b and run_start is None:
            run_start = i
        elif not b and run_start is not None:
            if i - run_start > best_run[1] - best_run[0]:
                best_run = (run_start, i)
            run_start = None
    mid = (best_run[0] + best_run[1] - 1) // 2
    return float(grid[mid]), float(cost[mid])


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
