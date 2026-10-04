# ML katmanı: TF-IDF (kelime + karakter n-gram) + rule-based feature'lar
# -> lineer classifier. Rule-based analyzer'ın yerine değil, yanına geliyor.

import re

import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from features import extract_features, feature_names, normalize_text

# hybrid skorda ML olasılığının ağırlığı (geri kalanı rule skoru). Sadece deneylerdeki
# karşılaştırma satırı için - ağırlıklar ayarlanmadı ve hybrid çoğu sette ML'den kötü,
# o yüzden CLI kararında kullanılmıyor.
ML_WEIGHT = 0.6

CLASSIFIERS = ["logreg", "svm", "nb"]

# scale edilmiş rule feature'ları bu aralığa kırpılıyor: eğitimde hiç görülmemiş
# uç değerler (ör. 40 linkli bir duyuru) skoru sınırsız itemesin
CLIP_RANGE = 3.0


def clip_scaled(X):
    # joblib için modül seviyesinde
    return np.clip(X, -CLIP_RANGE, CLIP_RANGE)


# "Subject:" etiketi Kaggle'da hiç yok, diğer bütün kaynaklarda var: "subject" kelimesi
# modelin en güçlü 50 feature'ı arasındaydı (ağırlık +2.06) - içerik değil, kaynak/format
# sinyali. Konu satırının kelimeleri kalıyor, sadece etiket siliniyor.
SUBJECT_LABEL_RE = re.compile(r"^subject:", re.IGNORECASE | re.MULTILINE)

# Kayıtlı model ile kodun feature tanımı uyuşmalı; feature/preprocess değişince artırılır
FEATURE_SCHEMA = 3


def preprocess(text):
    # joblib pickle edebilsin diye lambda değil modül seviyesinde fonksiyon
    return SUBJECT_LABEL_RE.sub(" ", normalize_text(text)).lower()


class RuleFeatures(BaseEstimator, TransformerMixin):
    # version=1 eski (sınırsız) feature'lar - sadece karşılaştırma için
    def __init__(self, version=2):
        self.version = version

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            rows.append(extract_features(text, version=self.version))
        return np.array(rows)

    def get_feature_names_out(self, input_features=None):
        return np.array(feature_names(self.version))


# experiments_tuning.py'nin gruplu 5-fold CV ile seçtiği değerler (bkz. reports/TUNING.md).
# Önceki ayar C=4, char 3-5 idi; CV F1 0.9827 -> 0.9844 (fark 1 standart hatanın biraz üstünde).
# Üretim modelinin feature seti. experiments_recall.py'nin önceden belirlenen kuralı
# (validation F1'i en yüksek versiyon) v4+lure'u seçti - fark ~1 mail ama kurala uyuyoruz.
# Deney script'leri feature_version'ı kendileri açıkça veriyor; build_pipeline'ın kendi
# varsayılanı (2) karşılaştırmalar için değişmedi.
PRODUCTION_FEATURE_VERSION = 3

DEFAULT_C = 16.0
DEFAULT_MAX_FEATURES = 50000
DEFAULT_CHAR_NGRAM = (2, 5)


def _make_classifier(name, C=DEFAULT_C):
    if name == "logreg":
        return LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
    if name == "svm":
        return LinearSVC(C=0.5, class_weight="balanced")
    if name == "nb":
        return ComplementNB(alpha=0.3)
    raise ValueError("unknown classifier: " + name)


def build_pipeline(classifier="logreg", use_text=True, use_rules=True, feature_version=2,
                   C=DEFAULT_C, max_features=DEFAULT_MAX_FEATURES, char_ngram=DEFAULT_CHAR_NGRAM,
                   stop_words=None, use_char=True):
    """use_text / use_rules ile ablation yapılabiliyor (hangi feature grubu ne katıyor).
    stop_words="english" / use_char=False: "ton" deneyi için (experiments_tone.py); varsayılanlar
    üretim modelinin ayarı."""
    if not use_text and not use_rules:
        raise ValueError("need at least one feature group")
    if classifier == "nb" and use_rules:
        # NB negatif değer kabul etmiyor, scale edilmiş rule feature'ları veremeyiz
        raise ValueError("nb only works with text features")

    parts = []
    if use_text:
        parts.append(("word", TfidfVectorizer(
            preprocessor=preprocess,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=max_features,
            sublinear_tf=True,
            stop_words=stop_words,
        )))
    if use_text and use_char:
        parts.append(("char", TfidfVectorizer(
            preprocessor=preprocess,
            analyzer="char_wb",
            ngram_range=tuple(char_ngram),
            min_df=3,
            max_features=max_features,
            sublinear_tf=True,
        )))
    if use_rules:
        steps = [("extract", RuleFeatures(feature_version)), ("scale", StandardScaler())]
        if feature_version >= 2:
            steps.append(("clip", FunctionTransformer(clip_scaled, feature_names_out="one-to-one")))
        parts.append(("rules", Pipeline(steps)))

    clf = _make_classifier(classifier, C) if classifier == "logreg" else _make_classifier(classifier)
    return Pipeline([("features", FeatureUnion(parts)), ("clf", clf)])


def pipeline_scores(pipeline, texts):
    """Her classifier için 'phishing'e ne kadar yakın' skoru (olasılık ya da margin)."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(texts)[:, 1]
    return pipeline.decision_function(texts)


class MLResult:
    def __init__(self, probability, threshold, top_signals, base_rate=None):
        self.probability = probability
        self.threshold = threshold
        self.top_signals = top_signals
        # kalibre modelde: olasılık bu phishing oranını varsayıyor
        self.base_rate = base_rate

    @property
    def is_phishing(self):
        return self.probability >= self.threshold

    def summary(self):
        lines = []
        verdict = "PHISHING" if self.is_phishing else "legit"
        if self.base_rate is not None:
            lines.append("ML phishing probability: " + str(round(self.probability * 100, 1)) + "%"
                         + " -> " + verdict + " (threshold " + str(round(self.threshold, 2)) + ")")
            lines.append("  calibrated on training data where " + str(round(self.base_rate * 100))
                         + "% of mail is phishing; in a real inbox phishing is much rarer")
        else:
            lines.append("ML phishing score: " + str(round(self.probability, 3)) + " -> " + verdict
                         + " (threshold " + str(round(self.threshold, 2)) + ", not calibrated)")
        if len(self.top_signals) > 0:
            lines.append("Top ML signals:")
            for name, weight in self.top_signals:
                lines.append("  - " + name + " (+" + str(round(weight, 3)) + ")")
        return "\n".join(lines)


class PhishingClassifier:
    def __init__(self, pipeline=None, threshold=0.5, meta=None, calibrator=None):
        self.pipeline = pipeline if pipeline is not None else build_pipeline()
        self.threshold = threshold
        self.meta = meta if meta is not None else {}
        # None ise ham skor; varsa threshold da kalibre skor üzerinden seçilmiştir
        self.calibrator = calibrator

    def fit(self, texts, labels):
        self.pipeline.fit(texts, labels)
        return self

    def predict_proba(self, texts):
        # 1 = phishing sınıfı. Kalibre ise: eğitimdeki phishing oranında gerçek olasılık
        raw = self.pipeline.predict_proba(texts)[:, 1]
        return self.calibrator.transform(raw) if self.calibrator is not None else raw

    def predict(self, texts):
        return (self.predict_proba(texts) >= self.threshold).astype(int)

    def _linear_parts(self):
        union = self.pipeline.named_steps["features"]
        clf = self.pipeline.named_steps["clf"]
        return union, clf.coef_[0], union.get_feature_names_out()

    def explain(self, text, top_k=5):
        """Bu mailde phishing yönüne en çok iten feature'lar (coef * değer)."""
        union, coef, names = self._linear_parts()
        x = union.transform([text])
        row = x.toarray()[0] if hasattr(x, "toarray") else np.asarray(x)[0]
        contrib = row * coef

        # scale edilmiş rule feature'larında 0 (yok) değeri negatife dönüşüyor ve
        # negatif katsayıyla çarpılınca "sinyal" gibi görünüyor - sadece mailde
        # gerçekten olan feature'ları göster
        rules = self.pipeline.named_steps["features"].transformer_list
        version = 2
        for name, step in rules:
            if name == "rules":
                version = step.named_steps["extract"].version
        raw_rules = dict(zip(feature_names(version), extract_features(text, version=version)))

        order = np.argsort(contrib)[::-1]
        out = []
        for i in order:
            if len(out) >= top_k or contrib[i] <= 0:
                break
            name = str(names[i])
            if name.startswith("rules__") and raw_rules[name[len("rules__"):]] == 0:
                continue
            out.append((name, float(contrib[i])))
        return out

    def top_features(self, k=20):
        """Global olarak en phishing / en legit feature'lar."""
        _, coef, names = self._linear_parts()
        order = np.argsort(coef)
        phishing = [(str(names[i]), float(coef[i])) for i in order[::-1][:k]]
        legit = [(str(names[i]), float(coef[i])) for i in order[:k]]
        return phishing, legit

    def analyze(self, text):
        # karar sadece modelin kendi threshold'u - raporlanan rakamlar da bu karara ait
        prob = float(self.predict_proba([text])[0])
        share = self.meta.get("phishing_share_in_training") if self.calibrator is not None else None
        return MLResult(probability=prob, threshold=self.threshold, top_signals=self.explain(text),
                        base_rate=share)

    def save(self, path):
        joblib.dump({"pipeline": self.pipeline, "threshold": self.threshold, "meta": self.meta,
                     "calibrator": self.calibrator, "feature_schema": FEATURE_SCHEMA}, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        schema = data.get("feature_schema", 1)
        if schema != FEATURE_SCHEMA:
            # eski model yeni feature koduyla sessizce yanlış skor üretirdi
            raise ValueError("model " + path + " was saved with feature schema " + str(schema)
                             + ", this code uses " + str(FEATURE_SCHEMA) + ": retrain with train.py")
        return cls(data["pipeline"], data["threshold"], data.get("meta"), data.get("calibrator"))


# CV threshold seçimi için aday threshold'lar (0.001 adım)
THRESHOLD_GRID = np.round(np.linspace(0.01, 0.99, 981), 3)


def f1_curve(y_true, scores, grid=THRESHOLD_GRID):
    """grid'deki her threshold için F1."""
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(scores)[:, None] >= grid[None, :]
    tp = (pred & y_true[:, None]).sum(axis=0)
    fp = (pred & ~y_true[:, None]).sum(axis=0)
    fn = (~pred & y_true[:, None]).sum(axis=0)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def cv_oof_scores(texts, labels, folds=5, seed=42, n_jobs=None, groups=None, **pipeline_kwargs):
    """k-fold CV ile her örnek için, onu eğitimde görmemiş modelin skoru (out-of-fold).
    groups verilirse aynı gruptaki (yakın kopya) mailler hep aynı fold'a düşer.
    Döner: (oof_skorlar, fold_numaraları)"""
    labels = np.asarray(labels)
    if groups is None:
        splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(texts, labels))
    else:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(texts, labels, groups))

    def one_fold(train_idx, test_idx):
        pipe = build_pipeline(**pipeline_kwargs).fit([texts[i] for i in train_idx], labels[train_idx])
        return pipeline_scores(pipe, [texts[i] for i in test_idx])

    results = Parallel(n_jobs=n_jobs)(delayed(one_fold)(a, b) for a, b in splits)
    oof = np.zeros(len(labels))
    fold = np.zeros(len(labels), dtype=int)
    for k, ((_, test_idx), scores) in enumerate(zip(splits, results)):
        oof[test_idx] = scores
        fold[test_idx] = k
    return oof, fold


def fold_f1_curves(labels, scores, fold):
    """Her fold için ayrı F1-threshold eğrisi (folds x len(THRESHOLD_GRID))."""
    labels = np.asarray(labels)
    return np.array([f1_curve(labels[fold == k], scores[fold == k]) for k in np.unique(fold)])


def cv_f1_curves(texts, labels, folds=5, seed=42, n_jobs=None, **pipeline_kwargs):
    """Kalibrasyonsuz skorlarla fold başına F1 eğrileri (karşılaştırma script'leri için)."""
    oof, fold = cv_oof_scores(texts, labels, folds, seed, n_jobs, **pipeline_kwargs)
    return fold_f1_curves(labels, oof, fold)


class PlattCalibrator:
    """Skor -> olasılık. class_weight="balanced" logistic regression'ın skorları kayık;
    out-of-fold skorlar üzerine tek değişkenli bir lojistik oturtup düzeltiyoruz.
    Monoton dönüşüm: sıralama (ROC-AUC) değişmez."""

    def fit(self, scores, labels):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(scores), np.asarray(labels))
        return self

    def transform(self, scores):
        return self.lr.predict_proba(self._logit(scores))[:, 1]

    @staticmethod
    def _logit(scores):
        p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)


def select_threshold(curves, rule="plateau", grid=THRESHOLD_GRID):
    """rule="max"    : ortalama F1'in zirvesi.
    rule="plateau": zirveye bitişik ve ortalama F1'i (zirve - 1 standart hata) üstünde
                    kalan aralığın ortası. F1 eğrisinin tepesi düz olduğu için zirvenin
                    tam yeri veriye göre kayıyor; aralığın ortası daha az kayar.
    Döner: (threshold, bilgi sözlüğü)"""
    mean = curves.mean(axis=0)
    best = int(np.argmax(mean))
    info = {"peak": float(grid[best]), "peak_f1": round(float(mean[best]), 4)}
    if rule == "max":
        return float(grid[best]), info
    se = curves[:, best].std(ddof=1) / np.sqrt(len(curves))
    ok = mean >= mean[best] - se
    lo = best
    while lo > 0 and ok[lo - 1]:
        lo = lo - 1
    hi = best
    while hi < len(grid) - 1 and ok[hi + 1]:
        hi = hi + 1
    info.update({"plateau": [float(grid[lo]), float(grid[hi])], "f1_se": round(float(se), 5)})
    return round(float(grid[lo] + grid[hi]) / 2, 4), info


def fit_with_cv_threshold(texts, labels, folds=5, seed=42, n_jobs=None, rule="plateau",
                          calibrate=True, groups=None, **pipeline_kwargs):
    """k-fold CV'den: (1) out-of-fold skorlarla Platt kalibrasyonu, (2) kalibre skorlarda
    threshold (tek validation dilimi threshold'u seed'e göre 0.49-0.55 arasında
    oynatıyordu, bkz. reports/THRESHOLD_STABILITY.md), sonra model verinin tamamıyla
    yeniden eğitilir."""
    from evaluation import calibration_stats

    labels = np.asarray(labels)
    oof, fold = cv_oof_scores(texts, labels, folds, seed, n_jobs, groups, **pipeline_kwargs)
    meta = {"phishing_share_in_training": round(float(labels.mean()), 4)}
    calibrator = None
    scores = oof
    if calibrate:
        calibrator = PlattCalibrator().fit(oof, labels)
        scores = calibrator.transform(oof)
        meta["calibration_oof"] = {"raw": calibration_stats(labels, oof),
                                   "platt": calibration_stats(labels, scores)}
    threshold, info = select_threshold(fold_f1_curves(labels, scores, fold), rule)
    pipe = build_pipeline(**pipeline_kwargs).fit(texts, labels)
    name = "plateau center (1-SE)" if rule == "plateau" else "max mean F1"
    meta.update({"threshold_method": str(folds) + "-fold CV, " + name
                 + (", Platt-calibrated" if calibrate else ""), "threshold_cv": info})
    return PhishingClassifier(pipe, threshold, meta, calibrator)
