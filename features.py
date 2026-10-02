# Rule-based analyzer çıktısını ML modeline sayısal feature olarak veriyoruz.
# Böylece model hem metni (TF-IDF) hem de elle yazdığımız kuralları görüyor.
#
# v2: Feature'ların hepsi sınırlı (bounded). v1'de url_count, url_score gibi ham
# toplamlar vardı ve lineer model bunları sınırsız extrapolate ediyordu:
# 20-30 linkli meşru duyuru mailleri (release announcement vs.) sırf link
# sayısı yüzünden phishing çıkıyordu. Gerçek phishing'de ortanca link sayısı 1.
# Artık toplam yerine oran / maksimum / var-yok, sayılar için de log kullanıyoruz.
# has_subject kaldırıldı: Subject satırı Kaggle'da hiç yok, diğer kaynaklarda hep var -
# içerik değil kaynak sinyaliydi.

import math
import re
from functools import lru_cache

from analyzer import PhishingEmailAnalyzer, URL_REGEX, is_valid_ip, host_is_shortener

FEATURE_NAMES = [
    "rule_score_log",
    "keyword_score",
    "keyword_hits_log",
    "has_url",
    "url_count_log",
    "http_url_fraction",
    "suspicious_url_fraction",
    "max_url_issue_score",
    "ip_url",
    "shortener_url",
    "domain_warning_fraction",
    "ip_in_text",
    "detail_count",
    "text_length_log",
    "caps_word_ratio",
    "exclamation_log",
]


def _log1p(x):
    return math.log(1 + x)


_DEFAULT_ANALYZER = PhishingEmailAnalyzer(resolve_dns=False)


# v1: ham sayılar (link sayısı, url skoru toplamı...). Sadece ablation / karşılaştırma için.
FEATURE_NAMES_V1 = [
    "rule_score",
    "keyword_score",
    "keyword_hit_count",
    "url_score",
    "url_count",
    "http_url_count",
    "url_issue_count",
    "ip_count",
    "domain_warning_count",
    "detail_count",
    "text_length_log",
    "caps_word_ratio",
    "exclamation_count",
    "has_subject",
]


# v3: v2 + "tuzak" (lure) feature'ları. Kaçırılan 2025 phishing'lerinin çoğu linksiz,
# kısa, ödeme/fatura/ek dosya bahaneli mailler - bunları açıkça işaretliyoruz.
LURE_PATTERNS = {
    "payment_pretext": r"\b(invoice|remittance|payment|ach|wire transfer|bank transfer|rfq|"
                       r"quotation|purchase order|payroll|salary|refund|receipt|order confirmation)\b",
    "attachment_pretext": r"(see attached|find attached|attached (file|document|invoice|copy|rfq)|"
                          r"document (shared|ready)|shared with you|review and sign|docu-?sign|"
                          r"e-?sign|view document|open the attachment)",
    "generic_greeting": r"\b(dear (customer|user|client|supplier|colleagues?|sir|madam|member|"
                        r"valued|account holder|beneficiary)|valued customer|hello dear)\b",
    "urgency": r"\b(urgent(ly)?|immediately|asap|within 24 hours|expires?|expired|suspend(ed)?|"
               r"final notice|last warning)\b",
    "credential_request": r"\b(password|log ?in|sign in|verify|confirm your|credentials?|"
                          r"update your (account|details|information))\b",
}
LURE_NAMES = ["lure_" + n for n in LURE_PATTERNS] + ["short_body"]
FEATURE_NAMES_V3 = FEATURE_NAMES + LURE_NAMES
_LURE_REGEXES = [re.compile(p, re.IGNORECASE) for p in LURE_PATTERNS.values()]
SHORT_BODY_CHARS = 300


def feature_names(version=2):
    if version == 1:
        return FEATURE_NAMES_V1
    if version == 3:
        return FEATURE_NAMES_V3
    return FEATURE_NAMES


def extract_features(text, analyzer=None, version=2):
    """Tek bir mail için feature_names(version) sırasında float listesi döner."""
    if analyzer is None:
        # CV ve model karşılaştırmasında aynı mailler defalarca geliyor - cache'le
        return list(_cached_features(text, version))
    if version == 1:
        return _compute_features_v1(text, analyzer)
    if version == 3:
        return _compute_features(text, analyzer) + _lure_features(text)
    return _compute_features(text, analyzer)


@lru_cache(maxsize=100000)
def _cached_features(text, version):
    if version == 1:
        return tuple(_compute_features_v1(text, _DEFAULT_ANALYZER))
    if version == 3:
        return _cached_features(text, 2) + tuple(_lure_features(text))
    return tuple(_compute_features(text, _DEFAULT_ANALYZER))


def _lure_features(text):
    # "Subject: ...\n\n<gövde>" formatı - kısa gövde kontrolü sadece gövdeye bakar
    body = text.split("\n\n", 1)[1] if text.startswith("Subject:") and "\n\n" in text else text
    out = [_log1p(len(rx.findall(text))) for rx in _LURE_REGEXES]
    out.append(1.0 if len(body.strip()) < SHORT_BODY_CHARS else 0.0)
    return out


def _compute_features_v1(text, analyzer):
    result = analyzer.analyze(text)
    http_count = sum(1 for u in result.urls if u.scheme == "http")
    issue_count = sum(len(u.issues) for u in result.urls)
    words = re.findall(r"[A-Za-z]{4,}", text)
    caps_ratio = sum(1 for w in words if w.isupper()) / len(words) if words else 0.0
    return [
        float(result.score),
        float(result.keyword_score),
        float(sum(result.keyword_hits.values())),
        float(result.url_score),
        float(len(result.urls)),
        float(http_count),
        float(issue_count),
        float(len(result.ip_addresses)),
        float(len(result.domain_warnings)),
        float(len(result.details)),
        _log1p(len(text)),
        caps_ratio,
        float(text.count("!")),
        1.0 if analyzer._extract_subject(text) else 0.0,
    ]


def _compute_features(text, analyzer):
    result = analyzer.analyze(text)
    urls = result.urls
    n_urls = len(urls)

    http_count = 0
    suspicious = 0
    max_issue = 0
    ip_url = 0.0
    shortener = 0.0
    hosts = set()
    for u in urls:
        if u.scheme == "http":
            http_count = http_count + 1
        if u.issue_score > 0:
            suspicious = suspicious + 1
        max_issue = max(max_issue, u.issue_score)
        if u.host:
            hosts.add(u.host)
            if is_valid_ip(u.host):
                ip_url = 1.0
            if host_is_shortener(u.host):
                shortener = 1.0

    # domain_warnings "host: uyarı" formatında - kaç farklı host uyarı almış
    warned_hosts = set(w.split(":", 1)[0] for w in result.domain_warnings)

    words = re.findall(r"[A-Za-z]{4,}", text)
    caps_ratio = 0.0
    if len(words) > 0:
        caps_ratio = sum(1 for w in words if w.isupper()) / len(words)

    text_ips = [ip for ip in result.ip_addresses if ip not in hosts]

    return [
        _log1p(result.score),
        float(result.keyword_score),
        _log1p(sum(result.keyword_hits.values())),
        1.0 if n_urls > 0 else 0.0,
        _log1p(n_urls),
        http_count / n_urls if n_urls else 0.0,
        suspicious / n_urls if n_urls else 0.0,
        float(max_issue),
        ip_url,
        shortener,
        len(warned_hosts) / len(hosts) if hosts else 0.0,
        1.0 if text_ips else 0.0,
        float(len(result.details)),
        _log1p(len(text)),
        caps_ratio,
        _log1p(text.count("!")),
    ]


def normalize_text(text):
    # URL'leri tek bir token'a çeviriyoruz, yoksa TF-IDF her linki ayrı kelime sanıyor
    return URL_REGEX.sub(" URLTOKEN ", text)
