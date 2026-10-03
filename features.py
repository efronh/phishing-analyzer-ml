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

from urllib.parse import urlparse

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


# v4: v3 + URL'nin kendisine bakan feature'lar. TF-IDF'te URL'ler tek bir URLTOKEN'a
# dönüşüyor, yani model linkin neye benzediğini sadece buradan görüyor. Seçim sadece eğitim
# kaynaklarına bakılarak yapıldı (bkz. reports/URL_FEATURES_PROTOCOL.md). .php/.asp linkleri
# bilerek yok: oranı phishing'den çok mailin yılını gösteriyordu (2002 ham %9, 2024 ham %1).
#
# Herkesin domain sahibi olmadan servisin kendi domain'inde sayfa / ham dosya yayınlayabildiği
# yerler. Listedeki host'lar veride geçtiği için değil bu tanıma uyduğu için var; meşru
# projelerin de kullandıkları dahil (github.io, netlify.app, s3).
FREE_HOSTING = [
    "ipfs.io", "dweb.link", "cloudflare-ipfs.com", "w3s.link", "nftstorage.link", "fleek.co",
    "r2.dev", "pages.dev", "workers.dev", "web.app", "firebaseapp.com", "appspot.com",
    "firebasestorage.googleapis.com", "storage.googleapis.com", "s3.amazonaws.com",
    "blob.core.windows.net", "web.core.windows.net", "azurewebsites.net",
    "glitch.me", "netlify.app", "vercel.app", "herokuapp.com", "onrender.com", "surge.sh",
    "github.io", "gitlab.io", "repl.co", "replit.app", "ngrok.io", "ngrok-free.app",
    "trycloudflare.com", "weebly.com", "wixsite.com", "000webhostapp.com", "blogspot.com",
    "wordpress.com", "sites.google.com", "webflow.io", "framer.app", "square.site",
    "godaddysites.com", "mystrikingly.com", "carrd.co", "notion.site",
]
# s3-eu-west-1.amazonaws.com, bucket.s3.us-east-2.amazonaws.com gibi bölgesel adresler
S3_REGIONAL = re.compile(r"(^|\.)s3[.-][a-z0-9-]+\.amazonaws\.com$")

# taklit edilen markalar -> markanın kendi domain'leri. Host'u "." ve "-" ile token'lara
# bölüp bakıyoruz ("purchase" içindeki "chase" sayılmasın diye substring değil).
BRANDS = {
    "paypal": ["paypal.com"], "apple": ["apple.com", "icloud.com"],
    "icloud": ["icloud.com", "apple.com"],
    "microsoft": ["microsoft.com", "microsoftonline.com", "live.com", "sharepoint.com", "office.com"],
    "office365": ["office.com", "office365.com", "microsoft.com"],
    "outlook": ["outlook.com", "live.com", "office.com"], "onedrive": ["onedrive.com", "live.com"],
    "sharepoint": ["sharepoint.com"], "amazon": ["amazonaws.com"], "netflix": ["netflix.com"],
    "docusign": ["docusign.com", "docusign.net"], "dhl": ["dhl.com", "dhl.de"],
    "fedex": ["fedex.com"], "usps": ["usps.com"], "wellsfargo": ["wellsfargo.com"],
    "chase": ["chase.com"], "adobe": ["adobe.com"], "dropbox": ["dropbox.com"],
    "wetransfer": ["wetransfer.com"], "linkedin": ["linkedin.com"],
    "facebook": ["facebook.com", "fb.com"], "instagram": ["instagram.com"],
    "google": ["googleapis.com", "googleusercontent.com", "gstatic.com"],
    "coinbase": ["coinbase.com"], "metamask": ["metamask.io"], "binance": ["binance.com"],
}
# co.uk, com.br gibi ikinci seviye uzantılar: kayıtlı domain son 3 etiket
_SECOND_LEVEL = {"co", "com", "net", "org", "gov", "ac", "edu", "ne", "or"}
_REDIRECT_RE = re.compile(r"[?&][^=&#]*=(https?(:|%3a)|www\.)", re.IGNORECASE)
_PCT_RE = re.compile(r"%[0-9a-f]{2}", re.IGNORECASE)
_HOST_TOKENS = re.compile(r"[.-]")

URL_FEATURE_NAMES = [
    "url_free_hosting_fraction",
    "url_redirect_param",
    "url_pct_encoded",
    "url_max_host_hyphens",
    "url_max_subdomain_depth",
    "url_brand_in_foreign_domain",
    "url_punycode",
    "url_distinct_domains_log",
]
FEATURE_NAMES_V4 = FEATURE_NAMES_V3 + URL_FEATURE_NAMES


def registered_domain(host):
    """Yaklaşık kayıtlı domain (Public Suffix List yok): "a.b.paypal.com" -> "paypal.com",
    "x.amazon.co.uk" -> "amazon.co.uk"."""
    parts = [p for p in host.lower().strip(".").split(".") if p]
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in _SECOND_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def is_free_hosting(host):
    host = host.lower()
    if S3_REGIONAL.search(host):
        return True
    return any(host == d or host.endswith("." + d) for d in FREE_HOSTING)


def brand_in_foreign_domain(host):
    """Host'ta bir marka adı geçiyor ama kayıtlı domain o markanın değil:
    "paypal.com.secure-login.xyz", "microsoft-verify.net"."""
    reg = registered_domain(host)
    reg_label = reg.split(".")[0]
    tokens = set(_HOST_TOKENS.split(host.lower()))
    for brand, owned in BRANDS.items():
        if brand in tokens and reg_label != brand and reg not in owned:
            return True
    return False


def _url_features(text):
    urls = []
    for raw in URL_REGEX.findall(text):
        url = raw.rstrip(".,;:!?)")
        try:
            host = (urlparse(url).hostname or "").lower()
        except ValueError:
            host = ""
        if host and not is_valid_ip(host):
            urls.append((url, host))
    if not urls:
        return [0.0] * len(URL_FEATURE_NAMES)
    hosts = set(h for _, h in urls)
    domains = set(registered_domain(h) for h in hosts)
    # kayıtlı domain'in üstündeki etiket sayısı ("a.b.example.com" -> 2), 4'te kesiliyor
    depth = max(max(0, h.count(".") - r.count(".")) for h in hosts for r in [registered_domain(h)])
    hyphens = max(h.count("-") for h in hosts)
    return [
        sum(1 for h in hosts if is_free_hosting(h)) / len(hosts),
        1.0 if any(_REDIRECT_RE.search(u) for u, _ in urls) else 0.0,
        1.0 if any(_PCT_RE.search(u) for u, _ in urls) else 0.0,
        min(hyphens, 4) / 4.0,
        min(depth, 4) / 4.0,
        1.0 if any(brand_in_foreign_domain(h) for h in hosts) else 0.0,
        1.0 if any("xn--" in h for h in hosts) else 0.0,
        _log1p(len(domains)),
    ]


def feature_names(version=2):
    if version == 1:
        return FEATURE_NAMES_V1
    if version == 3:
        return FEATURE_NAMES_V3
    if version == 4:
        return FEATURE_NAMES_V4
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
    if version == 4:
        return _compute_features(text, analyzer) + _lure_features(text) + _url_features(text)
    return _compute_features(text, analyzer)


@lru_cache(maxsize=100000)
def _cached_features(text, version):
    if version == 1:
        return tuple(_compute_features_v1(text, _DEFAULT_ANALYZER))
    if version == 3:
        return _cached_features(text, 2) + tuple(_lure_features(text))
    if version == 4:
        return _cached_features(text, 3) + tuple(_url_features(text))
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
