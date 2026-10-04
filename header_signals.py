# Ham mailin başlıklarından gönderen doğrulaması sinyalleri. ŞİMDİLİK MODELE GİRMİYOR:
# eğitimdeki meşru mail listesi arşivlerinde bu başlıklar hiç yok (%0), yani her başlık
# feature'ı kaynağı öğrenirdi. Sadece ölçüm için (experiments_headers.py).
#
# İki tür sinyal var:
#   - kayıtlı sonuçlar: son alıcı sunucunun Authentication-Results'a yazdığı spf/dkim/dmarc
#   - hesaplananlar: sunucudan bağımsız, sadece başlıkların kendisinden (DKIM d= ile From uyumu vb.)

import re
from email.utils import getaddresses, parseaddr

from features import _url_features, brand_in_foreign_domain, registered_domain

MECHANISMS = ["spf", "dkim", "dmarc"]
SIGNAL_NAMES = [
    "no_auth_results",
    "spf_not_pass",
    "dkim_not_pass",
    "dmarc_not_pass",
    "no_dkim_signature",
    "dkim_not_aligned",
    "return_path_not_aligned",
    "reply_to_other_domain",
    "from_freemail",
]
# "kaydedilmiş sonuç varsa" paydası kullanan sinyaller
CONDITIONAL = {"spf_not_pass": "spf", "dkim_not_pass": "dkim", "dmarc_not_pass": "dmarc"}

FREEMAIL = {
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com", "msn.com", "yahoo.com",
    "ymail.com", "aol.com", "icloud.com", "me.com", "mac.com", "mail.ru", "yandex.ru", "yandex.com",
    "gmx.com", "gmx.de", "gmx.net", "web.de", "proton.me", "protonmail.com", "zoho.com", "mail.com",
    "qq.com", "163.com", "126.com", "seznam.cz", "libero.it", "orange.fr", "wanadoo.fr",
}
_RESULT_RE = {m: re.compile(r"\b" + m + r"\s*=\s*([a-z]+)", re.IGNORECASE) for m in MECHANISMS}
_DKIM_D = re.compile(r"(?:^|;)\s*d\s*=\s*([A-Za-z0-9.-]+)")


def _domain(value):
    """Başlıktaki adresin kayıtlı domain'i ("Ad <a@b.example.com>" -> "example.com")."""
    addr = parseaddr(str(value or ""))[1]
    if "@" not in addr:
        return ""
    return registered_domain(addr.rsplit("@", 1)[1].strip(">. ").lower())


def recorded_results(msg):
    """Her mekanizma için en üstteki (son alıcının) Authentication-Results kaydı; yoksa None."""
    out = dict.fromkeys(MECHANISMS)
    for header in (msg.get_all("Authentication-Results") or []):
        text = str(header)
        for m in MECHANISMS:
            if out[m] is None:
                found = _RESULT_RE[m].search(text)
                if found:
                    out[m] = found.group(1).lower()
    return out


def header_signals(msg):
    """{sinyal: True/False/None}. None: o mail için tanımsız (ör. kayıtlı DMARC sonucu yok)."""
    res = recorded_results(msg)
    from_dom = _domain(msg.get("From"))
    dkim_domains = set()
    for sig in (msg.get_all("DKIM-Signature") or []):
        m = _DKIM_D.search(" ".join(str(sig).split()))
        if m:
            dkim_domains.add(registered_domain(m.group(1).lower()))
    rp = _domain(msg.get("Return-Path"))
    reply = [registered_domain(a.rsplit("@", 1)[1].lower())
             for _, a in getaddresses([str(v) for v in (msg.get_all("Reply-To") or [])]) if "@" in a]
    out = {
        "no_auth_results": not msg.get_all("Authentication-Results"),
        "no_dkim_signature": not msg.get_all("DKIM-Signature"),
        "dkim_not_aligned": bool(from_dom) and from_dom not in dkim_domains,
        "return_path_not_aligned": bool(from_dom and rp) and rp != from_dom,
        "reply_to_other_domain": bool(from_dom and reply) and any(d != from_dom for d in reply),
        "from_freemail": from_dom in FREEMAIL,
    }
    for name, mech in CONDITIONAL.items():
        out[name] = None if res[mech] is None else res[mech] != "pass"
    out["_recorded"] = res
    return out


# ---- ikinci aşama (reports/SENDER_STAGE_PROTOCOL.md)

def verified_sender(signals):
    """Gönderen doğrulanmış: kayıtlı DMARC pass VE From domain'iyle eşleşen DKIM imzası."""
    return signals["dmarc_not_pass"] is False and not signals["dkim_not_aligned"]


def from_host(msg):
    """From adresinin tam host'u ("a@mail.shop.example" -> "mail.shop.example")."""
    addr = parseaddr(str(msg.get("From", "") or ""))[1]
    return addr.rsplit("@", 1)[1].strip(">. ").lower() if "@" in addr else ""


def suspicious_sender(host, text, signals):
    """Doğrulanmış olsa bile rahatlatılmaz: From domain'i sahibi olmadığı bir marka adı taşıyor,
    punycode, linklerden biri ücretsiz hostingde ya da Reply-To başka bir domain'de."""
    free_hosting = _url_features(text)[0] > 0
    return bool((host and brand_in_foreign_domain(host)) or "xn--" in host or free_hosting
                or signals["reply_to_other_domain"])


def sender_relief(msg, text):
    """(doğrulanmış, şüpheli) - ikinci aşamanın ihtiyacı olan iki bayrak."""
    s = header_signals(msg)
    return verified_sender(s), suspicious_sender(from_host(msg), text, s)
