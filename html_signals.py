# Ham mailin HTML kısmından phishing sinyalleri. ŞİMDİLİK MODELE GİRMİYOR: eğitim verisinde
# phishing'in %98'i HTML, meşru maillerin neredeyse hiçbiri değil (hepsi mailing list), yani
# her HTML feature'ı kaynağı öğrenirdi. Bu modül sadece sinyallerin gerçek ticari maillerde
# ne sıklıkta tetiklendiğini ölçmek için (experiments_commercial.py).
# Sinyaller dilden bağımsız: Türkçe bir bültende de İngilizce bir bültendeki gibi ölçülür.

import re
from html.parser import HTMLParser
from urllib.parse import urlparse

from features import BRANDS, registered_domain

SIGNAL_NAMES = [
    "has_html",
    "html_only",
    "form",
    "password_input",
    "hidden_text",
    "link_text_domain_mismatch",
    "brand_text_foreign_link",
    "image_heavy",
    "meta_refresh",
    "script_or_data_link",
]

# içi gizlenmiş metin: newsletter "preheader"ı da tam olarak bunu yapıyor - ölçmek istediğimiz bu
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(px|pt|em)?\s*(;|$)"
                           r"|max-height\s*:\s*0(px)?\s*(;|$)|opacity\s*:\s*0(\.0+)?\s*(;|$)", re.IGNORECASE)
# link metni bir adres gibi görünüyor mu ("www.paypal.com", "https://...")
_URLISH = re.compile(r"^\s*(https?://)?(www\.)?[a-z0-9-]+(\.[a-z0-9-]+)+(/\S*)?\s*$", re.IGNORECASE)
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
# görünür metni bundan az, en az bir resmi olan mail "resimden ibaret"
IMAGE_HEAVY_TEXT_CHARS = 200
# gizli metin sayılması için gizli elemanın içinde en az bu kadar karakter olmalı
HIDDEN_MIN_CHARS = 20


class _Scan(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []          # açık elemanlar: (tag, gizli_mi)
        self.hidden_chars = 0
        self.visible_chars = 0
        self.images = 0
        self.form = False
        self.password = False
        self.meta_refresh = False
        self.script_link = False
        self.anchors = []        # (href, metin)
        self._a = None
        self._skip = 0           # <script>/<style> içi

    def _hidden(self):
        return any(h for _, h in self.stack)

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "form":
            self.form = True
        if tag == "input" and a.get("type", "").lower() == "password":
            self.password = True
        if tag == "img":
            self.images = self.images + 1
        if tag == "meta" and a.get("http-equiv", "").lower() == "refresh":
            self.meta_refresh = True
        if tag == "a":
            href = a.get("href", "").strip()
            if href.lower().startswith(("javascript:", "data:")):
                self.script_link = True
            self._a = [href, ""]
        if tag in ("script", "style"):
            self._skip = self._skip + 1
        if tag not in _VOID:
            hidden = bool(_HIDDEN_STYLE.search(a.get("style", ""))) or "hidden" in a
            self.stack.append((tag, hidden))

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip > 0:
            self._skip = self._skip - 1
        if tag == "a" and self._a is not None:
            self.anchors.append((self._a[0], self._a[1].strip()))
            self._a = None
        # bozuk HTML'de kapanmayan etiketler olabilir: en yakın aynı etikete kadar kapat
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        if self._skip:
            return
        n = len(data.strip())
        if self._hidden():
            self.hidden_chars = self.hidden_chars + n
        else:
            self.visible_chars = self.visible_chars + n
        if self._a is not None:
            self._a[1] = self._a[1] + data


def _host(url):
    try:
        return (urlparse(url if "://" in url else "http://" + url).hostname or "").lower()
    except ValueError:
        return ""


def _brand_text_foreign_link(text, href):
    """Link metninde marka adı geçiyor ama link o markanın domain'ine gitmiyor."""
    host = _host(href)
    if not host or not href.lower().startswith("http"):
        return False
    reg = registered_domain(host)
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    for brand, owned in BRANDS.items():
        if brand in words and reg.split(".")[0] != brand and reg not in owned:
            return True
    return False


def html_parts(msg):
    """(düz metin parçaları, HTML parçaları) - ekler hariç."""
    plain, html = [], []
    for part in msg.walk():
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        try:
            body = part.get_content()
        except Exception:
            body = (part.get_payload(decode=True) or b"").decode("utf-8", errors="replace")
        (plain if ctype == "text/plain" else html).append(body)
    return plain, html


def html_signals(msg):
    """Tek bir mail için {sinyal: bool}."""
    plain, html = html_parts(msg)
    out = dict.fromkeys(SIGNAL_NAMES, False)
    if not html:
        return out
    out["has_html"] = True
    out["html_only"] = not any(p.strip() for p in plain)
    s = _Scan()
    try:
        for h in html:
            s.feed(h)
        s.close()
    except Exception:
        pass
    out["form"] = s.form
    out["password_input"] = s.password
    out["hidden_text"] = s.hidden_chars >= HIDDEN_MIN_CHARS
    out["image_heavy"] = s.images > 0 and s.visible_chars < IMAGE_HEAVY_TEXT_CHARS
    out["meta_refresh"] = s.meta_refresh
    out["script_or_data_link"] = s.script_link
    for href, text in s.anchors:
        if _URLISH.match(text) and href.lower().startswith("http"):
            shown, target = _host(text), _host(href)
            if shown and target and registered_domain(shown) != registered_domain(target):
                out["link_text_domain_mismatch"] = True
        if _brand_text_foreign_link(text, href):
            out["brand_text_foreign_link"] = True
    return out


def sender_domain(msg):
    """From adresinin kayıtlı domain'i (gönderen başına sınır için). Raporlara yazılmaz."""
    m = re.search(r"@([A-Za-z0-9.-]+)", str(msg.get("From", "") or ""))
    return registered_domain(m.group(1)) if m else ""

