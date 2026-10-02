# Mail -> model metni. Hem veri hazırlama (prepare_data.py) hem CLI (main.py) bunu
# kullanıyor: model hangi formatla eğitildiyse tahminde de aynı formatı görmeli.
# Format: "Subject: <konu>\n\n<gövde>". Diğer header'lar bilerek alınmıyor (tarih,
# sunucu adı vs. dataset kaynağını ele verir ve model onu ezberler).

import email
import email.header
import email.policy
import re
from html.parser import HTMLParser

# 17M karakterlik outlier'lar var - feature çıkarmayı yavaşlatıyor, anlamı da yok
MAX_CHARS = 20000

# Bunlardan biri varsa girdi düz metin değil, ham mail (.eml / mbox mesajı)
EMAIL_HEADERS = {"from", "to", "date", "received", "message-id", "mime-version",
                 "content-type", "return-path"}


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip = self.skip + 1
        if tag == "a":
            # link hedefini metne ekle, yoksa URL feature'ları HTML maillerde kayboluyor
            for name, value in attrs:
                if name == "href" and value and value.startswith("http"):
                    self.parts.append(" " + value + " ")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip > 0:
            self.skip = self.skip - 1
        if tag in ("p", "br", "div", "tr", "li"):
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip == 0:
            self.parts.append(data)


def html_to_text(html):
    p = _HTMLText()
    try:
        p.feed(html)
        p.close()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)
    return "".join(p.parts)


def clean_whitespace(text):
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _part_text(part):
    try:
        return part.get_content()
    except Exception:
        payload = part.get_payload(decode=True) or b""
        return payload.decode("utf-8", errors="replace")


def decode_subject(subject):
    """Bozuk header'larda policy decode etmiyor ve "=?UTF-8?B?...?=" olduğu gibi kalıyor."""
    if "=?" in subject:
        try:
            parts = []
            for value, charset in email.header.decode_header(subject):
                if isinstance(value, bytes):
                    value = value.decode(charset or "utf-8", errors="replace")
                parts.append(value)
            subject = "".join(parts)
        except Exception:
            pass
    return " ".join(subject.split())


def message_to_text(msg):
    """Subject + gövde. Diğer header'ları bilerek almıyoruz (tarih, sunucu adı vs.
    dataset kaynağını ele verir ve model onu ezberler)."""
    plain = []
    html = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        if part.get_content_disposition() == "attachment":
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain":
            plain.append(_part_text(part))
        elif ctype == "text/html":
            html.append(_part_text(part))

    if plain and any(p.strip() for p in plain):
        body = "\n".join(plain)
    else:
        body = "\n".join(html_to_text(h) for h in html)

    subject = decode_subject(str(msg.get("Subject", "") or ""))
    text = "Subject: " + subject + "\n\n" + clean_whitespace(body)
    return text[:MAX_CHARS]


def parse_bytes(raw):
    return email.message_from_bytes(raw, policy=email.policy.default)


def looks_like_email(raw):
    """Ham mail mi (header'lı .eml), yoksa elle yapıştırılmış düz metin mi?"""
    try:
        keys = {k.lower() for k in parse_bytes(raw).keys()}
    except Exception:
        return False
    return bool(keys & EMAIL_HEADERS)


def to_model_text(raw):
    """CLI girdisi (bytes) -> eğitimdekiyle aynı formatta metin.
    Ham mail MIME olarak çözülür (base64 gövde, HTML, encoded subject); düz metin
    sadece boşlukları temizlenip kısaltılır."""
    if looks_like_email(raw):
        return message_to_text(parse_bytes(raw))
    return clean_whitespace(raw.decode("utf-8", errors="replace"))[:MAX_CHARS]
