import re
import socket

# kelimeler küçük harf - metni analyze ederken lower() yapıyoruz
KEYWORDS = [
    "urgent",
    "verify",
    "password",
    "login",
    "account",
    "suspended",
    "click",
    "immediately",
    "immedietly",  # typo ama phishing maillerde çok çıkıyor
    "winner",
    "lottery",
    "bitcoin",
    "invoice",
    "wire transfer",
    "gift card",
    "act now",
    "limited time",
    "confirm",
    "security alert",
]

# skorlar 
KEYWORD_SCORE = 8
MAX_KEYWORD_SCORE = 45

HTTP_SCORE = 15
HTTPS_SCORE = 5

IP_IN_TEXT_SCORE = 20
IP_IN_LINK_SCORE = 25

# şüpheli uzantılar (free TLD listesi internetten bakıldı)
BAD_TLDS = [
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work",
    "click", "link", "buzz", "cam", "rest",
]

SHORT_LINK_SITES = [
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "rebrand.ly", "shorturl.at",
]


FREE_EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "proton.me", "icloud.com", "aol.com",
]

URL_REGEX = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

IP_REGEX = re.compile(
    r"(?<![\d.])"
    r"(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d?\d)"
    r"(?![\d.])"
)


class URLCheckResult:
    def __init__(self, url, scheme, host, scheme_score, issues=None, issue_score=0):
        self.url = url
        self.scheme = scheme
        self.host = host
        self.scheme_score = scheme_score
        self.issues = issues if issues else []
        self.issue_score = issue_score

    @property
    def total_score(self):
        return self.scheme_score + self.issue_score


class AnalysisResult:
    def __init__(
        self,
        score,
        risk_level,
        keyword_hits,
        keyword_score,
        urls,
        url_score,
        ip_addresses,
        ip_score,
        domain_warnings,
        details,
    ):
        self.score = score
        self.risk_level = risk_level
        self.keyword_hits = keyword_hits
        self.keyword_score = keyword_score
        self.urls = urls
        self.url_score = url_score
        self.ip_addresses = ip_addresses
        self.ip_score = ip_score
        self.domain_warnings = domain_warnings
        self.details = details

    def summary(self):
        lines = []
        lines.append("Risk: " + self.risk_level + " (score: " + str(self.score) + ")")
        lines.append("")
        lines.append("Keywords:")
        if len(self.keyword_hits) == 0:
            lines.append("  (none)")
        else:
            for word in sorted(self.keyword_hits.keys()):
                lines.append("  - " + word + ": " + str(self.keyword_hits[word]) + "x")

        lines.append("")
        lines.append("URLs:")
        if len(self.urls) == 0:
            lines.append("  (none)")
        else:
            for u in self.urls:
                lines.append("  - " + u.url)
                lines.append("      scheme score: " + str(u.scheme_score) + " (" + u.scheme + ")")
                for issue in u.issues:
                    lines.append("      ! " + issue)

        lines.append("")
        lines.append("IP addresses:")
        if len(self.ip_addresses) == 0:
            lines.append("  (none)")
        else:
            for ip in self.ip_addresses:
                lines.append("  - " + ip)

        if len(self.domain_warnings) > 0:
            lines.append("")
            lines.append("Domain warnings:")
            for w in self.domain_warnings:
                lines.append("  - " + w)

        if len(self.details) > 0:
            lines.append("")
            lines.append("Notes:")
            for d in self.details:
                lines.append("  - " + d)

        return "\n".join(lines)


def is_valid_ip(ip_str):
    parts = ip_str.split(".")
    if len(parts) != 4:
        return False
    try:
        for p in parts:
            n = int(p)
            if n < 0 or n > 255:
                return False
        return True
    except ValueError:
        return False


def get_risk_level(score):
    if score >= 80:
        return "CRITICAL"
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MEDIUM"
    return "LOW"


def host_is_shortener(host):
    host = host.lower()
    for site in SHORT_LINK_SITES:
        if host == site or host.endswith("." + site):
            return True
    return False


class PhishingEmailAnalyzer:
    def __init__(self, keywords=None, resolve_dns=True):
        if keywords is None:
            self.keywords = KEYWORDS
        else:
            self.keywords = []
            for k in keywords:
                self.keywords.append(k.lower())
        self.resolve_dns = resolve_dns

    def analyze(self, text):
        text_lower = text.lower()
        details = []

        keyword_hits, keyword_score = self._check_keywords(text_lower)
        urls, url_score, domain_warnings = self._check_urls(text)
        ip_list, ip_score = self._find_ips(text, urls)

        extra_score = 0

        # Subject satırı varsa ayrı bak (sample dosyalarında var)
        subject = self._extract_subject(text)
        if subject:
            sub_score, sub_notes = self._check_subject_line(subject)
            extra_score = extra_score + sub_score
            for note in sub_notes:
                details.append(note)

        caps_score, caps_note = self._check_shouting(text)
        extra_score = extra_score + caps_score
        if caps_note:
            details.append(caps_note)

        punct_score, punct_msg = self._check_weird_punctuation(text)
        extra_score = extra_score + punct_score
        if punct_msg:
            details.append(punct_msg)

        attach_score, attach_msg = self._check_attachments(text_lower)
        extra_score = extra_score + attach_score
        if attach_msg:
            details.append(attach_msg)

        email_score, email_notes = self._check_email_addresses(text_lower)
        extra_score = extra_score + email_score
        for n in email_notes:
            details.append(n)

        b64_score, b64_msg = self._check_base64_blob(text)
        extra_score = extra_score + b64_score
        if b64_msg:
            details.append(b64_msg)

        if len(urls) >= 4:
            extra_score = extra_score + 12
            details.append("Many links in one message (" + str(len(urls)) + ")")

        total = keyword_score + url_score + ip_score + extra_score
        risk = get_risk_level(total)

        return AnalysisResult(
            score=total,
            risk_level=risk,
            keyword_hits=keyword_hits,
            keyword_score=keyword_score,
            urls=urls,
            url_score=url_score,
            ip_addresses=ip_list,
            ip_score=ip_score,
            domain_warnings=domain_warnings,
            details=details,
        )

    def _check_keywords(self, text_lower):
        hits = {}
        total_count = 0
        for word in self.keywords:
            # bazı kelimeler iki kelime (wire transfer) - word boundary yetmez
            if " " in word:
                count = text_lower.count(word)
            else:
                found = re.findall(r"\b" + re.escape(word) + r"\b", text_lower)
                count = len(found)
            if count > 0:
                hits[word] = count
                total_count = total_count + count

        raw_score = total_count * KEYWORD_SCORE
        if raw_score > MAX_KEYWORD_SCORE:
            raw_score = MAX_KEYWORD_SCORE
        return hits, raw_score

    def _check_urls(self, text):
        raw_urls = URL_REGEX.findall(text)
        seen = []
        results = []
        warnings = []
        total_score = 0

        for raw in raw_urls:
            url = raw.rstrip(".,;:!?)")
            if url in seen:
                continue
            seen.append(url)

            check = self._inspect_one_url(url)
            results.append(check)
            total_score = total_score + check.total_score

            domain_msgs = self._inspect_domain(check.host, url)
            for w in domain_msgs:
                full = check.host + ": " + w
                if full not in warnings:
                    warnings.append(full)
                total_score = total_score + 5

        return results, total_score, warnings

    def _inspect_one_url(self, url):
        issues = []
        issue_score = 0

        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
        except Exception:
            return URLCheckResult(url, "", "", 10, ["Malformed URL"], 10)

        scheme = parsed.scheme.lower() if parsed.scheme else ""
        host = parsed.hostname.lower() if parsed.hostname else ""

        if scheme == "https":
            scheme_score = HTTPS_SCORE
        elif scheme == "http":
            scheme_score = HTTP_SCORE
            issues.append("Uses HTTP (unencrypted)")
        else:
            scheme_score = 12
            issues.append("Unusual scheme: " + (scheme if scheme else "(missing)"))
            issue_score = issue_score + 7

        if host == "":
            issues.append("Missing hostname")
            issue_score = issue_score + 10
            return URLCheckResult(url, scheme, host, scheme_score, issues, issue_score)

        if is_valid_ip(host):
            issues.append("URL points to IP address instead of domain")
            issue_score = issue_score + IP_IN_LINK_SCORE

        if "@" in url:
            issues.append("URL contains @ (possible credential hiding)")
            issue_score = issue_score + 15

        if host_is_shortener(host):
            issues.append("URL shortener detected")
            issue_score = issue_score + 10

        if len(host) > 60:
            issues.append("Unusually long hostname")
            issue_score = issue_score + 8

        if re.search(r"\d{4,}", host):
            issues.append("Hostname contains long digit sequence")
            issue_score = issue_score + 6

        # bozuk port (ör. "host:27000To") urllib'de ValueError atıyor
        try:
            port = parsed.port
        except ValueError:
            port = None
            issues.append("Invalid port in URL")
            issue_score = issue_score + 8
        if port and port not in (80, 443):
            issues.append("Non-standard port: " + str(port))
            issue_score = issue_score + 8

        path = parsed.path if parsed.path else ""
        if len(path) > 80:
            issues.append("Very long URL path")
            issue_score = issue_score + 5

        # http ve https dışı garip şemalar
        if scheme not in ("http", "https", ""):
            issue_score = issue_score + 5

        return URLCheckResult(url, scheme, host, scheme_score, issues, issue_score)

    def _inspect_domain(self, host, url):
        if not host:
            return []
        warnings = []

        if is_valid_ip(host):
            return warnings

        parts = host.split(".")
        if len(parts) >= 2:
            tld = parts[-1]
            if tld in BAD_TLDS:
                warnings.append("suspicious TLD (." + tld + ")")

        if host.count(".") >= 4:
            warnings.append("many subdomains (possible obfuscation)")

        if host.count("-") >= 3:
            warnings.append("many hyphens in domain")

        if re.search(r"(paypa[l1]|amaz[o0]n|micr[o0]s[o0]ft|g[o0]{2}gle|appple|faceb[o0]ok)", host):
            warnings.append("possible brand typosquatting")

        # bank/paypal kelimesi + şüpheli tld kombinasyonu
        if ("bank" in host or "paypal" in host or "secure" in host) and len(parts) >= 2:
            if parts[-1] in BAD_TLDS:
                warnings.append("financial keyword on cheap TLD")

        if self.resolve_dns:
            dns_msg = self._try_dns(host)
            if dns_msg:
                warnings.append(dns_msg)

        return warnings

    def _try_dns(self, host):
        try:
            socket.getaddrinfo(host, None)
            return None
        except socket.gaierror:
            return "domain does not resolve (NXDOMAIN / DNS failure)"
        except Exception:
            return "DNS check failed"

    def _find_ips(self, text, url_results):
        ips = []
        for m in IP_REGEX.findall(text):
            if is_valid_ip(m) and m not in ips:
                ips.append(m)

        score = 0
        if len(ips) > 0:
            score = IP_IN_TEXT_SCORE

        for u in url_results:
            if u.host and is_valid_ip(u.host):
                if u.host not in ips:
                    ips.append(u.host)

        return ips, score

    def _extract_subject(self, text):
        for line in text.splitlines():
            if line.lower().startswith("subject:"):
                return line.split(":", 1)[1].strip()
        return None

    def _check_subject_line(self, subject):
        score = 0
        notes = []
        letters = [c for c in subject if c.isalpha()]
        if len(letters) > 5:
            upper = [c for c in letters if c.isupper()]
            ratio = len(upper) / len(letters)
            if ratio > 0.7:
                score = score + 15
                notes.append("Subject is mostly ALL CAPS")

        if "!!!" in subject or "???" in subject:
            score = score + 10
            notes.append("Subject has excessive punctuation")

        sub_lower = subject.lower()
        for kw in ["urgent", "verify", "suspended", "action required", "final notice"]:
            if kw in sub_lower:
                score = score + 8
                notes.append('Subject contains pressure phrase: "' + kw + '"')
                break

        return score, notes

    def _check_shouting(self, text):
        words = re.findall(r"[A-Za-z]{4,}", text)
        if len(words) < 3:
            return 0, None
        shout = 0
        for w in words:
            if w.isupper():
                shout = shout + 1
        if shout / len(words) > 0.35:
            return 12, "Lots of ALL CAPS words in body"
        return 0, None

    def _check_weird_punctuation(self, text):
        if text.count("!!!") >= 2 or text.count("???") >= 2:
            return 8, "Repeated !!! or ??? in message"
        if text.count("$") >= 5:
            return 6, "Many dollar signs (common in scam templates)"
        return 0, None

    def _check_attachments(self, text_lower):
        bad_ext = [".exe", ".scr", ".bat", ".cmd", ".js", ".vbs", ".iso", ".lnk"]
        score = 0
        msgs = []
        for ext in bad_ext:
            if ext in text_lower:
                score = score + 18
                msgs.append("Mentions risky attachment type: " + ext)
        triggers = [
            "open the attachment",
            "enable macros",
            "download the file",
            "see attached",
            "attached invoice",
        ]
        for t in triggers:
            if t in text_lower:
                score = score + 10
                msgs.append('Attachment bait phrase: "' + t + '"')
        if len(msgs) == 0:
            return 0, None
        return min(score, 35), "; ".join(msgs)

    def _check_email_addresses(self, text_lower):
        score = 0
        notes = []
        emails = EMAIL_REGEX.findall(text_lower)
        if len(emails) == 0:
            return 0, notes

        for em in emails:
            domain = em.split("@")[1]
            if domain in FREE_EMAIL_DOMAINS:
                if any(x in text_lower for x in ["bank", "paypal", "microsoft", "apple support", "irs", "tax"]):
                    score = score + 12
                    notes.append("Free email domain but message mentions institution: " + em)

        # display name trick: "PayPal" <noreply@gmail.com> basit regex
        if re.search(r"paypal|microsoft|amazon|apple", text_lower) and "@" in text_lower:
            for em in emails:
                dom = em.split("@")[1]
                if dom in FREE_EMAIL_DOMAINS and score == 0:
                    score = score + 8
                    notes.append("Brand name in text but sender-style email is free provider")

        return min(score, 25), notes

    def _check_base64_blob(self, text):
        # çok uzun base64 blok = gizlenmiş payload olabilir
        blobs = re.findall(r"[A-Za-z0-9+/]{80,}={0,2}", text)
        if len(blobs) >= 1:
            return 10, "Long base64-like block found (could hide content)"
        return 0, None
