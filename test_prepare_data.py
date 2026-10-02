# Mail parse etme (email_parsing.py) ve veri hazırlama (prepare_data.py) testleri (stdlib only).

import base64
import unittest

from email_parsing import (
    decode_subject,
    html_to_text,
    looks_like_email,
    message_to_text,
    parse_bytes,
    to_model_text,
)
from prepare_data import (
    RECIPIENT_REGEX,
    _source_from_filename,
    dedup,
    looks_english,
    strip_list_tags,
)

MULTIPART = b"""From: "PayPal" <service@paypa1.tk>
To: jose@monkey.org
Subject: =?utf-8?q?Hesab=C4=B1n=C4=B1z_ask=C4=B1ya_al=C4=B1nd=C4=B1?=
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="XX"

--XX
Content-Type: text/html; charset=utf-8

<html><style>.a{color:red}</style><body><p>Dear jose,</p>
<a href="http://paypa1.tk/login">Click here</a><script>evil()</script></body></html>
--XX--
"""


class TestPrepareData(unittest.TestCase):
    def test_html_only_message(self):
        text = message_to_text(parse_bytes(MULTIPART))
        self.assertTrue(text.startswith("Subject: Hesabınız askıya alındı"))
        # link hedefi metne eklenmeli, script/style içeriği eklenmemeli
        self.assertIn("http://paypa1.tk/login", text)
        self.assertNotIn("evil()", text)
        self.assertNotIn("color:red", text)

    def test_headers_other_than_subject_dropped(self):
        text = message_to_text(parse_bytes(MULTIPART))
        self.assertNotIn("service@paypa1.tk", text)
        self.assertNotIn("MIME-Version", text)

    def test_recipient_removed(self):
        out = RECIPIENT_REGEX.sub("", "Dear jose, mail jose@monkey.org or monkey.org. Salary Upgrade for Monkey Staff")
        self.assertNotIn("jose", out)
        self.assertNotIn("monkey", out.lower())
        # kelime içindeki "monkey"e dokunulmaz
        self.assertIn("monkeys", RECIPIENT_REGEX.sub("", "the monkeys"))

    def test_encoded_subject_left_by_policy_is_decoded(self):
        # Nazario 2025'te gerçekten var: decode edilmemiş base64 subject
        raw = "=?UTF-8?B?UGF5UGFsOiBXaXIgYmVuw7Z0aWdlbg==?="
        self.assertEqual(decode_subject(raw), "PayPal: Wir benötigen")
        self.assertEqual(decode_subject("  plain   subject "), "plain subject")

    def test_list_tag_removed_from_subject_only(self):
        text = "Subject: [Python-announce] Wing 10 released\n\nsee [Python-announce] archive"
        out = strip_list_tags(text)
        self.assertTrue(out.startswith("Subject: Wing 10 released"))
        # gövdedeki aynı ifadeye dokunulmaz
        self.assertIn("[Python-announce] archive", out)

    def test_looks_english(self):
        self.assertTrue(looks_english("Please verify your account and confirm the payment for this order."))
        self.assertFalse(looks_english("Réduisez vos mensualités dès aujourd'hui avec nous pour votre crédit."))
        self.assertFalse(looks_english("Verfolgen Sie die Schlüssel und Taschen mit der neuen App für Sie."))
        self.assertFalse(looks_english("XFAst304fDEBsb0JiCuPXrDLF8K62VbLbcg2BhpZfZ"))

    def test_source_from_filename(self):
        self.assertEqual(_source_from_filename("fedora-users-2024-03.mbox"), "fedora-users")
        self.assertEqual(_source_from_filename("ubuntu-security-announce-2025-April.mbox"),
                         "ubuntu-security-announce")

    def test_html_to_text_breaks_lines(self):
        self.assertIn("\n", html_to_text("<p>a</p><p>b</p>"))

    def test_dedup_ignores_case_and_whitespace(self):
        rows = [
            {"text": "Hello  World", "label": 0, "source": "x"},
            {"text": "hello world", "label": 0, "source": "x"},
            {"text": "other", "label": 1, "source": "x"},
        ]
        self.assertEqual(len(dedup(rows)), 2)


class TestModelText(unittest.TestCase):
    """CLI girdisi eğitimdeki formatla aynı metne dönüşmeli (train/serve skew olmasın)."""

    def test_raw_email_is_parsed_like_training_data(self):
        self.assertTrue(looks_like_email(MULTIPART))
        self.assertEqual(to_model_text(MULTIPART), message_to_text(parse_bytes(MULTIPART)))

    def test_base64_body_is_decoded(self):
        body = base64.b64encode("Your invoice is attached, please pay today.".encode()).decode()
        raw = ("From: billing@example.com\nSubject: Invoice\nMIME-Version: 1.0\n"
               "Content-Type: text/plain; charset=utf-8\nContent-Transfer-Encoding: base64\n\n"
               + body + "\n").encode()
        text = to_model_text(raw)
        self.assertIn("Your invoice is attached", text)
        self.assertNotIn(body, text)
        self.assertNotIn("Content-Transfer-Encoding", text)

    def test_plain_text_is_not_treated_as_email(self):
        plain = b"Subject: URGENT\n\nVerify your account:   http://x.tk/login\n\n\n\nThanks"
        self.assertFalse(looks_like_email(plain))
        self.assertEqual(to_model_text(plain),
                         "Subject: URGENT\n\nVerify your account: http://x.tk/login\n\nThanks")
        self.assertFalse(looks_like_email(b"Hello, your meeting is on Tuesday."))


if __name__ == "__main__":
    unittest.main()
