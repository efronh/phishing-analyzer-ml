# CLI (main.py) uçtan uca testleri: gerçek bir süreç olarak çalıştırılıyor.

import base64
import os
import subprocess
import sys
import tempfile
import unittest

try:
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "samples", "suspicious_sample.txt")


def run_cli(args, stdin=b""):
    proc = subprocess.run([sys.executable, os.path.join(HERE, "main.py"), "--no-dns"] + args,
                          input=stdin, capture_output=True, cwd=HERE)
    return proc.returncode, proc.stdout.decode("utf-8", errors="replace"), proc.stderr.decode()


def raw_email(body, subject="Account notice"):
    encoded = base64.b64encode(body.encode()).decode()
    return ("From: support@example.com\nTo: someone@example.org\nSubject: " + subject + "\n"
            "MIME-Version: 1.0\nContent-Type: text/plain; charset=utf-8\n"
            "Content-Transfer-Encoding: base64\n\n" + encoded + "\n").encode()


class TestCLIRules(unittest.TestCase):
    def test_rules_only_flags_sample(self):
        code, out, _ = run_cli([SAMPLE])
        self.assertIn("Risk: CRITICAL", out)
        self.assertEqual(code, 1)

    def test_empty_input(self):
        code, _, err = run_cli([], stdin=b"   \n")
        self.assertEqual(code, 1)
        self.assertIn("empty input", err)

    def test_raw_email_body_is_decoded_before_analysis(self):
        # gövde base64 - çözülmeden analiz edilseydi hiçbir keyword bulunamazdı
        code, out, _ = run_cli([], stdin=raw_email("URGENT: verify your password immediately."))
        self.assertIn("Input: raw email", out)
        self.assertIn("verify: 1x", out)
        self.assertIn("password: 1x", out)


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
class TestCLIModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ml_model import PhishingClassifier
        from test_ml import LEGIT, PHISH

        model = PhishingClassifier().fit(PHISH + LEGIT, [1] * len(PHISH) + [0] * len(LEGIT))
        model.threshold = 0.5
        cls.path = os.path.join(tempfile.mkdtemp(), "m.joblib")
        model.save(cls.path)

    def test_exit_code_follows_ml_verdict(self):
        # eski hata: ekranda "PHISHING" yazıp hybrid skor yüzünden 0 (güvenli) dönebiliyordu
        inputs = [
            b"URGENT: verify your password immediately at http://192.168.1.10/login",
            b"Hi, the meeting is moved to Thursday at 2pm, see you there",
            raw_email("Your account is suspended. Click http://paypa1-secure.tk/verify to confirm"),
        ]
        for data in inputs:
            code, out, _ = run_cli(["--model", self.path], stdin=data)
            self.assertNotIn("Hybrid", out)
            self.assertEqual(code, 1 if "-> PHISHING" in out else 0, out)
            self.assertTrue("-> PHISHING" in out or "-> legit" in out)


if __name__ == "__main__":
    unittest.main()
