#Unit tests for phishing email analyzer.

import unittest

from analyzer import PhishingEmailAnalyzer


class TestPhishingEmailAnalyzer(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = PhishingEmailAnalyzer(resolve_dns=False)

    def test_clean_email_low_risk(self) -> None:
        text = "Hello, your meeting is scheduled for Tuesday at 3pm."
        result = self.analyzer.analyze(text)
        self.assertEqual(result.risk_level, "LOW")
        self.assertEqual(result.keyword_hits, {})
        self.assertEqual(result.urls, [])

    def test_keywords_detected(self) -> None:
        text = "URGENT: verify your password and login to your account now."
        result = self.analyzer.analyze(text)
        self.assertIn("urgent", result.keyword_hits)
        self.assertIn("verify", result.keyword_hits)
        self.assertGreater(result.keyword_score, 0)

    def test_http_scores_higher_than_https(self) -> None:
        http_result = self.analyzer.analyze("Visit http://example.com/path")
        https_result = self.analyzer.analyze("Visit https://example.com/path")
        self.assertGreater(http_result.url_score, https_result.url_score)
        self.assertEqual(http_result.urls[0].scheme_score, 15)
        self.assertEqual(https_result.urls[0].scheme_score, 5)

    def test_ip_in_url(self) -> None:
        text = "Go to http://10.0.0.1/login"
        result = self.analyzer.analyze(text)
        self.assertIn("10.0.0.1", result.ip_addresses)
        self.assertTrue(any("IP address" in i for u in result.urls for i in u.issues))

    def test_typo_keyword_immedietly(self) -> None:
        text = "Act immedietly to avoid suspension."
        result = self.analyzer.analyze(text)
        self.assertIn("immedietly", result.keyword_hits)

    def test_invalid_port_does_not_crash(self) -> None:
        # Kaggle dataset'inde gerçekten var: "http://host:27000To..."
        result = self.analyzer.analyze("see http://example.com:27000To/x")
        self.assertTrue(any("Invalid port" in i for u in result.urls for i in u.issues))


if __name__ == "__main__":
    unittest.main()
