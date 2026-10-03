# ML katmanı testleri. sklearn yoksa atlanır.

import os
import tempfile
import unittest

try:
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from analyzer import PhishingEmailAnalyzer
from features import FEATURE_NAMES, extract_features, normalize_text

PHISH = [
    "URGENT: verify your password immediately at http://192.168.1.10/login",
    "Your account is suspended. Click http://paypa1-secure.tk/verify to confirm",
    "You are the lottery winner! Send gift card codes now, act now",
    "Security alert: login at http://bit.ly/abc123 to confirm your account",
    "Wire transfer needed urgently, confirm your bank account immediately",
    "Your invoice is attached, enable macros and open the attachment invoice.exe",
]
LEGIT = [
    "Hi, the meeting is moved to Thursday at 2pm, see you there",
    "Thanks for the notes from the lecture, they were very helpful",
    "Are you free for lunch tomorrow? There is a new place near the office",
    "The quarterly report draft is in the shared folder for review",
    "Happy birthday! Hope you have a wonderful day with your family",
    "Reminder: the library is closed on Friday, hours resume Saturday",
]


class TestFeatures(unittest.TestCase):
    def test_feature_vector_length(self):
        vec = extract_features("Hello there, see you at the meeting.")
        self.assertEqual(len(vec), len(FEATURE_NAMES))

    def test_phishing_has_higher_rule_features(self):
        clean = extract_features(LEGIT[0])
        phish = extract_features(PHISH[0])
        i = FEATURE_NAMES.index("rule_score_log")
        self.assertGreater(phish[i], clean[i])
        self.assertEqual(phish[FEATURE_NAMES.index("http_url_fraction")], 1.0)
        self.assertEqual(phish[FEATURE_NAMES.index("ip_url")], 1.0)

    def test_many_clean_links_stay_bounded(self):
        # v1 hatası: 40 temiz https linkli duyuru, link sayısı yüzünden phishing çıkıyordu
        one = extract_features("Release notes: https://example.org/a")
        many = extract_features("Release notes: " + " ".join(
            "https://example.org/p" + str(i) for i in range(40)))
        for name in ["suspicious_url_fraction", "http_url_fraction", "max_url_issue_score",
                     "domain_warning_fraction"]:
            i = FEATURE_NAMES.index(name)
            self.assertEqual(one[i], many[i], name)
        self.assertLess(many[FEATURE_NAMES.index("url_count_log")], 4.0)

    def test_fraction_features_in_unit_range(self):
        vec = extract_features(PHISH[1] + " https://ok.example.com/x")
        for name in ["http_url_fraction", "suspicious_url_fraction", "domain_warning_fraction"]:
            v = vec[FEATURE_NAMES.index(name)]
            self.assertTrue(0.0 <= v <= 1.0, name)

    def test_lure_features(self):
        from features import FEATURE_NAMES_V3
        lure = "Subject: RFQ\n\nDear Supplier, kindly see attached RFQ and send the invoice urgently."
        clean = "Subject: meeting\n\n" + "We moved the weekly meeting to Thursday afternoon. " * 10
        a = dict(zip(FEATURE_NAMES_V3, extract_features(lure, version=3)))
        b = dict(zip(FEATURE_NAMES_V3, extract_features(clean, version=3)))
        for name in ["lure_payment_pretext", "lure_attachment_pretext", "lure_generic_greeting",
                     "lure_urgency", "short_body"]:
            self.assertGreater(a[name], 0, name)
            self.assertEqual(b[name], 0, name)
        # v3 = v2 + lure feature'ları, v2 kısmı aynı kalmalı
        self.assertEqual(extract_features(lure, version=3)[:len(FEATURE_NAMES)], extract_features(lure))

    def test_preprocess_drops_subject_label_keeps_words(self):
        from ml_model import preprocess
        out = preprocess("Subject: URGENT invoice\n\nThis is subject to change.")
        self.assertNotIn("subject:", out)
        self.assertIn("urgent invoice", out)
        # gövdedeki normal "subject" kelimesine dokunulmaz
        self.assertIn("subject to change", out)

    def test_old_schema_model_is_rejected(self):
        import joblib
        from ml_model import PhishingClassifier
        path = os.path.join(tempfile.mkdtemp(), "old.joblib")
        joblib.dump({"pipeline": None, "threshold": 0.5, "meta": {}}, path)
        with self.assertRaises(ValueError):
            PhishingClassifier.load(path)

    def test_normalize_replaces_urls(self):
        out = normalize_text("go to https://example.com/x now")
        self.assertIn("URLTOKEN", out)
        self.assertNotIn("example.com", out)


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
class TestClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ml_model import PhishingClassifier
        cls.model = PhishingClassifier().fit(PHISH + LEGIT, [1] * len(PHISH) + [0] * len(LEGIT))

    def test_separates_training_classes(self):
        probs = self.model.predict_proba([PHISH[1], LEGIT[1]])
        self.assertGreater(probs[0], probs[1])

    def test_explain_returns_positive_signals(self):
        signals = self.model.explain(PHISH[0])
        self.assertGreater(len(signals), 0)
        for _, weight in signals:
            self.assertGreater(weight, 0)

    def test_explain_skips_absent_rule_features(self):
        # linksiz mailde URL feature'ları açıklamada çıkmamalı
        for name, _ in self.model.explain(LEGIT[0], top_k=50):
            if name.startswith("rules__"):
                self.assertNotIn("url", name)

    def test_analyze_verdict_matches_threshold(self):
        for text in (PHISH[0], LEGIT[0]):
            res = self.model.analyze(text)
            self.assertTrue(0.0 <= res.probability <= 1.0)
            self.assertEqual(res.is_phishing, res.probability >= self.model.threshold)
            self.assertIn("PHISHING" if res.is_phishing else "legit", res.summary())

    def test_save_and_load_keeps_threshold(self):
        from ml_model import PhishingClassifier
        path = os.path.join(tempfile.mkdtemp(), "m.joblib")
        self.model.threshold = 0.37
        self.model.save(path)
        loaded = PhishingClassifier.load(path)
        self.assertAlmostEqual(loaded.threshold, 0.37)
        self.assertAlmostEqual(
            float(self.model.predict_proba([LEGIT[2]])[0]),
            float(loaded.predict_proba([LEGIT[2]])[0]),
        )

    def test_top_features(self):
        phish, legit = self.model.top_features(3)
        self.assertEqual(len(phish), 3)
        self.assertGreater(phish[0][1], legit[0][1])


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
class TestPipelineVariants(unittest.TestCase):
    def test_all_configs_fit(self):
        from ml_model import build_pipeline, pipeline_scores
        y = [1] * len(PHISH) + [0] * len(LEGIT)
        configs = [
            {"classifier": "logreg", "use_text": False},
            {"classifier": "svm"},
            {"classifier": "nb", "use_rules": False},
        ]
        for cfg in configs:
            p = build_pipeline(**cfg).fit(PHISH + LEGIT, y)
            self.assertEqual(len(pipeline_scores(p, LEGIT[:2])), 2)

    def test_scaled_rule_features_are_clipped(self):
        from ml_model import CLIP_RANGE, build_pipeline
        p = build_pipeline(classifier="logreg", use_text=False)
        p.fit(PHISH + LEGIT, [1] * len(PHISH) + [0] * len(LEGIT))
        extreme = "WIN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        x = p.named_steps["features"].transform([extreme])
        self.assertLessEqual(abs(x).max(), CLIP_RANGE)

    def test_fit_with_cv_threshold(self):
        from ml_model import PhishingClassifier, fit_with_cv_threshold
        texts = PHISH + LEGIT
        y = [1] * len(PHISH) + [0] * len(LEGIT)
        model = fit_with_cv_threshold(texts, y, folds=2, seed=0)
        self.assertIsInstance(model, PhishingClassifier)
        self.assertTrue(0.0 < model.threshold < 1.0)
        self.assertIn("plateau", model.meta["threshold_method"])
        lo, hi = model.meta["threshold_cv"]["plateau"]
        self.assertTrue(lo <= model.threshold <= hi)
        # son model verinin tamamıyla eğitilmiş olmalı ve explain çalışmalı
        self.assertGreater(len(model.explain(PHISH[0])), 0)

    def test_platt_calibrator_is_monotonic(self):
        import numpy as np
        from ml_model import PlattCalibrator
        rng = np.random.RandomState(0)
        y = rng.randint(0, 2, 400)
        raw = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.15, 400), 0.01, 0.99)
        cal = PlattCalibrator().fit(raw, y)
        grid = np.linspace(0.01, 0.99, 50)
        out = cal.transform(grid)
        self.assertTrue(np.all(np.diff(out) > 0))
        self.assertTrue(np.all((out > 0) & (out < 1)))

    def test_calibrated_model_roundtrip_and_wording(self):
        from ml_model import PhishingClassifier, fit_with_cv_threshold
        texts = PHISH + LEGIT
        y = [1] * len(PHISH) + [0] * len(LEGIT)
        model = fit_with_cv_threshold(texts, y, folds=2, seed=0)
        self.assertIsNotNone(model.calibrator)
        self.assertIn("platt", model.meta["calibration_oof"])
        self.assertIn("calibrated", model.analyze(PHISH[0]).summary())
        path = os.path.join(tempfile.mkdtemp(), "cal.joblib")
        model.save(path)
        loaded = PhishingClassifier.load(path)
        self.assertAlmostEqual(float(model.predict_proba([LEGIT[1]])[0]),
                               float(loaded.predict_proba([LEGIT[1]])[0]))
        raw = fit_with_cv_threshold(texts, y, folds=2, seed=0, calibrate=False)
        self.assertIsNone(raw.calibrator)
        self.assertIn("not calibrated", raw.analyze(PHISH[0]).summary())
        self.assertNotIn("probability", raw.analyze(PHISH[0]).summary())

    def test_invalid_configs(self):
        from ml_model import build_pipeline
        with self.assertRaises(ValueError):
            build_pipeline(use_text=False, use_rules=False)
        with self.assertRaises(ValueError):
            build_pipeline(classifier="nb", use_rules=True)


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
class TestEvaluation(unittest.TestCase):
    def test_best_threshold_separates(self):
        from evaluation import best_f1_threshold, compute_metrics
        y = [0, 0, 0, 1, 1, 1]
        scores = [0.1, 0.2, 0.3, 0.6, 0.7, 0.9]
        th = best_f1_threshold(y, scores)
        self.assertTrue(0.3 < th <= 0.6)
        m = compute_metrics(y, scores, th)
        self.assertEqual(m["f1"], 1.0)
        self.assertEqual(m["fpr"], 0.0)

    def test_threshold_for_fpr(self):
        from evaluation import compute_metrics, threshold_for_fpr
        y = [0] * 10 + [1] * 5
        scores = [0.05 * i for i in range(10)] + [0.3, 0.6, 0.7, 0.8, 0.9]
        th = threshold_for_fpr(y, scores, 0.1)
        m = compute_metrics(y, scores, th)
        self.assertLessEqual(m["fpr"], 0.1)
        # 0.1 izin -> en yüksek skorlu tek meşru mail (0.45) geçebilir, 0.40 geçemez
        self.assertTrue(0.40 < th <= 0.45)
        self.assertEqual(threshold_for_fpr(y, scores, 0.0) > 0.45, True)

    def test_f1_curve_matches_sklearn(self):
        import numpy as np
        from sklearn.metrics import f1_score
        from ml_model import THRESHOLD_GRID, f1_curve
        rng = np.random.RandomState(0)
        y = rng.randint(0, 2, 200)
        s = np.clip(y * 0.3 + rng.rand(200) * 0.7, 0, 1)
        curve = f1_curve(y, s)
        for t in [0.2, 0.5, 0.8]:
            i = int(np.argmin(np.abs(THRESHOLD_GRID - t)))
            self.assertAlmostEqual(curve[i], f1_score(y, s >= THRESHOLD_GRID[i]))

    def test_select_threshold_plateau_center(self):
        import numpy as np
        from ml_model import select_threshold
        grid = np.round(np.linspace(0.0, 1.0, 11), 2)
        # zirve 0.4'te, 0.3-0.6 arası zirveye çok yakın (düz tepe), 0.9'daki ikinci tepe bitişik değil
        base = np.array([0.5, 0.6, 0.7, 0.895, 0.9, 0.896, 0.895, 0.6, 0.5, 0.899, 0.4])
        curves = np.array([base + d for d in (-0.01, 0.0, 0.01)])
        th_max, _ = select_threshold(curves, "max", grid)
        th_mid, info = select_threshold(curves, "plateau", grid)
        self.assertEqual(th_max, 0.4)
        self.assertEqual(info["plateau"], [0.3, 0.6])
        self.assertAlmostEqual(th_mid, 0.45)

    def test_describe_email_hides_personal_data(self):
        from evaluation import describe_email
        text = ("Subject: Re: question from celine.dupont at example.gouv.fr https://x.org/a\n\n"
                "Hi Daniel, see https://example.com/doc and http://bad.tk/x\nRegards, Olga Gorun\n"
                "olga@example.com")
        d = describe_email(text)
        for secret in ["celine", "example.gouv.fr", "x.org", "Daniel", "Olga", "olga@example.com"]:
            self.assertNotIn(secret, d)
        self.assertIn("<email>", d)
        self.assertIn("2 links", d)
        self.assertTrue(describe_email("no subject here").startswith("Subject: (none)"))

    def test_wilson_ci(self):
        from evaluation import wilson_ci
        lo, hi = wilson_ci(444, 454)
        self.assertTrue(0.959 < lo < 0.961 and 0.987 < hi < 0.989)
        self.assertEqual(wilson_ci(0, 0), [None, None])
        self.assertEqual(wilson_ci(0, 10)[0], 0.0)

    def test_precision_at_prevalence(self):
        from evaluation import precision_at_prevalence
        conf = {"tp": 99, "fn": 1, "fp": 1, "tn": 99}
        r = precision_at_prevalence(conf, 0.5)
        self.assertAlmostEqual(r["precision"], 0.99, places=3)
        # TPR 0.99, FPR 0.01, %1 phishing -> 0.0099 / (0.0099 + 0.0099) = 0.5
        r = precision_at_prevalence(conf, 0.01)
        self.assertAlmostEqual(r["precision"], 0.5, places=3)
        self.assertLess(r["precision_ci"][0], 0.5)
        self.assertGreater(r["precision_ci"][1], 0.5)
        # sıfır yanlış alarmda da aralık tanımlı ve 1'in altında
        r = precision_at_prevalence({"tp": 50, "fn": 0, "fp": 0, "tn": 50}, 0.01)
        self.assertLess(r["precision_ci"][0], 1.0)

    def test_pr_auc_at_prevalence(self):
        from sklearn.metrics import average_precision_score
        import numpy as np
        from evaluation import pr_auc_at_prevalence
        y = np.array([1, 1, 0, 1, 0, 0, 0, 0])
        s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        # test setinin kendi payında ağırlıksız PR-AUC ile aynı
        self.assertAlmostEqual(pr_auc_at_prevalence(y, s, y.mean()),
                               average_precision_score(y, s), places=4)
        self.assertLess(pr_auc_at_prevalence(y, s, 0.01), average_precision_score(y, s))

    def test_mcnemar_exact(self):
        from evaluation import mcnemar_exact
        y = [1] * 20
        a = [1] * 20
        b = [1] * 10 + [0] * 10  # B 10 maili kaçırıyor, A hepsini buluyor
        r = mcnemar_exact(y, a, b)
        self.assertEqual((r["only_a_correct"], r["only_b_correct"]), (10, 0))
        self.assertAlmostEqual(r["p_value"], 2 / 2 ** 10, places=4)
        self.assertEqual(mcnemar_exact(y, a, a)["p_value"], 1.0)

    def test_calibration_stats(self):
        import numpy as np
        from evaluation import calibration_stats
        rng = np.random.RandomState(0)
        p = rng.rand(5000)
        y = (rng.rand(5000) < p).astype(int)  # mükemmel kalibre
        self.assertLess(calibration_stats(y, p)["ece"], 0.03)
        self.assertGreater(calibration_stats(y, np.sqrt(p))["ece"], 0.1)  # aşırı emin değil, kaymış

    def test_near_duplicate_groups(self):
        from evaluation import near_duplicate_groups
        base = "Your mailbox storage is full, verify your account at the portal to keep receiving email "
        texts = [base + "ID 1001", base + "ID 2002", base + "ID 3003",
                 "Lunch tomorrow at noon near the office with the whole team",
                 "Quarterly report draft is ready for review in the shared folder",
                 "Lunch tomorrow at noon near the office with the whole team!"]
        g = near_duplicate_groups(texts)
        self.assertEqual(len({g[0], g[1], g[2]}), 1)  # aynı kampanya
        self.assertEqual(g[3], g[5])
        self.assertNotEqual(g[0], g[3])
        self.assertNotEqual(g[3], g[4])

    def test_grouped_split_keeps_groups_together(self):
        import numpy as np
        from train import split_indices
        rng = np.random.RandomState(0)
        y = rng.randint(0, 2, 3000)
        groups = rng.randint(0, 1500, 3000)
        tr, va, te = split_indices(y, 42, groups)
        self.assertEqual(len(tr) + len(va) + len(te), 3000)
        for a, b in [(tr, te), (tr, va), (va, te)]:
            self.assertEqual(set(groups[a]) & set(groups[b]), set())
        self.assertTrue(0.12 < len(te) / 3000 < 0.18)

    def test_group_split_helpers(self):
        import numpy as np
        from evaluation import group_split_single_class, stratified_group_split
        rng = np.random.RandomState(1)
        groups = rng.randint(0, 400, 2000)
        parts = group_split_single_class(groups, 0)
        self.assertEqual(sum(len(p) for p in parts), 2000)
        self.assertTrue(0.65 < len(parts[0]) / 2000 < 0.75)
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertEqual(set(groups[parts[i]]) & set(groups[parts[j]]), set())
        y = rng.randint(0, 2, 2000)
        rest, test = stratified_group_split(y, groups, 0.5, 0)
        self.assertEqual(set(groups[rest]) & set(groups[test]), set())
        self.assertTrue(0.4 < len(test) / 2000 < 0.6)

    def test_single_class_auc_is_none(self):
        from evaluation import compute_metrics
        m = compute_metrics([1, 1], [0.2, 0.9], 0.5)
        self.assertIsNone(m["roc_auc"])
        self.assertEqual(m["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
