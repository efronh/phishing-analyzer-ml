#!/usr/bin/env python3

import argparse
import sys

from analyzer import PhishingEmailAnalyzer
from email_parsing import looks_like_email, to_model_text


def read_input(path):
    # bytes okuyoruz: ham .eml'in MIME çözümü (base64, charset) bytes üzerinden yapılıyor
    if path:
        f = open(path, "rb")
        raw = f.read()
        f.close()
        return raw
    return sys.stdin.buffer.read()


def main():
    parser = argparse.ArgumentParser(
        description="Phishing email analyzer: rule-based engine + optional ML model."
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Email file: a raw .eml or plain text. Reads stdin if you don't pass a file.",
    )
    parser.add_argument(
        "--no-dns",
        action="store_true",
        help="Skip DNS lookup for domains",
    )
    parser.add_argument(
        "--model",
        help="Trained model file (from train.py). Its verdict then decides the exit code. "
             "Only load model files you trained yourself (joblib = pickle).",
    )
    args = parser.parse_args()

    raw = read_input(args.file)
    if raw.strip() == b"":
        print("Error: empty input.", file=sys.stderr)
        return 1

    # model eğitimdeki formatı görmeli: ham mail ise MIME çözülüp "Subject + gövde"ye
    # çevriliyor (eskiden ham metin veriliyordu ve kararların %15'i değişiyordu)
    if looks_like_email(raw):
        print("Input: raw email -> analyzing subject + decoded body")
        print("")
    text = to_model_text(raw)

    analyzer = PhishingEmailAnalyzer(resolve_dns=not args.no_dns)
    result = analyzer.analyze(text)
    print(result.summary())

    if args.model:
        # sklearn sadece model kullanılınca gerekli, o yüzden import burada
        from ml_model import PhishingClassifier

        model = PhishingClassifier.load(args.model)
        ml_result = model.analyze(text)
        print("")
        print(ml_result.summary())
        # model varsa karar onun - README'deki rakamlar da bu karara ait
        return 1 if ml_result.is_phishing else 0

    if result.risk_level in ("LOW", "MEDIUM"):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
