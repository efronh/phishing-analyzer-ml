#!/usr/bin/env bash
# Public dataset'leri data/raw altına indirir (~100 MB). Sonra: python prepare_data.py
set -euo pipefail
mkdir -p data/raw
cd data/raw

echo "Kaggle Phishing Email Dataset (HF mirror, ~52 MB)"
curl -fSL -o Phishing_Email.csv \
  https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/resolve/main/Phishing_Email.csv

echo "Nazario phishing corpus 2023 + 2024 (CC-BY 4.0, ~23 MB)"
for y in 2023 2024; do
  curl -fSL -o "nazario-$y.mbox" "https://monkey.org/~jose/phishing/phishing-$y"
done
curl -fSL -o nazario-LICENSE.txt https://monkey.org/~jose/phishing/LICENSE.txt

echo "SpamAssassin public corpus - ham only (~4 MB)"
for f in 20030228_easy_ham 20030228_easy_ham_2 20030228_hard_ham; do
  curl -fSL "https://spamassassin.apache.org/old/publiccorpus/$f.tar.bz2" | tar xjf -
done

echo "Turkish spam email dataset (~370 KB)"
for f in spam_train spam_eval; do
  curl -fSL -o "tr_$f.csv" "https://huggingface.co/datasets/anilguven/turkish_spam_email/resolve/main/$f.csv"
done

echo "Hold-out test set: Nazario 2025 + Apache mailing lists 2025 (~42 MB)"
curl -fSL -o nazario-2025.mbox https://monkey.org/~jose/phishing/phishing-2025
mkdir -p apache2025
for l in users@tomcat.apache.org users@httpd.apache.org user@spark.apache.org users@kafka.apache.org \
         user@flink.apache.org dev@httpd.apache.org announce@apache.org users@maven.apache.org; do
  for m in 02 05 08; do
    name=${l%@*}; domain=${l#*@}
    curl -fsSL -o "apache2025/${domain%%.*}-$name-2025-$m.mbox" \
      "https://lists.apache.org/api/mbox.lua?list=$name&domain=$domain&d=2025-$m"
  done
done

cd ../..
echo "Modern legitimate mailing lists (training: 2024, hold-out B: 2025)"
./download_modern_legit.sh

echo "Recall experiment: older Nazario (training) + Phishing Pot / OSGeo (hold-out C)"
./download_recall_data.sh

echo "Lockbox test set (evaluated once at the very end, see reports/LOCKBOX_PROTOCOL.md)"
./download_lockbox.sh

echo "Done."
