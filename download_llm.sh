#!/usr/bin/env bash
# LLM ile yazılmış phishing test setleri (SADECE TEST, eğitimde kullanılmıyor):
#   Zenodo 10.5281/zenodo.20250116 (Gutierrez et al. 2026, CC BY 4.0): GPT-4.1, DeepSeek 3.2, Llama 3.3
#   Kaggle francescogreco97/human-llm-generated-phishing-legitimate-emails (Greco et al. 2024, CC BY-NC-SA 4.0)
# Kullanım: ./download_llm.sh && python experiments_llm.py
set -euo pipefail
cd "$(dirname "$0")"
OUT=data/raw/llm
mkdir -p "$OUT"

curl -sL -o "$OUT/cross-model-phishing.zip" \
  "https://zenodo.org/api/records/20250116/files/cross-model-phishing.zip/content"
echo "7bd6097768e0f44e433a153a77b374f5  $OUT/cross-model-phishing.zip" | md5sum -c - 2>/dev/null \
  || [ "$(md5 -q "$OUT/cross-model-phishing.zip")" = "7bd6097768e0f44e433a153a77b374f5" ]
unzip -q -o "$OUT/cross-model-phishing.zip" "data/*" README.md LICENSE -d "$OUT/zenodo"

curl -sL -o "$OUT/greco.zip" \
  "https://www.kaggle.com/api/v1/datasets/download/francescogreco97/human-llm-generated-phishing-legitimate-emails"
unzip -q -o "$OUT/greco.zip" -d "$OUT/greco"
echo "LLM test sets in $OUT"
