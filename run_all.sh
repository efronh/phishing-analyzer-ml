#!/usr/bin/env bash
# Bütün raporları ve üretim modelini aynı kod + aynı veriyle baştan üretir (~45 dk).
# Önce veriyi indir: ./download_data.sh
#
# experiments_tuning.py bilerek yok: varsayılan hiperparametreleri bir kez seçti
# (reports/TUNING.md). Seçilen ayar artık varsayılan olduğu için tekrar çalıştırmak
# sadece onu doğrular (~25 dk): ./run_all.sh --with-tuning
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

step() { echo; echo "=== $* ($(date +%H:%M:%S))"; }

step "prepare data";               $PY prepare_data.py
step "unit tests";                 $PY -m unittest
step "model comparison";           $PY experiments.py
step "false-alarm study";          $PY experiments_false_alarms.py
step "recall study";               $PY experiments_recall.py
if [ "${1:-}" = "--with-tuning" ]; then
  step "hyperparameter search";    $PY experiments_tuning.py
fi
step "URL feature study";        $PY experiments_url.py
step "train production model";     $PY train.py
step "evaluate on hold-outs";      $PY evaluate_holdout.py
step "threshold stability";        $PY threshold_stability.py
step "done"
