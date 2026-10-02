#!/usr/bin/env bash
# Güncel, bol linkli meşru mailler (~6 MB). download_data.sh tarafından çağrılır.
#   data/raw/modern2024/  -> EĞİTİM: Python, Fedora, GNU listeleri (2024)
#   data/raw/ubuntu2025/  -> SADECE TEST (hold-out B): Ubuntu listeleri (2025)
# Apache listeleri bilerek yok: Apache sürüm duyuruları announce@apache.org'a da
# gidiyor ve o liste hold-out A'nın içinde.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/raw/modern2024 data/raw/ubuntu2025

MONTHS_2024="01 02 03 04 05 06 07 08 09 10 11 12"

next_month() { # 2024-05 -> 2024-06-01
  local y=${1%-*} m=$((10#${1#*-}))
  if [ "$m" -eq 12 ]; then echo "$((y + 1))-01-01"; else printf "%s-%02d-01" "$y" $((m + 1)); fi
}

# mailman3 / hyperkitty export: host list YYYY-MM out
mm3() {
  local start="$3-01" end
  end=$(next_month "$3")
  curl -fsSL "https://$1/archives/list/$2/export/$2-$3.mbox.gz?start=$start&end=$end" | gunzip > "$4"
}

echo "Python (2024)"
for m in $MONTHS_2024; do
  mm3 mail.python.org python-announce-list@python.org "2024-$m" "data/raw/modern2024/python-announce-2024-$m.mbox"
done
for m in 01 04 07 10; do
  mm3 mail.python.org python-list@python.org "2024-$m" "data/raw/modern2024/python-list-2024-$m.mbox"
done

echo "Fedora (2024)"
for m in $MONTHS_2024; do
  mm3 lists.fedoraproject.org devel-announce@lists.fedoraproject.org "2024-$m" \
    "data/raw/modern2024/fedora-devel-announce-2024-$m.mbox"
done
mm3 lists.fedoraproject.org package-announce@lists.fedoraproject.org 2024-05 \
  data/raw/modern2024/fedora-package-announce-2024-05.mbox
for m in 03 09; do
  mm3 lists.fedoraproject.org users@lists.fedoraproject.org "2024-$m" "data/raw/modern2024/fedora-users-2024-$m.mbox"
done

echo "GNU (2024)"
for m in $MONTHS_2024; do
  curl -fsSL -o "data/raw/modern2024/gnu-info-gnu-2024-$m.mbox" "https://lists.gnu.org/archive/mbox/info-gnu/2024-$m"
done
for m in 02 08; do
  curl -fsSL -o "data/raw/modern2024/gnu-help-bash-2024-$m.mbox" "https://lists.gnu.org/archive/mbox/help-bash/2024-$m"
done

echo "Ubuntu (2025, hold-out B - test only)"
for m in January April July October; do
  for l in ubuntu-security-announce ubuntu-users ubuntu-announce; do
    # bazı aylarda ubuntu-announce boş / yok
    curl -fsSL "https://lists.ubuntu.com/archives/$l/2025-$m.txt.gz" 2>/dev/null | gunzip \
      > "data/raw/ubuntu2025/$l-2025-$m.mbox" || rm -f "data/raw/ubuntu2025/$l-2025-$m.mbox"
  done
done

echo "Done."
