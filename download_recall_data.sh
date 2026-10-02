#!/usr/bin/env bash
# Recall deneyi için veri (~62 MB). download_data.sh tarafından çağrılır.
#   data/raw/nazario-2019..2022.mbox -> EĞİTİM: daha fazla gerçek phishing
#   data/raw/phishing_pot/           -> SADECE TEST (hold-out C): Phishing Pot, en yeni 800 örnek
#                                       (CC BY-NC 4.0 - github.com/rf-peixoto/phishing_pot)
#   data/raw/osgeo2025/              -> SADECE TEST (hold-out C): OSGeo mail listeleri 2025 (meşru)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/raw/phishing_pot data/raw/osgeo2025

echo "Nazario phishing 2019-2022 (~18.5 MB)"
for y in 2019 2020 2021 2022; do
  curl -fsSL -o "data/raw/nazario-$y.mbox" "https://monkey.org/~jose/phishing/phishing-$y"
done

echo "Phishing Pot - newest 800 samples (~40 MB)"
# repo'nun son public commit'ine sabitliyoruz, örnek listesi değişmesin
SHA=$(curl -fsSL https://api.github.com/repos/rf-peixoto/phishing_pot/commits/main \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['sha'])")
echo "$SHA" > data/raw/phishing_pot/COMMIT
curl -fsSL "https://api.github.com/repos/rf-peixoto/phishing_pot/git/trees/$SHA?recursive=1" \
  | python3 -c "
import sys, json, re
tree = json.load(sys.stdin)['tree']
nums = sorted(int(m.group(1)) for t in tree
              for m in [re.fullmatch(r'email/sample-(\d+)\.eml', t['path'])] if m)
print('\n'.join(str(n) for n in nums[-800:]))" \
  | xargs -P 8 -I{} curl -fsSL -o "data/raw/phishing_pot/sample-{}.eml" \
      "https://raw.githubusercontent.com/rf-peixoto/phishing_pot/$SHA/email/sample-{}.eml"

echo "OSGeo mailing lists 2025 (~3 MB)"
for m in January April July October; do
  for l in gdal-dev qgis-user qgis-developer; do
    curl -fsSL "https://lists.osgeo.org/pipermail/$l/2025-$m.txt.gz" 2>/dev/null | gunzip \
      > "data/raw/osgeo2025/$l-2025-$m.mbox" || rm -f "data/raw/osgeo2025/$l-2025-$m.mbox"
  done
done

echo "Done."
