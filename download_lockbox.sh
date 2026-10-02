#!/usr/bin/env bash
# Kilitli (lockbox) test seti (~38 MB). Bütün geliştirme bittikten sonra evaluate_lockbox.py
# ile TEK BİR KEZ ölçülür; eğitimde, model seçiminde ve diğer testlerde hiç kullanılmaz.
#   data/raw/lockbox/phishing_pot/  Phishing Pot'un hold-out C'de KULLANILMAMIŞ eski
#                                   örneklerinden rastgele 1.000 tane (seed 2026, sabit commit)
#   data/raw/lockbox/legit2026/     ISC bind-users, Samba, Mailman mailman-users (2026)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/raw/lockbox/phishing_pot data/raw/lockbox/legit2026

echo "Phishing Pot - 1000 random older samples not used in hold-out C (~37 MB)"
SHA=49f63777126b0bdb9eb1f6e770a5c3f9df2b0306
echo "$SHA" > data/raw/lockbox/phishing_pot/COMMIT
curl -fsSL "https://api.github.com/repos/rf-peixoto/phishing_pot/git/trees/$SHA?recursive=1" \
  | python3 -c "
import sys, json, re, random
tree = json.load(sys.stdin)['tree']
nums = sorted(int(m.group(1)) for t in tree
              for m in [re.fullmatch(r'email/sample-(\d+)\.eml', t['path'])] if m)
older = nums[:-800]                      # en yeni 800 = hold-out C
random.seed(2026)
print('\n'.join(str(n) for n in sorted(random.sample(older, 1000))))" \
  | tee data/raw/lockbox/phishing_pot/SAMPLES \
  | xargs -P 8 -I{} curl -fsSL -o "data/raw/lockbox/phishing_pot/sample-{}.eml" \
      "https://raw.githubusercontent.com/rf-peixoto/phishing_pot/$SHA/email/sample-{}.eml"

echo "Legitimate mailing lists 2026: ISC, Samba, Mailman (~0.5 MB)"
for m in January April July; do
  curl -fsSL "https://lists.isc.org/pipermail/bind-users/2026-$m.txt.gz" | gunzip \
    > "data/raw/lockbox/legit2026/isc-bind-users-2026-$m.mbox"
  curl -fsSL "https://lists.samba.org/archive/samba/2026-$m.txt.gz" | gunzip \
    > "data/raw/lockbox/legit2026/samba-samba-2026-$m.mbox"
done
for m in 01 04 07; do
  next=$(printf "%02d" $((10#$m + 1)))
  curl -fsSL "https://lists.mailman3.org/archives/list/mailman-users@mailman3.org/export/mailman-users@mailman3.org-2026-$m.mbox.gz?start=2026-$m-01&end=2026-$next-01" \
    | gunzip > "data/raw/lockbox/legit2026/mailman-users-2026-$m.mbox"
done

echo "Done."
