#!/usr/bin/env python3
# data/raw altındaki ham dataset'leri data/processed altında ortak formata çevirir:
#   text,label,source   (label: 1 = phishing/spam, 0 = legit)
#
# Çıktılar:
#   kaggle.csv        - Kaggle "Phishing Email Dataset" (temizlenmiş, dedup, SpamAssassin kopyaları çıkarılmış)
#   nazario_sa.csv    - Nazario phishing (2023+2024) + SpamAssassin ham
#   turkish_train.csv / turkish_eval.csv - Türkçe spam seti (kendi split'i)
#   holdout_2025.csv  - SADECE TEST: Nazario 2025 phishing + Apache mail listeleri
#                       2025 (meşru). Eğitim setlerinde birebir geçen mailler çıkarılır.
#   modern_legit.csv  - EĞİTİM: güncel meşru mailler (Python, Fedora, GNU listeleri, 2024).
#                       Hold-out'lardaki herhangi bir maile yakın kopya olanlar çıkarılır.
#   holdout_b_2025.csv - SADECE TEST: Ubuntu listeleri 2025 (meşru, yanlış alarm testi).
#   nazario_older.csv - EĞİTİM: Nazario 2019-2022 phishing (recall için daha fazla gerçek phishing)
#   holdout_c.csv     - SADECE TEST: Phishing Pot (2026, sadece İngilizce) + OSGeo listeleri 2025

import csv
import hashlib
import mailbox
import os
import random
import re
import sys

from email_parsing import MAX_CHARS, message_to_text, parse_bytes

RAW = os.path.join("data", "raw")
OUT = os.path.join("data", "processed")

SA_HAM_DIRS = ["easy_ham", "easy_ham_2", "hard_ham"]
NAZARIO_FILES = ["nazario-2023.mbox", "nazario-2024.mbox"]
HOLDOUT_NAZARIO = "nazario-2025.mbox"
HOLDOUT_APACHE_DIR = "apache2025"
MODERN_DIR = "modern2024"
HOLDOUT_B_DIR = "ubuntu2025"
NAZARIO_OLDER_FILES = ["nazario-2019.mbox", "nazario-2020.mbox", "nazario-2021.mbox", "nazario-2022.mbox"]
POT_DIR = "phishing_pot"
HOLDOUT_C_LEGIT_DIR = "osgeo2025"

# Phishing Pot alıcı adresini bununla anonimleştiriyor - Nazario'daki gibi siliyoruz
POT_TOKEN_REGEX = re.compile(r"phishing@pot", re.IGNORECASE)

# Phishing Pot çok dilli; proje İngilizce. Kural sonuçlara bakmadan önce sabitlendi:
# İngilizce stopword oranı diğer dillerinkinden yüksek ve en az %3 ise İngilizce.
STOPWORDS = {
    "en": set("the and you your to of for is in on this please we our with be are have will from".split()),
    "fr": set("le la les de des et vous votre pour est un une dans sur avec nous pas".split()),
    "de": set("der die das und sie ihr ihre zu ist mit auf den für nicht ein eine wir".split()),
    "pt": set("o os de do da e você seu sua para em não um uma com que por".split()),
    "es": set("el los las de del y usted su para en es un una con que por".split()),
    "it": set("il di e che per la le un una con non sono suo tuo questo".split()),
    "nl": set("de het een en van je uw is op te met voor niet wij".split()),
}
ENGLISH_MIN_RATIO = 0.03

# Fedora paket bildirimleri ayda 3.600 tane ve hepsi aynı şablon - modeli
# boğmasın diye örnekliyoruz
MODERN_SAMPLE = {"fedora-package-announce": 500}

# mailman'in subject'e eklediği liste etiketi - kaynağı ele veriyor
LIST_SUBJECT_TAGS = ["[Python-announce]"]

# eğitime giren bir mail, test setlerindeki bir maile bu kadar benziyorsa atılır
NEAR_DUP_SIMILARITY = 0.8

# Nazario maillerinin hepsi aynı kişiye gitmiş - model bu adresi "phishing" diye
# ezberlemesin diye alıcı bilgisini siliyoruz (yerine "user" koymak da
# yeni bir sızıntı token'ı yaratıyordu: phishing maillerinin %70'inde geçiyordu).
# Saldırganlar domain'den şirket adı da türetiyor ("Salary Upgrade for Monkey Staff") -
# o yüzden tek başına "monkey" de siliniyor.
RECIPIENT_REGEX = re.compile(r"jose(@monkey\.org)?|monkey\.org|\bmonkey\b", re.IGNORECASE)


def text_key(text):
    return hashlib.md5(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def dedup(rows, exclude=None):
    """Tekrarları siler. exclude: bu key'lere sahip satırlar da atılır (eğitimde
    geçen bir mailin test setine sızmaması için)."""
    seen = set(exclude) if exclude else set()
    out = []
    for r in rows:
        key = text_key(r["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def read_mbox(path, label, source, strip_recipient=False):
    rows = []
    for msg in mailbox.mbox(path, factory=lambda fp: parse_bytes(fp.read())):
        try:
            text = message_to_text(msg)
        except Exception:
            continue
        if strip_recipient:
            text = RECIPIENT_REGEX.sub("", text)
        if len(text) < 40:
            continue
        rows.append({"text": text, "label": label, "source": source})
    return rows


def write_csv(path, rows):
    f = open(path, "w", encoding="utf-8", newline="")
    w = csv.DictWriter(f, fieldnames=["text", "label", "source"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
    f.close()
    pos = sum(1 for r in rows if r["label"] == 1)
    print("  " + path + ": " + str(len(rows)) + " rows (" + str(pos) + " positive)")


def prepare_kaggle():
    path = os.path.join(RAW, "Phishing_Email.csv")
    rows = []
    f = open(path, "r", encoding="utf-8", errors="replace", newline="")
    for row in csv.DictReader(f):
        text = (row.get("Email Text") or "").strip()
        kind = (row.get("Email Type") or "").strip().lower()
        if text == "" or text == "empty" or len(text) < 20:
            continue
        if kind == "phishing email":
            label = 1
        elif kind == "safe email":
            label = 0
        else:
            continue
        rows.append({"text": text[:MAX_CHARS], "label": label, "source": "kaggle"})
    f.close()
    return dedup(rows)


def prepare_nazario_sa():
    rows = []
    for name in NAZARIO_FILES:
        rows.extend(read_mbox(os.path.join(RAW, name), 1, "nazario", strip_recipient=True))

    for d in SA_HAM_DIRS:
        folder = os.path.join(RAW, d)
        for fname in sorted(os.listdir(folder)):
            if fname == "cmds":
                continue
            f = open(os.path.join(folder, fname), "rb")
            raw = f.read()
            f.close()
            try:
                text = message_to_text(parse_bytes(raw))
            except Exception:
                continue
            if len(text) < 40:
                continue
            rows.append({"text": text, "label": 0, "source": "spamassassin_" + d})
    return dedup(rows)


def prepare_holdout(train_rows):
    rows = read_mbox(os.path.join(RAW, HOLDOUT_NAZARIO), 1, "nazario_2025", strip_recipient=True)
    folder = os.path.join(RAW, HOLDOUT_APACHE_DIR)
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".mbox"):
            continue
        # dosya adı: <proje>-<liste>-2025-<ay>.mbox
        source = "apache_" + "-".join(fname.split("-")[:2])
        rows.extend(read_mbox(os.path.join(folder, fname), 0, source))

    before = len(dedup(rows))
    out = dedup(rows, exclude={text_key(r["text"]) for r in train_rows})
    print("  holdout: removed " + str(before - len(out)) + " emails that also appear in training data")
    return out


def _source_from_filename(fname):
    # "fedora-users-2024-03.mbox" -> "fedora-users"
    return re.split(r"-20\d\d-", fname)[0]


def strip_list_tags(text):
    for tag in LIST_SUBJECT_TAGS:
        text = text.replace("Subject: " + tag + " ", "Subject: ", 1)
    return text


def read_mbox_dir(folder, label, prefix=""):
    by_source = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".mbox"):
            continue
        source = prefix + _source_from_filename(fname)
        rows = read_mbox(os.path.join(folder, fname), label, source)
        by_source.setdefault(source, []).extend(rows)
    return by_source


def prepare_modern_legit():
    by_source = read_mbox_dir(os.path.join(RAW, MODERN_DIR), 0)
    rows = []
    for source in sorted(by_source):
        items = dedup(by_source[source])
        n = MODERN_SAMPLE.get(source)
        if n and len(items) > n:
            items = random.Random(42).sample(items, n)
        for r in items:
            r["text"] = strip_list_tags(r["text"])
        rows.extend(items)
    return dedup(rows)


def prepare_holdout_b(train_rows):
    by_source = read_mbox_dir(os.path.join(RAW, HOLDOUT_B_DIR), 0)
    rows = []
    for source in sorted(by_source):
        rows.extend(by_source[source])
    return dedup(rows, exclude={text_key(r["text"]) for r in train_rows})


def stopword_ratios(text):
    words = re.findall(r"[^\W\d_]+", text.lower())
    n = max(len(words), 1)
    return {lang: sum(1 for w in words if w in sw) / n for lang, sw in STOPWORDS.items()}


def looks_english(text):
    r = stopword_ratios(text)
    en = r.pop("en")
    return en >= ENGLISH_MIN_RATIO and en > max(r.values())


def prepare_nazario_older():
    rows = []
    for name in NAZARIO_OLDER_FILES:
        rows.extend(read_mbox(os.path.join(RAW, name), 1, "nazario_older", strip_recipient=True))
    return dedup(rows)


def prepare_holdout_c(train_rows):
    rows = []
    folder = os.path.join(RAW, POT_DIR)
    total = 0
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".eml"):
            continue
        total = total + 1
        f = open(os.path.join(folder, fname), "rb")
        raw = f.read()
        f.close()
        try:
            text = message_to_text(parse_bytes(raw))
        except Exception:
            continue
        text = POT_TOKEN_REGEX.sub("", text)
        if len(text) < 40 or not looks_english(text):
            continue
        rows.append({"text": text, "label": 1, "source": "phishing_pot"})
    print("  holdout C: " + str(len(rows)) + " of " + str(total) + " Phishing Pot samples are English")

    by_source = read_mbox_dir(os.path.join(RAW, HOLDOUT_C_LEGIT_DIR), 0, prefix="osgeo_")
    for source in sorted(by_source):
        rows.extend(by_source[source])
    return dedup(rows, exclude={text_key(r["text"]) for r in train_rows})


def drop_cross_copies(rows, other_rows, name):
    """rows içinden, other_rows'taki bir maille aynı yakın kopya grubuna düşenleri atar.
    Kaggle seti SpamAssassin korpusunu (farklı formatta) içeriyor: SpamAssassin meşru
    maillerinin %92.8'inin Kaggle'da yakın kopyası vardı. Kaggle'da eğitip
    SpamAssassin'de test eden deney bu yüzden kirliydi."""
    from evaluation import near_duplicate_groups

    texts = [r["text"] for r in rows] + [r["text"] for r in other_rows]
    groups = near_duplicate_groups(texts)
    other_groups = set(groups[len(rows):].tolist())
    kept = [r for r, g in zip(rows, groups[:len(rows)]) if g not in other_groups]
    print("  " + name + ": removed " + str(len(rows) - len(kept)) + " near-copies of the other dataset")
    return kept


def drop_near_duplicates(rows, test_rows, name="modern_legit"):
    """Test setlerindeki herhangi bir maile çok benzeyen eğitim maillerini atar
    (ör. birden fazla listeye gönderilmiş aynı duyuru)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [r["text"] for r in rows]
    tests = [r["text"] for r in test_rows]
    vec = TfidfVectorizer(sublinear_tf=True).fit(texts + tests)
    best = cosine_similarity(vec.transform(texts), vec.transform(tests)).max(axis=1)
    kept = [r for r, b in zip(rows, best) if b < NEAR_DUP_SIMILARITY]
    print("  " + name + ": removed " + str(len(rows) - len(kept))
          + " near-duplicates of hold-out emails")
    return kept


def prepare_turkish(fname):
    rows = []
    f = open(os.path.join(RAW, fname), "r", encoding="utf-8", errors="replace", newline="")
    for row in csv.DictReader(f):
        text = (row.get("text") or "").strip()
        if text == "":
            continue
        rows.append({"text": text, "label": int(row["labels"]), "source": "turkish_spam"})
    f.close()
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    csv.field_size_limit(sys.maxsize)
    print("Preparing datasets:")
    nazario_sa = prepare_nazario_sa()
    kaggle = drop_cross_copies(prepare_kaggle(), nazario_sa, "kaggle")
    write_csv(os.path.join(OUT, "kaggle.csv"), kaggle)
    write_csv(os.path.join(OUT, "nazario_sa.csv"), nazario_sa)

    modern = []
    if os.path.isdir(os.path.join(RAW, MODERN_DIR)):
        modern = prepare_modern_legit()
    older = []
    if os.path.exists(os.path.join(RAW, NAZARIO_OLDER_FILES[0])):
        older = prepare_nazario_older()
    train_rows = kaggle + nazario_sa + modern + older

    holdouts = []
    if os.path.exists(os.path.join(RAW, HOLDOUT_NAZARIO)):
        holdout_a = prepare_holdout(train_rows)
        write_csv(os.path.join(OUT, "holdout_2025.csv"), holdout_a)
        holdouts.extend(holdout_a)
    if os.path.isdir(os.path.join(RAW, HOLDOUT_B_DIR)):
        holdout_b = prepare_holdout_b(train_rows)
        write_csv(os.path.join(OUT, "holdout_b_2025.csv"), holdout_b)
        holdouts.extend(holdout_b)
    if os.path.isdir(os.path.join(RAW, POT_DIR)):
        holdout_c = prepare_holdout_c(train_rows)
        write_csv(os.path.join(OUT, "holdout_c.csv"), holdout_c)
        holdouts.extend(holdout_c)

    if modern:
        if holdouts:
            modern = drop_near_duplicates(modern, holdouts)
        write_csv(os.path.join(OUT, "modern_legit.csv"), modern)
    if older:
        if holdouts:
            older = drop_near_duplicates(older, holdouts, "nazario_older")
        write_csv(os.path.join(OUT, "nazario_older.csv"), older)
    write_csv(os.path.join(OUT, "turkish_train.csv"), prepare_turkish("tr_spam_train.csv"))
    write_csv(os.path.join(OUT, "turkish_eval.csv"), prepare_turkish("tr_spam_eval.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
