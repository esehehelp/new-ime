"""synth_numeric の拡張パターン: 時刻・日付・通貨組合せ・分数・小数。

synth_numeric.py の 37K (数詞+助数詞+助詞) に追加。同スキーマで出力、
学習時は同じ synth_numeric ratio pool 内で混ざる。

パターン:
  1. 時刻: 3時30分、7時15分、午後3時、午前10時半、etc.
  2. 日付: 3月5日、2020年3月5日、令和5年、昭和63年 etc.
  3. 通貨組合せ: 3万5000円、1億2000万円、1.5億、etc.
  4. 分数: 2分の1、3分の2、1/2、etc.
  5. 小数: 0.5キロ、1.5メートル、3.14、etc.
  6. 連続数: 第1回、第2回、第3回、No.1、No.2、etc.

規則ベースなので正確。目標: 100K-200K 追加。

Usage:
    uv run python -m datasets.tools.corpus.synth_numeric_ext \
        --out datasets/v2_bunsetsu/synth_numeric_ext.jsonl \
        --target-size 150000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. 時刻
# ---------------------------------------------------------------------------

HOUR_READINGS = {
    1: "いちじ", 2: "にじ", 3: "さんじ", 4: "よじ", 5: "ごじ",
    6: "ろくじ", 7: "しちじ", 8: "はちじ", 9: "くじ", 10: "じゅうじ",
    11: "じゅういちじ", 12: "じゅうにじ",
}

MINUTE_READINGS = {
    0: "", 5: "ごふん", 10: "じゅっぷん", 15: "じゅうごふん",
    20: "にじゅっぷん", 30: "さんじゅっぷん", 45: "よんじゅうごふん",
}

# 時刻接頭語
TIME_PREFIXES = [
    ("", ""),
    ("ごぜん", "午前"),
    ("ごご", "午後"),
]

TIME_HALF = [  # 半
    ("", ""),
    ("はん", "半"),  # 3時半
]


def gen_time(rng: random.Random, n: int) -> list[dict]:
    items = []
    particles = [("を", "を"), ("に", "に"), ("の", "の"), ("から", "から"),
                 ("まで", "まで"), ("", "")]
    for _ in range(n):
        h_num = rng.randint(1, 12)
        h_read = HOUR_READINGS[h_num]
        prefix_r, prefix_s = rng.choice(TIME_PREFIXES)

        # 3 パターン: 時のみ / 時+分 / 時半
        p = rng.random()
        if p < 0.4:
            # 時のみ
            read = prefix_r + h_read
            surf = f"{prefix_s}{h_num}時"
        elif p < 0.7:
            # 時+分
            m_num = rng.choice([5, 10, 15, 20, 30, 45])
            m_read = MINUTE_READINGS[m_num]
            read = prefix_r + h_read + m_read
            surf = f"{prefix_s}{h_num}時{m_num}分"
        else:
            # 時半
            read = prefix_r + h_read + "はん"
            surf = f"{prefix_s}{h_num}時半"

        part_r, part_s = rng.choice(particles)
        items.append({
            "reading": read + part_r,
            "surface": surf + part_s,
            "left_context_surface": "",
            "left_context_reading": "",
            "span_bunsetsu": 1,
            "source": "synth_time",
            "sentence_id": f"synth_time:{len(items)}",
        })
    return items


# ---------------------------------------------------------------------------
# 2. 日付
# ---------------------------------------------------------------------------

MONTH_READINGS = {
    1: "いちがつ", 2: "にがつ", 3: "さんがつ", 4: "しがつ", 5: "ごがつ",
    6: "ろくがつ", 7: "しちがつ", 8: "はちがつ", 9: "くがつ",
    10: "じゅうがつ", 11: "じゅういちがつ", 12: "じゅうにがつ",
}

DAY_READINGS = {
    1: "ついたち", 2: "ふつか", 3: "みっか", 4: "よっか", 5: "いつか",
    6: "むいか", 7: "なのか", 8: "ようか", 9: "ここのか", 10: "とおか",
    11: "じゅういちにち", 14: "じゅうよっか", 15: "じゅうごにち",
    20: "はつか", 21: "にじゅういちにち", 30: "さんじゅうにち",
}

# 年代 (西暦・和暦)
YEAR_PATTERNS = [
    # 西暦 4桁
    ("にせんにじゅう", "2020"), ("にせんにじゅういち", "2021"),
    ("にせんにじゅうに", "2022"), ("にせんにじゅうさん", "2023"),
    ("にせんにじゅうよん", "2024"), ("にせんにじゅうご", "2025"),
    ("せんきゅうひゃくきゅうじゅう", "1990"),
    ("せんきゅうひゃくきゅうじゅうきゅう", "1999"),
    ("にせん", "2000"),
    # 令和
    ("れいわがんねん", "令和元年"), ("れいわにねん", "令和2年"),
    ("れいわさんねん", "令和3年"), ("れいわごねん", "令和5年"),
    # 平成
    ("へいせいさんねん", "平成3年"), ("へいせいじゅうねん", "平成10年"),
    ("へいせいにじゅうねん", "平成20年"),
    # 昭和
    ("しょうわにじゅうねん", "昭和20年"),
    ("しょうわろくじゅうさんねん", "昭和63年"),
]


def gen_date(rng: random.Random, n: int) -> list[dict]:
    items = []
    particles = [("を", "を"), ("に", "に"), ("の", "の"), ("から", "から"),
                 ("まで", "まで"), ("", "")]
    for _ in range(n):
        p = rng.random()
        if p < 0.3:
            # 月日のみ
            m_num = rng.randint(1, 12)
            d_num = rng.choice(list(DAY_READINGS.keys()))
            read = MONTH_READINGS[m_num] + DAY_READINGS[d_num]
            surf = f"{m_num}月{d_num}日"
        elif p < 0.6:
            # 西暦+月日
            y_read, y_surf = rng.choice(YEAR_PATTERNS[:10])  # 西暦のみ
            m_num = rng.randint(1, 12)
            d_num = rng.choice(list(DAY_READINGS.keys()))
            read = y_read + "ねん" + MONTH_READINGS[m_num] + DAY_READINGS[d_num]
            surf = f"{y_surf}年{m_num}月{d_num}日"
        elif p < 0.8:
            # 和暦+月日
            y_read, y_surf = rng.choice(YEAR_PATTERNS[10:])  # 和暦のみ
            m_num = rng.randint(1, 12)
            d_num = rng.choice(list(DAY_READINGS.keys()))
            read = y_read + MONTH_READINGS[m_num] + DAY_READINGS[d_num]
            surf = f"{y_surf}{m_num}月{d_num}日"
        else:
            # 年のみ
            y_read, y_surf = rng.choice(YEAR_PATTERNS)
            if "ねん" in y_read or "がんねん" in y_read:
                read = y_read
                surf = y_surf
            else:
                read = y_read + "ねん"
                surf = f"{y_surf}年"

        part_r, part_s = rng.choice(particles)
        items.append({
            "reading": read + part_r,
            "surface": surf + part_s,
            "left_context_surface": "",
            "left_context_reading": "",
            "span_bunsetsu": 1,
            "source": "synth_date",
            "sentence_id": f"synth_date:{len(items)}",
        })
    return items


# ---------------------------------------------------------------------------
# 3. 通貨組合せ (3万5000円、1億2000万円 etc.)
# ---------------------------------------------------------------------------

def gen_currency(rng: random.Random, n: int) -> list[dict]:
    items = []
    particles = [("を", "を"), ("に", "に"), ("の", "の"), ("で", "で"),
                 ("", "")]

    # 万+千 単位
    man_digits = list(range(1, 10))  # 1-9 万
    sen_digits = [0, 1000, 2000, 3000, 5000, 8000]
    oku_digits = list(range(1, 10))  # 1-9 億
    man_after_oku = [0, 1000, 2000, 5000, 8000]  # 0, 1000万, 2000万...

    for _ in range(n):
        p = rng.random()
        if p < 0.3:
            # n万
            n1 = rng.choice(man_digits)
            n1_r = {1: "いち", 2: "に", 3: "さん", 4: "よん", 5: "ご",
                    6: "ろく", 7: "なな", 8: "はち", 9: "きゅう"}[n1]
            read = f"{n1_r}まんえん"
            surf = f"{n1}万円"
        elif p < 0.6:
            # n万 + m千 円
            n1 = rng.choice(man_digits)
            n1_r = {1: "いち", 2: "に", 3: "さん", 4: "よん", 5: "ご",
                    6: "ろく", 7: "なな", 8: "はち", 9: "きゅう"}[n1]
            n2 = rng.choice([x for x in sen_digits if x > 0])
            n2_r = {1000: "せん", 2000: "にせん", 3000: "さんぜん",
                    5000: "ごせん", 8000: "はっせん"}[n2]
            read = f"{n1_r}まん{n2_r}えん"
            surf = f"{n1}万{n2}円"
        elif p < 0.85:
            # n億円
            n1 = rng.choice(oku_digits)
            n1_r = {1: "いち", 2: "に", 3: "さん", 4: "よん", 5: "ご",
                    6: "ろく", 7: "なな", 8: "はち", 9: "きゅう"}[n1]
            read = f"{n1_r}おくえん"
            surf = f"{n1}億円"
        else:
            # n億 m万 円
            n1 = rng.choice(oku_digits)
            n1_r = {1: "いち", 2: "に", 3: "さん", 4: "よん", 5: "ご",
                    6: "ろく", 7: "なな", 8: "はち", 9: "きゅう"}[n1]
            n2 = rng.choice([x for x in man_after_oku if x > 0])
            # Normalize man reading
            man_r = {1000: "いっせんまん", 2000: "にせんまん",
                     5000: "ごせんまん", 8000: "はっせんまん"}[n2]
            read = f"{n1_r}おく{man_r}えん"
            surf = f"{n1}億{n2}万円"

        part_r, part_s = rng.choice(particles)
        items.append({
            "reading": read + part_r,
            "surface": surf + part_s,
            "left_context_surface": "",
            "left_context_reading": "",
            "span_bunsetsu": 1,
            "source": "synth_currency",
            "sentence_id": f"synth_currency:{len(items)}",
        })
    return items


# ---------------------------------------------------------------------------
# 4. 分数 (2分の1, 3分の2 など)
# ---------------------------------------------------------------------------

def gen_fraction(rng: random.Random, n: int) -> list[dict]:
    items = []
    particles = [("を", "を"), ("に", "に"), ("の", "の"), ("で", "で"),
                 ("", "")]

    num_readings = {1: "いち", 2: "に", 3: "さん", 4: "よん", 5: "ご",
                    6: "ろく", 7: "なな", 8: "はち", 9: "きゅう",
                    10: "じゅう"}

    for _ in range(n):
        denom = rng.randint(2, 10)
        numer = rng.randint(1, denom - 1)
        d_r = num_readings[denom]
        n_r = num_readings[numer]
        read = f"{d_r}ぶんの{n_r}"
        surf = f"{denom}分の{numer}"

        part_r, part_s = rng.choice(particles)
        items.append({
            "reading": read + part_r,
            "surface": surf + part_s,
            "left_context_surface": "",
            "left_context_reading": "",
            "span_bunsetsu": 1,
            "source": "synth_fraction",
            "sentence_id": f"synth_fraction:{len(items)}",
        })
    return items


# ---------------------------------------------------------------------------
# 5. 小数 + 単位
# ---------------------------------------------------------------------------

def gen_decimal(rng: random.Random, n: int) -> list[dict]:
    items = []
    units = [("キロ", "きろ"), ("メートル", "めーとる"), ("センチ", "せんち"),
             ("グラム", "ぐらむ"), ("リットル", "りっとる"),
             ("パーセント", "ぱーせんと"), ("", "")]
    particles = [("を", "を"), ("の", "の"), ("で", "で"), ("", "")]

    num_readings = {0: "ぜろ", 1: "いち", 2: "に", 3: "さん", 4: "よん",
                    5: "ご", 6: "ろく", 7: "なな", 8: "はち", 9: "きゅう",
                    10: "じゅう"}

    for _ in range(n):
        whole = rng.randint(0, 10)
        decimal = rng.randint(1, 9)
        w_r = num_readings[whole]
        d_r = num_readings[decimal]
        read_base = f"{w_r}てん{d_r}"
        surf_base = f"{whole}.{decimal}"

        unit_s, unit_r = rng.choice(units)
        part_r, part_s = rng.choice(particles)

        read = read_base + unit_r + part_r
        surf = surf_base + unit_s + part_s

        items.append({
            "reading": read,
            "surface": surf,
            "left_context_surface": "",
            "left_context_reading": "",
            "span_bunsetsu": 1,
            "source": "synth_decimal",
            "sentence_id": f"synth_decimal:{len(items)}",
        })
    return items


# ---------------------------------------------------------------------------
# 6. 連番 (第1回、No.1 etc.)
# ---------------------------------------------------------------------------

def gen_ordinal(rng: random.Random, n: int) -> list[dict]:
    items = []
    num_readings = {1: "いち", 2: "に", 3: "さん", 4: "よん", 5: "ご",
                    6: "ろく", 7: "なな", 8: "はち", 9: "きゅう",
                    10: "じゅう", 20: "にじゅう", 30: "さんじゅう",
                    50: "ごじゅう", 100: "ひゃく"}

    patterns = [
        # (reading prefix, surface prefix)
        ("だい", "第"),
        # No. (英字系)
        # ("", "No."),
    ]
    suffixes_for_dai = [
        ("かい", "回"), ("ごう", "号"), ("しょう", "章"),
        ("はい", "杯"), ("だん", "段"), ("", ""),
    ]

    for _ in range(n):
        num = rng.choice(list(num_readings.keys()))
        n_r = num_readings[num]
        prefix_r, prefix_s = rng.choice(patterns)
        suff_r, suff_s = rng.choice(suffixes_for_dai)
        read = prefix_r + n_r + suff_r
        surf = f"{prefix_s}{num}{suff_s}"

        items.append({
            "reading": read,
            "surface": surf,
            "left_context_surface": "",
            "left_context_reading": "",
            "span_bunsetsu": 1,
            "source": "synth_ordinal",
            "sentence_id": f"synth_ordinal:{len(items)}",
        })
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-size", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # 目標サイズを 6 パターンに分配
    # time:30K, date:30K, currency:25K, fraction:20K, decimal:25K, ordinal:20K
    # = 150K default
    frac = {
        "time": 0.20,
        "date": 0.20,
        "currency": 0.17,
        "fraction": 0.13,
        "decimal": 0.17,
        "ordinal": 0.13,
    }
    all_items: list[dict] = []
    all_items.extend(gen_time(rng, int(args.target_size * frac["time"])))
    all_items.extend(gen_date(rng, int(args.target_size * frac["date"])))
    all_items.extend(gen_currency(rng, int(args.target_size * frac["currency"])))
    all_items.extend(gen_fraction(rng, int(args.target_size * frac["fraction"])))
    all_items.extend(gen_decimal(rng, int(args.target_size * frac["decimal"])))
    all_items.extend(gen_ordinal(rng, int(args.target_size * frac["ordinal"])))

    rng.shuffle(all_items)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    from collections import Counter
    src_dist = Counter(i["source"] for i in all_items)
    print(f"wrote {len(all_items):,} items -> {out}")
    print("  source distribution:")
    for s, n in src_dist.most_common():
        print(f"    {s:<20} {n:,}")

    print("\n  sample:")
    for it in all_items[:12]:
        print(f"    [{it['source']:<18}] {it['reading']:<24}  -> {it['surface']}")


if __name__ == "__main__":
    main()
