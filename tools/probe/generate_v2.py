"""probe_v2 を 500+ 項目に拡張する生成スクリプト。

設計方針:
  curated (手書きリスト)
    - numeric:    数詞 × 助数詞の網羅 (80 items)
    - edge:       かな保持が正解の副詞/接続詞/感動詞 (70 items)
    - homophone:  同音異義語ペアから最頻出を選択 (60 items)
    - particle:   複合助詞 (40 items)

  corpus-sampled (v2 corpus から fugashi で bunsetsu 抽出)
    - general:    wikibooks + tatoeba + aozora_dialogue (高頻度短句 80 items)
    - tech:       wikibooks (技術トピック含む文から 60 items)
    - names:      wiktionary + wikinews (固有名詞中心 60 items)

出力: datasets/probe_v2/probe.tsv (旧 50 項目版を上書き)

Usage:
    uv run python -m tools.probe.generate_v2
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import fugashi

TAGGER = fugashi.Tagger()

HIRA = re.compile(r"^[\u3041-\u309F]+$")
KANJI = re.compile(r"[\u4E00-\u9FFF]")
CONTENT_POS = {"名詞", "動詞", "形容詞", "副詞"}
PARTICLE_POS = {"助詞", "助動詞", "接尾辞"}

# ---------------------------------------------------------------------------
# curated: numeric (80 items)
# ---------------------------------------------------------------------------

NUMBER_READINGS = {
    # 単独数詞 (generic)
    "1": ("いち", "いち"),
    "2": ("に", "に"),
    "3": ("さん", "さん"),
    "5": ("ご", "ご"),
    "10": ("じゅう", "じゅう"),
    "100": ("ひゃく", "ひゃく"),
    "1000": ("せん", "せん"),
    "10000": ("いちまん", "いちまん"),
}

# (助数詞,  surface表記, 1/3/5/10/100/1000 の reading)
COUNTERS = [
    # (助数詞漢字, 1ぴきみたいな読み, particle)
    ("本", {1: "いっぽん", 2: "にほん", 3: "さんぼん", 5: "ごほん", 10: "じゅっぽん",
            100: "ひゃっぽん", 1000: "せんぼん"}),
    ("個", {1: "いっこ", 2: "にこ", 3: "さんこ", 5: "ごこ", 10: "じゅっこ",
            100: "ひゃっこ", 1000: "せんこ"}),
    ("人", {1: "ひとり", 2: "ふたり", 3: "さんにん", 5: "ごにん", 10: "じゅうにん",
            100: "ひゃくにん", 1000: "せんにん"}),
    ("回", {1: "いっかい", 2: "にかい", 3: "さんかい", 5: "ごかい", 10: "じゅっかい",
            100: "ひゃっかい", 1000: "せんかい"}),
    ("時", {1: "いちじ", 2: "にじ", 3: "さんじ", 5: "ごじ", 10: "じゅうじ"}),
    ("分", {1: "いっぷん", 2: "にふん", 3: "さんぷん", 5: "ごふん", 10: "じゅっぷん",
            100: "ひゃっぷん"}),
    ("円", {1: "いちえん", 2: "にえん", 3: "さんえん", 5: "ごえん", 10: "じゅうえん",
            100: "ひゃくえん", 1000: "せんえん", 10000: "いちまんえん"}),
    ("年", {1: "いちねん", 2: "にねん", 3: "さんねん", 5: "ごねん", 10: "じゅうねん",
            100: "ひゃくねん", 1000: "せんねん", 10000: "いちまんねん"}),
    ("月", {1: "いっかげつ", 2: "にかげつ", 3: "さんかげつ", 5: "ごかげつ",
            10: "じゅっかげつ"}),
    ("日", {1: "ついたち", 2: "ふつか", 3: "みっか", 5: "いつか", 10: "とおか"}),
    ("センチ", {1: "いっせんち", 2: "にせんち", 3: "さんせんち", 5: "ごせんち",
                10: "じゅっせんち", 100: "ひゃくせんち"}),
    ("パーセント", {1: "いっぱーせんと", 2: "にぱーせんと", 3: "さんぱーせんと",
                    5: "ごぱーせんと", 10: "じゅっぱーせんと", 100: "ひゃくぱーせんと"}),
    ("ミリ", {1: "いちみり", 5: "ごみり", 10: "じゅうみり", 100: "ひゃくみり",
              1000: "せんみり"}),
    ("グラム", {1: "いちぐらむ", 5: "ごぐらむ", 10: "じゅうぐらむ",
                100: "ひゃくぐらむ", 1000: "せんぐらむ"}),
    ("キロ", {1: "いちきろ", 3: "さんきろ", 5: "ごきろ", 10: "じゅっきろ",
              100: "ひゃくきろ"}),
    ("メートル", {1: "いちめーとる", 3: "さんめーとる", 10: "じゅうめーとる",
                  100: "ひゃくめーとる", 1000: "せんめーとる"}),
    ("歳", {1: "いっさい", 3: "さんさい", 5: "ごさい", 10: "じゅっさい",
            20: None}),  # 20 歳 = はたち (特殊) 除外
    ("冊", {1: "いっさつ", 2: "にさつ", 3: "さんさつ", 5: "ごさつ", 10: "じゅっさつ"}),
    ("匹", {1: "いっぴき", 2: "にひき", 3: "さんびき", 5: "ごひき", 10: "じゅっぴき"}),
    ("枚", {1: "いちまい", 2: "にまい", 3: "さんまい", 5: "ごまい", 10: "じゅうまい",
            100: "ひゃくまい"}),
]


def gen_numeric() -> list[tuple[str, str]]:
    """Return list of (reading, surface) for numeric+counter combinations."""
    items: list[tuple[str, str]] = []
    particles = ["を", "に", "の", "が"]
    rng = random.Random(42)
    for kanji, readings in COUNTERS:
        for num, rd in readings.items():
            if rd is None:
                continue
            p = rng.choice(particles)
            items.append((rd + p, f"{num}{kanji}{p}"))
    return items


# ---------------------------------------------------------------------------
# curated: edge (kana-only, adverbs/discourse markers). 70 items.
# ---------------------------------------------------------------------------

EDGE_WORDS = [
    "やっぱり", "ちゃんと", "もちろん", "たぶん", "きっと", "ちょうど", "たしか",
    "いつか", "いつも", "あんまり", "だいたい", "ずっと", "だんだん", "だいじょうぶ",
    "こっそり", "しっかり", "うっかり", "すっかり", "ゆっくり", "ひっそり", "ぼんやり",
    "はっきり", "さっぱり", "のんびり", "あっさり", "きっかり", "ばっちり", "てっきり",
    "なんとなく", "いわゆる", "ちょっと", "そっと", "ぐっと", "はっと", "ふっと",
    "まさか", "わざわざ", "いきなり", "どんどん", "いろいろ", "ちらちら", "ぺらぺら",
    "ぽつぽつ", "ざあざあ", "ごろごろ", "ぱらぱら", "わくわく", "どきどき", "はらはら",
    "ぐるぐる", "ぴかぴか", "にこにこ", "すやすや", "ぐっすり", "ますます", "どうも",
    "たいして", "せっかく", "やけに", "なんだか", "なぜか", "どうやら", "おそらく",
    "たしかに", "さすがに", "ほとんど", "とりあえず", "しばらく", "むしろ", "かえって",
    "どうしても",
]


def gen_edge() -> list[tuple[str, str]]:
    # reading == surface (正解はひらがな保持)
    return [(w, w) for w in EDGE_WORDS]


# ---------------------------------------------------------------------------
# curated: particle (複合助詞). 40 items.
# ---------------------------------------------------------------------------

PARTICLES = [
    "にとっては", "においては", "においても", "としては", "としても", "からこそ",
    "だからこそ", "までには", "までにも", "によって", "によっては", "について",
    "についても", "に対して", "に対しては", "に関して", "に関しては", "という",
    "ということで", "ということは", "というのは", "というより", "というのも",
    "といっても", "ばかりでなく", "だけでなく", "のみならず", "にもかかわらず",
    "にかかわらず", "ともに", "とともに", "なしには", "なくしては", "をめぐって",
    "をめぐり", "を通じて", "を通して", "に基づいて", "に基づき", "に従って",
]


def gen_particle() -> list[tuple[str, str]]:
    """Use fugashi reading for each particle string (ensures accuracy)."""
    out = []
    for p in PARTICLES:
        reading = _fugashi_reading(p)
        out.append((reading, p))
    return out


# ---------------------------------------------------------------------------
# curated: homophone (最頻出の正解を優先). 60 items.
# Source: multi-surface kana candidates chosen so that the "expected" is the
# most-common reading in modern Japanese text.
# ---------------------------------------------------------------------------

HOMOPHONE_ITEMS = [
    # 単独 → multi-surface kana。expected は最頻出表記。
    ("はしを", "橋を"),       # (箸を/端を)
    ("はしで", "橋で"),
    ("かみを", "紙を"),       # (髪を/神を)
    ("かみが", "髪が"),
    ("あめが", "雨が"),       # (飴が)
    ("あめを", "飴を"),
    ("きを", "気を"),         # (木を/黄を)
    ("きが", "気が"),
    ("くもが", "雲が"),       # (蜘蛛が)
    ("こうえんで", "公園で"), # (講演で/後援で/公演で)
    ("こうえんに", "公演に"),
    ("せいさんが", "生産が"), # (清算が/精算が)
    ("せいさんを", "清算を"),
    ("こうせいの", "構成の"), # (厚生の/校正の/公正の)
    ("こうせいを", "校正を"),
    ("はかる", "測る"),       # (計る/図る/諮る/謀る/量る)
    ("はかった", "図った"),
    ("きかい", "機会"),       # (機械/器械)
    ("きかいを", "機械を"),
    ("かんしん", "関心"),     # (感心/歓心/寒心)
    ("かんしんを", "感心を"),
    ("ついきゅう", "追求"),   # (追究/追及)
    ("ついきゅうを", "追及を"),
    ("たいしょう", "対象"),   # (対称/対照/大賞)
    ("たいしょうに", "対照に"),
    ("いどう", "移動"),       # (異動/異同)
    ("いどうが", "移動が"),
    ("せいか", "成果"),       # (聖歌/製菓/青果/正価)
    ("せいかが", "成果が"),
    ("こうい", "行為"),       # (好意/厚意/高位)
    ("こういを", "好意を"),
    ("しじ", "支持"),         # (指示/師事/私事/四時)
    ("しじを", "指示を"),
    ("かんしょう", "鑑賞"),   # (干渉/感傷/観賞)
    ("かんしょうする", "干渉する"),
    ("ほしょう", "保証"),     # (保障/補償)
    ("ほしょうを", "補償を"),
    ("しゅうせい", "修正"),   # (習性/終生/集成)
    ("しゅうせいを", "修正を"),
    ("こくじ", "告示"),       # (酷似/国字)
    ("こくじを", "告示を"),
    ("ようけん", "用件"),     # (要件/要検)
    ("ようけんを", "要件を"),
    ("じき", "時期"),         # (時機/次期/磁気)
    ("じきが", "時期が"),
    ("しょうがい", "生涯"),   # (障害/渉外)
    ("しょうがいを", "障害を"),
    ("いがい", "以外"),       # (意外/遺骸)
    ("いがいに", "意外に"),
    ("きせい", "規制"),       # (既製/既成/帰省/寄生)
    ("きせいを", "既製を"),
    ("こうしょう", "交渉"),   # (高尚/公称/考証)
    ("こうしょうを", "交渉を"),
    ("せいしん", "精神"),     # (誠心/清新/西進)
    ("せいしんが", "精神が"),
    ("しんりょう", "診療"),   # (新涼/心療)
    ("しんりょうを", "診療を"),
    ("しんかん", "新刊"),     # (震撼/森閑/神官)
    ("しんかんが", "新刊が"),
    ("しょうひ", "消費"),     # (小比/焼費)
    ("しょうひが", "消費が"),
    ("ちゅうしん", "中心"),   # (忠心/衷心)
    ("ちゅうしんに", "中心に"),
    ("かくしん", "確信"),     # (革新/核心/隔心)
    ("かくしんが", "確信が"),
    ("じしん", "自信"),       # (自身/地震/磁針)
    ("じしんが", "自信が"),
    ("しき", "式"),           # (四季/士気/指揮/死期)
    ("しきが", "式が"),
    ("きかん", "期間"),       # (機関/帰還/気管)
    ("きかんが", "期間が"),
    ("しゅうり", "修理"),
    ("しゅうりを", "修理を"),
]


def gen_homophone() -> list[tuple[str, str]]:
    return list(HOMOPHONE_ITEMS)


# ---------------------------------------------------------------------------
# corpus-sampled: general / tech / names (fugashi bunsetsu extraction)
# ---------------------------------------------------------------------------

def kata_to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in s)


def _fugashi_reading(surface: str) -> str:
    parts = []
    for w in TAGGER(surface):
        kana = getattr(w.feature, "kana", None) or getattr(w.feature, "pron", None)
        parts.append(kana if kana and kana != "*" else w.surface)
    return kata_to_hira("".join(parts))


def extract_bunsetsu(sentence: str) -> list[tuple[str, str]]:
    """Extract (reading, surface) bunsetsu-like chunks from a sentence.

    Rule of thumb: 自立語 (content) + 付属語 (particle/aux) up to next content.
    Only return chunks that contain at least one kanji and are 3-10 chars.
    """
    tokens = list(TAGGER(sentence))
    if not tokens:
        return []
    out: list[tuple[str, str]] = []
    cur_surface: list[str] = []
    cur_reading: list[str] = []
    in_chunk = False

    def flush():
        if not cur_surface:
            return
        surf = "".join(cur_surface)
        read = kata_to_hira("".join(cur_reading))
        if 3 <= len(surf) <= 10 and KANJI.search(surf) and HIRA.match(read):
            out.append((read, surf))

    for w in tokens:
        pos = getattr(w.feature, "pos1", "?")
        kana = getattr(w.feature, "kana", None) or getattr(w.feature, "pron", None)
        kana = kana if kana and kana != "*" else w.surface
        if pos in CONTENT_POS:
            if in_chunk:
                flush()
                cur_surface, cur_reading = [], []
            cur_surface.append(w.surface)
            cur_reading.append(kana)
            in_chunk = True
        elif pos in PARTICLE_POS and in_chunk:
            cur_surface.append(w.surface)
            cur_reading.append(kana)
        else:
            flush()
            cur_surface, cur_reading = [], []
            in_chunk = False
    flush()
    return out


def sample_from_corpus(path: Path, n: int, seed: int,
                       predicate=None) -> list[tuple[str, str]]:
    """Randomly sample `n` high-freq bunsetsu from a JSONL corpus."""
    rng = random.Random(seed)
    if not path.exists():
        print(f"  [warn] {path} not found")
        return []

    # Reservoir sample 2000 sentences, extract bunsetsu, frequency-rank.
    reservoir: list[str] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 2000:
                try:
                    reservoir.append(json.loads(line).get("surface", ""))
                except Exception:
                    continue
            else:
                j = rng.randrange(i + 1)
                if j < 2000:
                    try:
                        reservoir[j] = json.loads(line).get("surface", "")
                    except Exception:
                        continue

    counter: Counter[tuple[str, str]] = Counter()
    for s in reservoir:
        if predicate and not predicate(s):
            continue
        for pair in extract_bunsetsu(s):
            counter[pair] += 1

    # Take the top 4n by frequency, then random-sample n to avoid the same
    # ubiquitous bunsetsu dominating.
    top = [pair for pair, c in counter.most_common(4 * n) if c >= 2]
    rng.shuffle(top)
    return top[:n]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    out_path = Path("datasets/probe_v2/probe.tsv")

    print("== curated ==")
    numeric = gen_numeric()
    edge = gen_edge()
    particle = gen_particle()
    homophone = gen_homophone()
    print(f"  numeric:    {len(numeric)}")
    print(f"  edge:       {len(edge)}")
    print(f"  particle:   {len(particle)}")
    print(f"  homophone:  {len(homophone)}")

    print("\n== corpus-sampled ==")
    general = sample_from_corpus(
        Path("datasets/v2/tatoeba_v2.jsonl"), 80, seed=1)
    tech = sample_from_corpus(
        Path("datasets/v2/wikibooks_v2.clean.jsonl"), 60, seed=2,
        predicate=lambda s: any(w in s for w in ["プログラム", "関数", "データ",
                                                  "アルゴリズム", "サーバー",
                                                  "ネットワーク", "回路", "方程式",
                                                  "定理", "化学", "物理", "原子",
                                                  "分子", "細胞", "生物", "統計"]))
    names = sample_from_corpus(
        Path("datasets/v2/wikinews_v2.clean.jsonl"), 60, seed=3,
        predicate=lambda s: True)
    print(f"  general:    {len(general)}")
    print(f"  tech:       {len(tech)}")
    print(f"  names:      {len(names)}")

    total = (len(numeric) + len(edge) + len(particle) + len(homophone)
             + len(general) + len(tech) + len(names))
    print(f"\n  TOTAL: {total}")

    # Write output
    lines = [
        "# Phrase-level probe v2 — expanded to ~500 items",
        "# Format: category<TAB>reading<TAB>expected_surface",
        "# Generated by tools.probe.generate_v2 — curated + corpus-sampled mix.",
        "",
    ]
    for tag, items in [("numeric", numeric), ("edge", edge),
                        ("particle", particle), ("homophone", homophone),
                        ("general", general), ("tech", tech),
                        ("names", names)]:
        lines.append(f"# ---- {tag} ({len(items)}) ----")
        for reading, surface in items:
            lines.append(f"{tag}\t{reading}\t{surface}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[written] {out_path}  ({total} items)")


if __name__ == "__main__":
    main()
