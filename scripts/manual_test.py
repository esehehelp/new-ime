"""Manual test for AR baseline with 100 test cases.

Usage:
    uv run python -m scripts.manual_test --checkpoint checkpoints/ar_baseline/best.pt
"""

from __future__ import annotations

import argparse
import sys

import torch

from src.data.dataset import ARCollator
from src.training.train_ar import SimpleGPT2

sys.stdout.reconfigure(encoding="utf-8")

# 100 test cases: (context, reading, expected_surface)
TEST_CASES = [
    # === 基本変換 (1-20) ===
    ("", "きょうはいいてんきですね", "今日はいい天気ですね"),
    ("", "とうきょうとしぶやく", "東京都渋谷区"),
    ("", "がっこうにいく", "学校に行く"),
    ("", "しんぶんをよむ", "新聞を読む"),
    ("", "けいざいのせいちょう", "経済の成長"),
    ("", "あしたはあめがふるでしょう", "明日は雨が降るでしょう"),
    ("", "かれはとてもやさしいひとです", "彼はとても優しい人です"),
    ("", "やまにのぼる", "山に登る"),
    ("", "うみでおよぐ", "海で泳ぐ"),
    ("", "えきまであるく", "駅まで歩く"),
    ("", "ほんをかう", "本を買う"),
    ("", "てがみをかく", "手紙を書く"),
    ("", "おんがくをきく", "音楽を聴く"),
    ("", "えいがをみる", "映画を見る"),
    ("", "りょうりをつくる", "料理を作る"),
    ("", "しごとがおわった", "仕事が終わった"),
    ("", "でんしゃにのる", "電車に乗る"),
    ("", "ともだちにあう", "友達に会う"),
    ("", "にほんごをべんきょうする", "日本語を勉強する"),
    ("", "せんせいにしつもんする", "先生に質問する"),
    # === 同音異義語 (21-40) ===
    ("", "かんじへんかんのせいどをひょうかする", "漢字変換の精度を評価する"),
    ("", "こうしょうがなんこうしている", "交渉が難航している"),
    ("", "にほんのれきしをまなぶ", "日本の歴史を学ぶ"),
    ("", "きかいがこしょうした", "機械が故障した"),
    ("", "かがくのじっけんをする", "科学の実験をする"),
    ("", "しぜんかんきょうをまもる", "自然環境を守る"),
    ("", "せいじかのえんぜつをきく", "政治家の演説を聞く"),
    ("", "きしゃがきしゃできしゃした", "記者が汽車で帰社した"),
    ("", "いしがいしをもっていしをつたえた", "医師が意志を持って意思を伝えた"),
    ("", "しんりがくをけんきゅうする", "心理学を研究する"),
    ("", "ほけんにかにゅうする", "保険に加入する"),
    ("", "かぶしきしじょうのどうこう", "株式市場の動向"),
    ("", "こうつうじこがはっせいした", "交通事故が発生した"),
    ("", "きょういくかいかくをすすめる", "教育改革を進める"),
    ("", "ちほうじちたいのざいせい", "地方自治体の財政"),
    ("", "かんきょうもんだいにとりくむ", "環境問題に取り組む"),
    ("", "いりょうひのふたんがおおきい", "医療費の負担が大きい"),
    ("", "じょうほうぎじゅつのしんぽ", "情報技術の進歩"),
    ("", "ぶんかいさんのほぞん", "文化遺産の保存"),
    ("", "こくさいかんけいのあんてい", "国際関係の安定"),
    # === 文脈依存 (41-60) ===
    ("今日は天気が良いので、", "さんぽにいきましょう", "散歩に行きましょう"),
    ("プログラミングの", "きほんをまなぶ", "基本を学ぶ"),
    ("大学で", "けいざいがくをせんこうする", "経済学を専攻する"),
    ("昨日の会議で", "しんきじぎょうのていあんがあった", "新規事業の提案があった"),
    ("彼は記者として", "きしゃにのってしゅっちょうした", "汽車に乗って出張した"),
    ("東京から大阪まで", "しんかんせんでにじかんかかる", "新幹線で二時間かかる"),
    ("試験に合格するために", "まいにちべんきょうしている", "毎日勉強している"),
    ("医者に", "くすりをしょほうされた", "薬を処方された"),
    ("図書館で", "しずかにほんをよむ", "静かに本を読む"),
    ("来月の", "しゅっちょうのよていをたてる", "出張の予定を立てる"),
    ("友人の結婚式に", "しゅっせきする", "出席する"),
    ("子供たちが公園で", "たのしそうにあそんでいる", "楽しそうに遊んでいる"),
    ("新しいレストランで", "ばんごはんをたべた", "晩御飯を食べた"),
    ("会社の", "ぎょうせきがこうちょうだ", "業績が好調だ"),
    ("冬になると", "ゆきがたくさんふる", "雪がたくさん降る"),
    ("先生が生徒に", "しゅくだいをだした", "宿題を出した"),
    ("政府は", "しんたいさくをはっぴょうした", "新対策を発表した"),
    ("選手たちは", "ゆうしょうをめざしてがんばっている", "優勝を目指して頑張っている"),
    ("病院で", "けんこうしんだんをうけた", "健康診断を受けた"),
    ("空港に", "ひこうきがとうちゃくした", "飛行機が到着した"),
    # === 固有名詞 (61-70) ===
    ("", "とうきょうすかいつりー", "東京スカイツリー"),
    ("", "ほっかいどうのさっぽろし", "北海道の札幌市"),
    ("", "きょうとのきんかくじ", "京都の金閣寺"),
    ("", "おおさかじょうをけんがくする", "大阪城を見学する"),
    ("", "ふじさんにのぼる", "富士山に登る"),
    ("", "なごやしにすんでいる", "名古屋市に住んでいる"),
    ("", "おきなわのうみはきれいだ", "沖縄の海はきれいだ"),
    ("", "ひろしまのへいわきねんこうえん", "広島の平和記念公園"),
    ("", "とうきょうだいがくにごうかくした", "東京大学に合格した"),
    ("", "しんじゅくえきはにほんでいちばんおおきい", "新宿駅は日本で一番大きい"),
    # === カタカナ語 (71-80) ===
    ("", "こんぴゅーたーさいえんす", "コンピューターサイエンス"),
    ("", "いんたーねっとにせつぞくする", "インターネットに接続する"),
    ("", "すまーとふぉんをつかう", "スマートフォンを使う"),
    ("", "ぷろぐらみんぐげんご", "プログラミング言語"),
    ("", "でーたべーすのせっけい", "データベースの設計"),
    ("", "あるごりずむをかいはつする", "アルゴリズムを開発する"),
    ("", "そふとうぇあのあっぷでーと", "ソフトウェアのアップデート"),
    ("", "くらうどこんぴゅーてぃんぐ", "クラウドコンピューティング"),
    ("", "せきゅりてぃたいさく", "セキュリティ対策"),
    ("", "でじたるとらんすふぉーめーしょん", "デジタルトランスフォーメーション"),
    # === 長文 (81-90) ===
    ("", "にほんのでんとうてきなぶんかはせかいでひょうかされている", "日本の伝統的な文化は世界で評価されている"),
    ("", "じんこうちのうのはってんはめざましい", "人工知能の発展はめざましい"),
    ("", "ちきゅうおんだんかはしんこくなもんだいだ", "地球温暖化は深刻な問題だ"),
    ("", "しょうしこうれいかがしゃかいもんだいになっている", "少子高齢化が社会問題になっている"),
    ("", "さいせいかのうえねるぎーのふきゅうがすすんでいる", "再生可能エネルギーの普及が進んでいる"),
    ("", "けいざいのぐろーばるかがかそくしている", "経済のグローバル化が加速している"),
    ("", "きょういくのでじたるかがきゅうそくにすすんでいる", "教育のデジタル化が急速に進んでいる"),
    ("", "にほんのしょくぶんかはゆねすこにとうろくされた", "日本の食文化はユネスコに登録された"),
    ("", "とうきょうおりんぴっくはにせんにじゅうねんにかいさいされた", "東京オリンピックは二千二十年に開催された"),
    ("", "うちゅうたんさのぎじゅつがひやくてきにしんぽした", "宇宙探査の技術が飛躍的に進歩した"),
    # === 短文・日常 (91-100) ===
    ("", "おはようございます", "おはようございます"),
    ("", "ありがとうございます", "ありがとうございます"),
    ("", "すみません", "すみません"),
    ("", "おなかがすいた", "お腹が空いた"),
    ("", "つかれた", "疲れた"),
    ("", "いそがしい", "忙しい"),
    ("", "たのしかった", "楽しかった"),
    ("", "おいしいりょうり", "おいしい料理"),
    ("", "あついなつ", "暑い夏"),
    ("", "さむいふゆ", "寒い冬"),
]


def load_model(checkpoint_path: str, device: torch.device):
    collator = ARCollator()
    vocab_path = checkpoint_path.replace(".pt", "_vocab.json")
    collator.load_vocab(vocab_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    hidden = state["embed_tokens.weight"].shape[1]
    max_pos = state["embed_positions.weight"].shape[0]
    layer_keys = [
        k for k in state
        if k.startswith("transformer.layers.") and k.endswith(".self_attn.in_proj_weight")
    ]
    num_layers = len(layer_keys)

    model = SimpleGPT2(
        vocab_size=collator.vocab_size,
        hidden_size=hidden,
        num_layers=num_layers,
        num_heads=8,
        max_positions=max_pos,
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, collator, max_pos, ckpt["step"]


@torch.no_grad()
def generate(model, collator, prefix_ids, device, max_pos, max_new=100):
    prefix_ids = prefix_ids[-(max_pos - 2):]
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    out = []
    for _ in range(max_new):
        if ids.shape[1] >= max_pos:
            break
        mask = torch.ones_like(ids)
        logits = model(ids, mask)
        nid = logits[0, -1].argmax().item()
        if nid == collator.EOS or nid == collator.PAD:
            break
        out.append(nid)
        ids = torch.cat([ids, torch.tensor([[nid]], device=device)], dim=1)
    return collator.decode_ids(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/ar_baseline/best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, collator, max_pos, step = load_model(args.checkpoint, device)
    print(f"Model: step {step}, vocab {collator.vocab_size}, max_pos {max_pos}")
    print(f"Test cases: {len(TEST_CASES)}")
    print()

    perfect = 0
    partial = 0
    wrong = 0
    results = []

    for i, (ctx, reading, expected) in enumerate(TEST_CASES):
        ctx_ids = collator.encode_text(ctx[-40:]) if ctx else []
        read_ids = collator.encode_text(reading)
        prefix = ctx_ids + [collator.SEP] + read_ids + [collator.OUT]
        pred = generate(model, collator, prefix, device, max_pos)

        if pred == expected:
            mark = "○"
            perfect += 1
        elif pred in expected or expected in pred:
            # Partial: one contains the other or very close
            from src.eval.metrics import character_accuracy
            acc = character_accuracy(expected, pred)
            if acc >= 0.8:
                mark = "△"
                partial += 1
            else:
                mark = "×"
                wrong += 1
        else:
            from src.eval.metrics import character_accuracy
            acc = character_accuracy(expected, pred)
            if acc >= 0.8:
                mark = "△"
                partial += 1
            else:
                mark = "×"
                wrong += 1

        results.append((mark, ctx, reading, expected, pred))

    # Print results
    print(f"{'#':>3s} {'Mark':>4s}  {'Reading':<30s}  {'Expected':<30s}  {'Predicted':<30s}")
    print("-" * 105)
    for i, (mark, ctx, reading, expected, pred) in enumerate(results):
        ctx_str = f"[{ctx[:10]}…]" if ctx else ""
        r_display = f"{ctx_str}{reading}"[:30]
        print(f"{i+1:>3d} {mark:>4s}  {r_display:<30s}  {expected[:30]:<30s}  {pred[:30]:<30s}")

    # Summary
    print()
    print(f"=== Summary (step {step}) ===")
    print(f"  Perfect (○): {perfect}/{len(TEST_CASES)} ({perfect/len(TEST_CASES)*100:.1f}%)")
    print(f"  Partial (△): {partial}/{len(TEST_CASES)} ({partial/len(TEST_CASES)*100:.1f}%)")
    print(f"  Wrong   (×): {wrong}/{len(TEST_CASES)} ({wrong/len(TEST_CASES)*100:.1f}%)")

    # Print only failures
    failures = [(i, r) for i, r in enumerate(results) if r[0] == "×"]
    if failures:
        print(f"\n=== Failures ({len(failures)}) ===")
        for i, (mark, ctx, reading, expected, pred) in failures:
            print(f"  #{i+1}")
            print(f"    ctx:      \"{ctx}\"")
            print(f"    reading:  {reading}")
            print(f"    expected: {expected}")
            print(f"    got:      {pred}")
            print()


if __name__ == "__main__":
    main()
