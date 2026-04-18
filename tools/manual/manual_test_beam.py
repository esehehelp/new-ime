"""Manual test with beam search for AR baseline.

Usage:
    uv run python -m scripts.manual_test_beam --checkpoint checkpoints/ar_v3_local/best.pt --beam 10
"""

from __future__ import annotations

import argparse
import sys

import torch

from models.src.data.dataset import ARCollator
from models.src.eval.metrics import character_accuracy
from models.src.training.train_ar import SimpleGPT2

sys.stdout.reconfigure(encoding="utf-8")

# Same test cases as manual_test.py
TEST_CASES = [
    ("", "きょうはいいてんきですね", "今日はいい天気ですね"),
    ("", "とうきょうとしぶやく", "東京都渋谷区"),
    ("", "がっこうにいく", "学校に行く"),
    ("", "しんぶんをよむ", "新聞を読む"),
    ("", "かんじへんかんのせいどをひょうかする", "漢字変換の精度を評価する"),
    ("", "にほんごにゅうりょくのしくみ", "日本語入力の仕組み"),
    ("彼は記者として", "きしゃにのってしゅっちょうした", "汽車に乗って出張した"),
    ("", "こうしょうがなんこうしている", "交渉が難航している"),
    ("今日は天気が良いので、", "さんぽにいきましょう", "散歩に行きましょう"),
    ("", "やまにのぼる", "山に登る"),
    ("", "ほんをかう", "本を買う"),
    ("", "えきまであるく", "駅まで歩く"),
    ("", "ふじさんにのぼる", "富士山に登る"),
    ("", "つかれた", "疲れた"),
    ("", "いそがしい", "忙しい"),
    ("", "たのしかった", "楽しかった"),
    ("", "あついなつ", "暑い夏"),
    ("", "さむいふゆ", "寒い冬"),
    ("", "おなかがすいた", "お腹が空いた"),
    ("", "おいしいりょうり", "おいしい料理"),
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

    model = SimpleGPT2(
        vocab_size=collator.vocab_size,
        hidden_size=hidden,
        num_layers=len(layer_keys),
        num_heads=8,
        max_positions=max_pos,
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, collator, max_pos, ckpt["step"]


@torch.no_grad()
def greedy_generate(model, prefix_ids, device, max_pos, collator, max_new=80):
    prefix_ids = prefix_ids[-(max_pos - 2):]
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    out = []
    for _ in range(max_new):
        if ids.shape[1] >= max_pos:
            break
        logits = model(ids, torch.ones_like(ids))
        nid = logits[0, -1].argmax().item()
        if nid == collator.EOS or nid == collator.PAD:
            break
        out.append(nid)
        ids = torch.cat([ids, torch.tensor([[nid]], device=device)], dim=1)
    return collator.decode_ids(out)


@torch.no_grad()
def beam_generate(
    model, prefix_ids, device, max_pos, collator,
    beam_width=10, max_new=80, length_penalty=0.6, repetition_penalty=1.2,
):
    """Beam search with length normalization and repetition penalty."""
    prefix_ids = prefix_ids[-(max_pos - 2):]

    # Each beam: (token_ids, log_prob, finished)
    beams = [(prefix_ids[:], 0.0)]
    finished = []

    for step in range(max_new):
        candidates = []
        for seq, score in beams:
            if len(seq) >= max_pos:
                finished.append((seq, score))
                continue

            ids = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(ids, torch.ones_like(ids))
            log_probs = torch.log_softmax(logits[0, -1], dim=0)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                seen_tokens = set(seq[len(prefix_ids):])
                for tok in seen_tokens:
                    if tok < log_probs.shape[0]:
                        log_probs[tok] /= repetition_penalty

            top_k = torch.topk(log_probs, min(beam_width * 2, log_probs.shape[0]))
            for i in range(min(beam_width * 2, top_k.indices.shape[0])):
                tok = top_k.indices[i].item()
                lp = top_k.values[i].item()
                if tok == collator.EOS or tok == collator.PAD:
                    finished.append((seq, score + lp))
                else:
                    candidates.append((seq + [tok], score + lp))

        if not candidates:
            break

        # Length-normalized scoring for beam pruning
        def norm_score(item):
            s, sc = item
            gen_len = max(len(s) - len(prefix_ids), 1)
            return sc / (gen_len ** length_penalty)

        candidates.sort(key=norm_score, reverse=True)
        beams = [(s, sc) for s, sc in candidates[:beam_width]]

    # Combine finished and remaining beams
    all_results = finished + [(s, sc) for s, sc in beams]

    def final_score(item):
        s, sc = item
        gen_len = max(len(s) - len(prefix_ids), 1)
        return sc / (gen_len ** length_penalty)

    all_results.sort(key=final_score, reverse=True)

    prefix_len = len(prefix_ids)
    results = []
    for seq, score in all_results[:beam_width]:
        generated = seq[prefix_len:]
        text = collator.decode_ids(generated)
        if text:
            results.append(text)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/ar_v3_local/best.pt")
    parser.add_argument("--beam", type=int, default=10)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, collator, max_pos, step = load_model(args.checkpoint, device)
    print(f"Model: step {step}, beam={args.beam}, "
          f"len_pen={args.length_penalty}, rep_pen={args.repetition_penalty}")
    print()

    print(f"{'#':>2s} {'Reading':20s} {'Expected':15s} "
          f"{'Greedy':15s} {'Beam top1':15s} {'InBeam':>6s}")
    print("-" * 80)

    greedy_ok = 0
    beam_ok = 0
    beam_in = 0

    for i, (ctx, reading, expected) in enumerate(TEST_CASES):
        ctx_ids = collator.encode_text(ctx[-40:]) if ctx else []
        read_ids = collator.encode_text(reading)
        prefix = ctx_ids + [collator.SEP] + read_ids + [collator.OUT]

        g = greedy_generate(model, prefix, device, max_pos, collator,
                            max_new=len(reading) + 15)
        b = beam_generate(model, prefix, device, max_pos, collator,
                          beam_width=args.beam, max_new=len(reading) + 15,
                          length_penalty=args.length_penalty,
                          repetition_penalty=args.repetition_penalty)

        g_ok = "○" if g == expected else "×"
        b_ok = "○" if b and b[0] == expected else "×"
        in_beam = "○" if expected in b else "×"

        if g == expected:
            greedy_ok += 1
        if b and b[0] == expected:
            beam_ok += 1
        if expected in b:
            beam_in += 1

        print(f"{i+1:>2d} {reading[:20]:20s} {expected[:15]:15s} "
              f"{g_ok}{g[:14]:14s} {b_ok}{(b[0] if b else ''):14s} {in_beam:>6s}")

    print()
    n = len(TEST_CASES)
    print(f"Greedy correct: {greedy_ok}/{n}")
    print(f"Beam top1 correct: {beam_ok}/{n}")
    print(f"In beam (any position): {beam_in}/{n}")


if __name__ == "__main__":
    main()
