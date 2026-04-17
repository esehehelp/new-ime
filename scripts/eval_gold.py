"""Evaluate models on gold_1k.jsonl dataset.

Usage:
    uv run python -m scripts.eval_gold --checkpoint checkpoints/ar_v3_vast/checkpoint_step_70000.pt
    uv run python -m scripts.eval_gold --zenz references/zenz-v2.5-xsmall
    uv run python -m scripts.eval_gold --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8")

GOLD_PATH = "datasets/gold_1k.jsonl"


def load_gold():
    data = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# --- AR model backend ---

def load_ar_model(checkpoint_path, device="cpu"):
    from src.data.dataset import ARCollator
    from src.training.train_ar import SimpleGPT2

    collator = ARCollator()
    vocab_path = checkpoint_path.replace(".pt", "_vocab.json")
    collator.load_vocab(vocab_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    hidden = state["embed_tokens.weight"].shape[1]
    max_pos = state["embed_positions.weight"].shape[0]
    n_layers = len([k for k in state if k.startswith("transformer.layers.") and k.endswith(".self_attn.in_proj_weight")])

    model = SimpleGPT2(
        vocab_size=collator.vocab_size, hidden_size=hidden,
        num_layers=n_layers, num_heads=8, max_positions=max_pos,
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, collator, max_pos, ckpt["step"]


@torch.no_grad()
def ar_generate(model, collator, prefix_ids, device, max_pos, max_new=80):
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


def eval_ar(checkpoint_path, gold_data, device="cpu"):
    model, collator, max_pos, step = load_ar_model(checkpoint_path, device)
    results = []
    for item in gold_data:
        ctx = item.get("context", "")[-40:]
        reading = item["reading"]
        ctx_ids = collator.encode_text(ctx) if ctx else []
        read_ids = collator.encode_text(reading)
        prefix = ctx_ids + [collator.SEP] + read_ids + [collator.OUT]

        t0 = time.perf_counter()
        pred = ar_generate(model, collator, prefix, device, max_pos, max_new=len(reading) + 20)
        t1 = time.perf_counter()
        results.append({"pred": pred, "time_ms": (t1 - t0) * 1000})
    return results, f"AR step {step}"


# --- Zenz v2.5 backend ---

def eval_zenz(model_dir, gold_data, device="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()

    # Zenz prompt format: [input_tag]reading[context_tag]context[output_tag]
    INPUT_TAG = "\uEE00"
    OUTPUT_TAG = "\uEE01"
    CONTEXT_TAG = "\uEE02"

    def to_katakana(text):
        return "".join(
            chr(ord(c) + 0x60) if "\u3041" <= c <= "\u3096" else c
            for c in text
        )

    results = []
    for item in gold_data:
        ctx = item.get("context", "")[-40:]
        reading = item["reading"]
        kata = to_katakana(reading)

        prompt = f"{INPUT_TAG}{kata}{CONTEXT_TAG}{ctx}{OUTPUT_TAG}"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=len(reading) + 20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        t1 = time.perf_counter()

        generated = output[0][input_ids.shape[1]:]
        pred = tokenizer.decode(generated, skip_special_tokens=True)
        results.append({"pred": pred, "time_ms": (t1 - t0) * 1000})

    model_name = Path(model_dir).name
    return results, f"zenz {model_name}"


# --- Evaluation ---

def evaluate(gold_data, predictions):
    from src.eval.metrics import character_accuracy

    exact = 0
    char_acc_sum = 0.0
    total_time = 0.0

    for item, pred_info in zip(gold_data, predictions):
        expected = item["surface"]
        pred = pred_info["pred"]
        total_time += pred_info["time_ms"]

        if pred == expected:
            exact += 1
        char_acc_sum += character_accuracy(expected, pred)

    n = len(gold_data)
    times = sorted([p["time_ms"] for p in predictions])

    return {
        "total": n,
        "exact_match": exact / n,
        "char_acc": char_acc_sum / n,
        "p50_ms": times[n // 2] if n else 0,
        "p95_ms": times[int(n * 0.95)] if n else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="", help="AR checkpoint path")
    parser.add_argument("--zenz", default="", help="Zenz model directory")
    parser.add_argument("--all", action="store_true", help="Run all available models")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="Limit gold entries (0=all)")
    args = parser.parse_args()

    gold = load_gold()
    if args.limit:
        gold = gold[:args.limit]
    print(f"Gold dataset: {len(gold)} entries")

    models_to_run = []

    if args.all:
        # AR models
        for ckpt in [
            "checkpoints/ar_v3_vast/checkpoint_step_70000.pt",
            "checkpoints/ar_v3_local/best.pt",
            "checkpoints/ar_v3_chunks/best.pt",
        ]:
            if Path(ckpt).exists():
                models_to_run.append(("ar", ckpt))

        # Zenz models
        for zdir in [
            "references/zenz-v2.5-xsmall",
            "references/zenz-v2.5-medium",
        ]:
            if Path(zdir).exists() and (Path(zdir) / "config.json").exists():
                models_to_run.append(("zenz", zdir))
    elif args.checkpoint:
        models_to_run.append(("ar", args.checkpoint))
    elif args.zenz:
        models_to_run.append(("zenz", args.zenz))

    if not models_to_run:
        print("No models specified. Use --checkpoint, --zenz, or --all")
        return

    print(f"\nModels to evaluate: {len(models_to_run)}")
    all_results = {}

    for model_type, model_path in models_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_path}")

        if model_type == "ar":
            preds, name = eval_ar(model_path, gold, args.device)
        else:
            preds, name = eval_zenz(model_path, gold, args.device)

        metrics = evaluate(gold, preds)
        all_results[name] = metrics

        print(f"  {name}:")
        print(f"    EM={metrics['exact_match']:.3f}  CharAcc={metrics['char_acc']:.3f}  "
              f"p50={metrics['p50_ms']:.0f}ms  p95={metrics['p95_ms']:.0f}ms")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Model':35s} {'EM':>6s} {'CharAcc':>8s} {'p50ms':>7s}")
    print("-" * 60)
    for name, m in sorted(all_results.items(), key=lambda x: -x[1]["exact_match"]):
        print(f"{name:35s} {m['exact_match']:>6.3f} {m['char_acc']:>8.3f} {m['p50_ms']:>7.0f}")


if __name__ == "__main__":
    main()
