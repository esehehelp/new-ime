"""Dev-loader evaluation: loss, EM1, blank fraction.

Uses `model.greedy_decode(input_ids, attention_mask)` for the EM1 path,
which keeps loop / checkpoint / optimizer fully arch-agnostic while letting
each architecture decide how to convert encoder output into text. CTCNAT
implements it via CTC collapse; AR/DAT models would implement their own.

`evaluate_probe_em1` runs a separate, structured probe set (BenchItem list
loaded via `eval/loaders.py:load_bench`) at a much coarser cadence than
dev-loss evaluation. Probe accuracy is the main best.pt selection signal
for CTC-NAT; pre-v2 anchored at probe EM1 ≥ 0.59.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch


def _move(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


@torch.no_grad()
def evaluate_model(
    *,
    model: torch.nn.Module,
    loader: Iterable[dict],
    device: torch.device,
    tokenizer: Any,
    max_batches: int = 0,
) -> dict:
    """Compute dev loss, EM1, char-level acc, and blank fraction.

    Returns a dict with keys: loss, exact_match_top1, char_acc_top1,
    blank_fraction, num_samples. char_acc_top1 uses the same metric as
    eval/runner.py.
    """
    from new_ime.eval.metrics import character_accuracy

    was_training = model.training
    model.eval()
    losses: list[float] = []
    em1_count = 0
    char_acc_sum = 0.0
    total = 0
    blank_count = 0
    blank_total = 0

    blank_id = getattr(model, "blank_id", None)
    has_blank = blank_id is not None
    has_greedy = hasattr(model, "greedy_decode")

    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        batch = _move(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target_ids=batch["target_ids"],
            target_lengths=batch["target_lengths"],
        )
        if "loss" in outputs:
            losses.append(float(outputs["loss"].item()))
        if has_blank and "logits" in outputs:
            argmax = outputs["logits"].argmax(dim=-1)
            valid = batch["attention_mask"].bool()
            # Some archs (DAT) emit logits over an upsampled length; the
            # CTC-NAT-style blank fraction only makes sense when the time
            # axis matches the source attention mask. Skip cleanly otherwise.
            if argmax.shape == valid.shape:
                blank_count += int(((argmax == blank_id) & valid).sum().item())
                blank_total += int(valid.sum().item())

        if has_greedy:
            decoded = model.greedy_decode(
                batch["input_ids"], batch["attention_mask"]
            )
            for b, ids in enumerate(decoded):
                pred = tokenizer.decode(ids)
                tlen = int(batch["target_lengths"][b].item())
                ref_ids = batch["target_ids"][b, :tlen].tolist()
                ref = tokenizer.decode(ref_ids)
                if pred == ref:
                    em1_count += 1
                char_acc_sum += character_accuracy(ref, pred)
                total += 1

    if was_training:
        model.train()

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "exact_match_top1": em1_count / max(total, 1),
        "char_acc_top1": char_acc_sum / max(total, 1),
        "blank_fraction": (
            blank_count / blank_total if blank_total > 0 else 0.0
        ),
        "num_samples": total,
    }


@torch.no_grad()
def evaluate_probe_em1(
    *,
    model: torch.nn.Module,
    probe_items: Sequence,
    tokenizer: Any,
    device: torch.device,
    max_seq_len: int = 128,
    max_context: int = 32,
    limit: int = 0,
) -> dict:
    """Greedy-decode each probe item and compute EM1 against its references.

    Probe items are `BenchItem` instances from `eval/loaders.py:load_bench`,
    with `reading`, `context`, `references` (list[str]) attributes.
    """
    from collections import defaultdict

    was_training = model.training
    model.eval()
    matches = 0
    total = 0
    cat_correct: dict[str, int] = defaultdict(int)
    cat_total: dict[str, int] = defaultdict(int)
    for i, item in enumerate(probe_items):
        if limit > 0 and i >= limit:
            break
        ids = tokenizer.encode_with_special(
            (item.context or "")[-max_context:], item.reading
        )
        ids = ids[:max_seq_len]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        decoded = model.greedy_decode(input_ids, attention_mask)
        pred = tokenizer.decode(decoded[0])
        is_match = pred in item.references
        if is_match:
            matches += 1
        total += 1
        cat = getattr(item, "category", None)
        if cat is not None:
            cat_total[cat] += 1
            if is_match:
                cat_correct[cat] += 1

    if was_training:
        model.train()

    result: dict = {
        "em1": matches / max(total, 1),
        "n": total,
    }
    if cat_total:
        result["categories"] = {
            c: {
                "n": cat_total[c],
                "em1": cat_correct[c] / max(cat_total[c], 1),
            }
            for c in sorted(cat_total)
        }
    return result
