"""Teacher (Transformer encoder + AR decoder) 学習スクリプト。

CTC student (ctc_nat_30m / 90m) と異なる inductive bias を持つ teacher を
scratch で学習する。KD で使い回される前提。

Design:
- AR NLL loss (decoder cross-entropy)
- teacher forcing
- shared SharedCharTokenizer (vocab 4801)
- cosine LR with warmup
- checkpointing, resume-first

Usage (vast.ai 5090):
    python -m models.src.training.train_teacher \
        --train datasets/mixes/train_teacher_v1_20m.jsonl \
        --dev datasets/eval/eval_v3/dev.jsonl \
        --preset teacher_150m \
        --tokenizer-path models/checkpoints/ctc_nat_90m/checkpoint_step_27500_tokenizer.json \
        --batch-size 32 --grad-accum 4 --max-seq-len 192 \
        --fp16 --num-workers 4 \
        --max-steps 200000 --lr 2e-4 --warmup-steps 2000 \
        --lr-schedule cosine \
        --checkpoint-every 5000 --eval-every 2000 \
        --output models/checkpoints/teacher_v1_150m
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.src.data.dataset import KanaKanjiDataset
from models.src.data.tokenizer import (
    BLANK_ID,
    CLS_ID,
    PAD_ID,
    SEP_ID,
    SharedCharTokenizer,
)
from models.src.model.teacher_seq2seq import TEACHER_PRESETS, TeacherSeq2Seq


class TeacherCollator:
    """Build encoder+decoder inputs + labels with teacher forcing.

    Layout:
        input_ids:         [CLS][ctx][SEP][reading]              (pad to max_enc)
        attention_mask:    1 for real, 0 for pad
        decoder_input_ids: [BOS=CLS][surface[:-1]]               (pad to max_dec)
        decoder_attn_mask: 1 for real, 0 for pad
        labels:            [surface tokens][EOS=SEP]             (-100 for pad)
    """

    def __init__(
        self,
        tokenizer: SharedCharTokenizer,
        max_seq_len: int = 192,
        max_context: int = 40,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_context = max_context

    def _encode_input(self, context: str, reading: str) -> list[int]:
        ctx = context[-self.max_context :] if context else ""
        ids = self.tokenizer.encode_with_special(ctx, reading)
        return ids[: self.max_seq_len]

    def _encode_target(self, surface: str) -> list[int]:
        ids = self.tokenizer.encode(surface)
        # Strip blanks to keep target clean
        ids = [t for t in ids if t != BLANK_ID]
        # Teacher decoder: BOS + surface -> surface + EOS
        return ids[: self.max_seq_len - 1]  # reserve 1 slot for EOS

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        enc_inputs: list[list[int]] = []
        dec_inputs: list[list[int]] = []
        labels_list: list[list[int]] = []

        for sample in batch:
            ctx = sample.get("context", "") or ""
            reading = sample["reading"]
            surface = sample["surface"]
            enc = self._encode_input(ctx, reading)
            tgt = self._encode_target(surface)
            if not tgt:
                continue
            dec_input = [CLS_ID] + tgt
            label = tgt + [SEP_ID]
            enc_inputs.append(enc)
            dec_inputs.append(dec_input)
            labels_list.append(label)

        if not enc_inputs:
            # Fallback: empty batch safety
            return {
                "input_ids": torch.zeros((0, 1), dtype=torch.long),
                "attention_mask": torch.zeros((0, 1), dtype=torch.long),
                "decoder_input_ids": torch.zeros((0, 1), dtype=torch.long),
                "decoder_attention_mask": torch.zeros((0, 1), dtype=torch.long),
                "labels": torch.zeros((0, 1), dtype=torch.long),
            }

        max_enc = max(len(x) for x in enc_inputs)
        max_dec = max(len(x) for x in dec_inputs)

        enc_padded = []
        enc_mask = []
        dec_padded = []
        dec_mask = []
        labels_padded = []
        for enc, dec, lab in zip(enc_inputs, dec_inputs, labels_list, strict=True):
            enc_pad = max_enc - len(enc)
            dec_pad = max_dec - len(dec)
            enc_padded.append(enc + [PAD_ID] * enc_pad)
            enc_mask.append([1] * len(enc) + [0] * enc_pad)
            dec_padded.append(dec + [PAD_ID] * dec_pad)
            dec_mask.append([1] * len(dec) + [0] * dec_pad)
            labels_padded.append(lab + [-100] * dec_pad)

        return {
            "input_ids": torch.tensor(enc_padded, dtype=torch.long),
            "attention_mask": torch.tensor(enc_mask, dtype=torch.long),
            "decoder_input_ids": torch.tensor(dec_padded, dtype=torch.long),
            "decoder_attention_mask": torch.tensor(dec_mask, dtype=torch.long),
            "labels": torch.tensor(labels_padded, dtype=torch.long),
        }


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def build_tokenizer(args) -> SharedCharTokenizer:
    if args.tokenizer_path:
        return SharedCharTokenizer.load(args.tokenizer_path)
    raise ValueError("--tokenizer-path is required for teacher training")


def build_model(args, vocab_size: int) -> TeacherSeq2Seq:
    return TeacherSeq2Seq.from_preset(args.preset, vocab_size=vocab_size)


def make_dataloader(
    jsonl_path: str,
    collator: TeacherCollator,
    batch_size: int,
    max_samples: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = KanaKanjiDataset(
        jsonl_path, max_samples=max_samples, max_seq_len=collator.max_seq_len
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def make_lr_lambda(warmup_steps: int, max_steps: int, schedule: str):
    def lr_lambda(step: int) -> float:
        warmup = max(warmup_steps, 1)
        if step < warmup:
            return (step + 1) / warmup
        if schedule == "cosine":
            progress = (step - warmup) / max(max_steps - warmup, 1)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0
    return lr_lambda


@torch.no_grad()
def evaluate(model: TeacherSeq2Seq, dev_loader: DataLoader, device, max_batches: int = 20):
    model.eval()
    losses = []
    token_count = 0
    correct = 0
    for i, batch in enumerate(dev_loader):
        if i >= max_batches:
            break
        batch = move_to_device(batch, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )
        losses.append(float(out["loss"].item()))
        preds = out["logits"].argmax(dim=-1)
        labels = batch["labels"]
        mask = labels != -100
        correct += int((preds[mask] == labels[mask]).sum().item())
        token_count += int(mask.sum().item())
    model.train()
    mean_loss = sum(losses) / max(len(losses), 1)
    token_acc = correct / max(token_count, 1)
    return {"loss": mean_loss, "token_acc": token_acc}


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int, tokenizer_path: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "preset": model.__class__.__name__,
    }
    torch.save(ckpt, path)
    # Also copy tokenizer sidecar (if teacher ckpt is reused by backends that
    # expect a *_tokenizer.json next to the .pt, e.g. CTCTeacher).
    sidecar = Path(str(path).replace(".pt", "_tokenizer.json"))
    if Path(tokenizer_path).exists():
        import shutil
        shutil.copy(tokenizer_path, sidecar)


def train_teacher(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(args)
    model = build_model(args, vocab_size=tokenizer.vocab_size).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"teacher params: {num_params/1_000_000:.2f}M", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_lr_lambda(args.warmup_steps, args.max_steps, args.lr_schedule)
    )

    collator = TeacherCollator(tokenizer, max_seq_len=args.max_seq_len, max_context=args.max_context)

    train_loader = make_dataloader(
        args.train, collator, args.batch_size,
        max_samples=args.max_train_samples,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        shuffle=True,
    )
    dev_loader = make_dataloader(
        args.dev, collator, args.batch_size,
        max_samples=args.max_dev_samples,
        num_workers=0, pin_memory=False, shuffle=False,
    )

    scaler = torch.amp.GradScaler("cuda") if args.fp16 and device.type == "cuda" else None
    use_amp = args.fp16 and device.type == "cuda"

    step = 0
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        step = int(ckpt.get("step", 0))
        print(f"Resumed from {args.resume} @ step {step}", flush=True)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_window = deque(maxlen=args.loss_window)
    t_start = time.perf_counter()
    t_last = t_start
    last_step = step
    model.train()

    epoch = start_epoch
    done = False
    while not done:
        for batch_idx, batch in enumerate(train_loader):
            if step >= args.max_steps:
                done = True
                break
            batch = move_to_device(batch, device)

            with torch.amp.autocast(
                device_type=device.type, enabled=use_amp, dtype=torch.float16
            ):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                    labels=batch["labels"],
                )
                loss = out["loss"] / args.grad_accum

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_boundary = ((batch_idx + 1) % args.grad_accum) == 0
            if is_boundary:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                loss_window.append(float(out["loss"].item()))

                if step % args.log_every == 0 and loss_window:
                    now = time.perf_counter()
                    rate = (step - last_step) / max(now - t_last, 1e-6)
                    lr_now = scheduler.get_last_lr()[0]
                    avg = sum(loss_window) / len(loss_window)
                    print(
                        f"[step {step}] loss={out['loss'].item():.4f} "
                        f"avg{len(loss_window)}={avg:.4f} lr={lr_now:.6f} "
                        f"rate={rate:.2f} steps/s",
                        flush=True,
                    )
                    t_last = now
                    last_step = step

                if step % args.eval_every == 0:
                    eval_stats = evaluate(model, dev_loader, device)
                    print(
                        f"[eval {step}] loss={eval_stats['loss']:.4f} "
                        f"token_acc={eval_stats['token_acc']:.4f}",
                        flush=True,
                    )

                if step % args.checkpoint_every == 0:
                    path = out_dir / f"checkpoint_step_{step}.pt"
                    save_checkpoint(path, model, optimizer, scheduler, step, args.tokenizer_path)
                    print(f"[ckpt] saved {path}", flush=True)

                if step >= args.max_steps:
                    done = True
                    break
        epoch += 1
        if args.epochs and epoch >= args.epochs:
            break

    final = out_dir / "final.pt"
    save_checkpoint(final, model, optimizer, scheduler, step, args.tokenizer_path)
    print(f"[done] final saved to {final}, total step {step}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--preset", default="teacher_150m", choices=list(TEACHER_PRESETS.keys()))
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", default="")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=192)
    p.add_argument("--max-context", type=int, default=40)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-dev-samples", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--max-steps", type=int, default=200000)
    p.add_argument("--epochs", type=int, default=99)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--lr-schedule", choices=["flat", "cosine"], default="cosine")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--loss-window", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    torch.manual_seed(args.seed)
    train_teacher(args)


if __name__ == "__main__":
    main()
