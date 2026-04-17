"""Phase 2: Autoregressive baseline training.

GPT-2 style decoder-only causal LM for kana-kanji conversion.
This serves as:
  1. Baseline to beat with CTC-NAT
  2. Knowledge distillation teacher for CTC-NAT training

Usage:
    uv run python -m src.training.train_ar \
        --train datasets/eval/train.jsonl \
        --dev datasets/eval/dev.jsonl \
        --output checkpoints/ar_baseline \
        --epochs 3 \
        --batch-size 16 \
        --max-samples 1000000

Checkpoint format: saves model, optimizer, scheduler, step, vocab every N steps.
Resume: pass --resume checkpoints/ar_baseline/checkpoint_step_XXXX.pt
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import ARCollator, KanaKanjiDataset
from src.eval.metrics import EvalResult


class SimpleGPT2(nn.Module):
    """Minimal GPT-2 style decoder-only transformer.

    No pretrained weights — trained from scratch on kana-kanji data.
    For Phase 2 baseline only. Phase 3 uses encoder-decoder CTC-NAT.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        max_positions: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = nn.Embedding(max_positions, hidden_size)
        self.drop = nn.Dropout(dropout)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) 1=valid, 0=pad

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)
        x = self.drop(x)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

        # Padding mask (True = ignore)
        pad_mask = None
        if attention_mask is not None:
            pad_mask = ~attention_mask.bool()

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def evaluate(model, dataloader, device, collator, max_batches=50):
    """Quick evaluation on dev set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    eval_result = EvalResult()

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)

            # Shift for causal LM: predict next token
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            valid_tokens = (shift_labels != -100).sum().item()
            total_loss += loss.item()
            total_tokens += valid_tokens

            # Greedy decode for CharAcc
            predictions = logits.argmax(dim=-1)
            for b in range(input_ids.shape[0]):
                # Find OUT token position
                seq = input_ids[b].tolist()
                if collator.OUT in seq:
                    out_pos = seq.index(collator.OUT)
                    # Predicted surface: tokens after OUT until EOS or PAD
                    pred_ids = []
                    for t in range(out_pos + 1, len(seq)):
                        pid = predictions[b, t].item()
                        if pid == collator.EOS or pid == collator.PAD:
                            break
                        pred_ids.append(pid)
                    pred_text = collator.decode_ids(pred_ids)

                    # Reference surface
                    ref_ids = []
                    lab = labels[b].tolist()
                    for t in range(len(lab)):
                        if lab[t] != -100 and lab[t] != collator.EOS:
                            ref_ids.append(lab[t])
                    ref_text = collator.decode_ids(ref_ids)

                    eval_result.add(ref_text, [pred_text])

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
    return {
        "loss": avg_loss,
        "perplexity": ppl,
        **eval_result.summary(),
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    epoch: int,
    collator: ARCollator,
    best_loss: float,
):
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "epoch": epoch,
        "vocab_size": collator.vocab_size,
        "best_loss": best_loss,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    # Save vocab alongside
    vocab_path = path.replace(".pt", "_vocab.json")
    collator.save_vocab(vocab_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
):
    """Load training checkpoint. Returns (step, epoch, best_loss)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["step"], checkpoint["epoch"], checkpoint.get("best_loss", float("inf"))


def main():
    parser = argparse.ArgumentParser(description="Train AR baseline")
    parser.add_argument("--train", required=True, help="Training JSONL")
    parser.add_argument("--dev", required=True, help="Dev JSONL")
    parser.add_argument("--output", default="checkpoints/ar_baseline", help="Output directory")
    parser.add_argument("--resume", default="", help="Resume from checkpoint path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit training samples (0=all)")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--checkpoint-every", type=int, default=5000, help="Save every N steps")
    parser.add_argument("--eval-every", type=int, default=2000, help="Evaluate every N steps")
    parser.add_argument("--log-every", type=int, default=100, help="Log every N steps")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Data
    print(f"\nLoading training data: {args.train}")
    train_dataset = KanaKanjiDataset(args.train, max_samples=args.max_samples)
    print(f"Training samples: {len(train_dataset):,}")

    dev_dataset = KanaKanjiDataset(args.dev)
    print(f"Dev samples: {len(dev_dataset):,}")

    collator = ARCollator(max_seq_len=args.max_seq_len)

    # Pre-scan to build vocab
    print("Building vocabulary...")
    for sample in train_dataset.data:
        collator.encode_text(sample.get("context", ""))
        collator.encode_text(sample["reading"])
        collator.encode_text(sample["surface"])
    for sample in dev_dataset.data:
        collator.encode_text(sample.get("context", ""))
        collator.encode_text(sample["reading"])
        collator.encode_text(sample["surface"])
    print(f"Vocabulary size: {collator.vocab_size}")

    # Resume vocab if needed
    if args.resume:
        vocab_path = args.resume.replace(".pt", "_vocab.json")
        if Path(vocab_path).exists():
            collator.load_vocab(vocab_path)
            print(f"Loaded vocab from {vocab_path}: {collator.vocab_size} tokens")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # Model
    model = SimpleGPT2(
        vocab_size=collator.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_positions=args.max_seq_len,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {param_count / 1e6:.1f}M parameters")
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Scheduler: linear warmup + cosine decay
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    start_step = 0
    start_epoch = 0
    best_loss = float("inf")
    if args.resume and Path(args.resume).exists():
        print(f"\nResuming from {args.resume}")
        start_step, start_epoch, best_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        print(f"Resumed at step {start_step}, epoch {start_epoch}, best_loss {best_loss:.4f}")

    # Loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        autocast_ctx = torch.amp.autocast("cpu")

    # Training loop
    print(f"\nTraining for {args.epochs} epochs, {total_steps} total steps")
    effective_batch = args.batch_size * args.grad_accum
    print(f"Batch: {args.batch_size} x {args.grad_accum} accum = {effective_batch} effective")
    print(f"Checkpoints: every {args.checkpoint_every} steps → {args.output}/")
    print()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = start_step
    running_loss = 0.0
    log_steps = 0
    t_start = time.time()

    model.train()
    optimizer.zero_grad()

    for epoch in range(start_epoch, args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast_ctx:
                logits = model(input_ids, attention_mask)
                # Shift for causal LM
                shift_logits = logits[:, :-1].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                loss = loss / args.grad_accum

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * args.grad_accum
            log_steps += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log
                if global_step % args.log_every == 0:
                    avg = running_loss / log_steps
                    elapsed = time.time() - t_start
                    lr = scheduler.get_last_lr()[0]
                    if device.type == "cuda":
                        mem = torch.cuda.memory_allocated() / 1024**3
                        print(
                            f"step {global_step:>6d} | loss {avg:.4f} | "
                            f"lr {lr:.2e} | {elapsed:.0f}s | "
                            f"GPU {mem:.1f}GB",
                            flush=True,
                        )
                    else:
                        print(
                            f"step {global_step:>6d} | loss {avg:.4f} | "
                            f"lr {lr:.2e} | {elapsed:.0f}s",
                            flush=True,
                        )
                    running_loss = 0.0
                    log_steps = 0

                # Eval
                if global_step % args.eval_every == 0:
                    eval_results = evaluate(model, dev_loader, device, collator)
                    print(
                        f"  [eval] loss={eval_results['loss']:.4f} "
                        f"ppl={eval_results['perplexity']:.1f} "
                        f"CharAcc={eval_results.get('char_acc_top1', 0):.4f}",
                        flush=True,
                    )
                    if eval_results["loss"] < best_loss:
                        best_loss = eval_results["loss"]
                        best_path = str(output_dir / "best.pt")
                        save_checkpoint(
                            best_path, model, optimizer, scheduler,
                            global_step, epoch, collator, best_loss,
                        )
                        print(f"  [best] saved to {best_path}", flush=True)
                    model.train()

                # Checkpoint
                if global_step % args.checkpoint_every == 0:
                    ckpt_path = str(output_dir / f"checkpoint_step_{global_step}.pt")
                    save_checkpoint(
                        ckpt_path, model, optimizer, scheduler,
                        global_step, epoch, collator, best_loss,
                    )
                    print(f"  [ckpt] saved to {ckpt_path}", flush=True)

        print(f"\n=== Epoch {epoch + 1}/{args.epochs} complete ===\n", flush=True)

    # Final checkpoint
    final_path = str(output_dir / "final.pt")
    save_checkpoint(
        final_path, model, optimizer, scheduler,
        global_step, args.epochs, collator, best_loss,
    )
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    print(f"Best dev loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
