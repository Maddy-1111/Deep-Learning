"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler
from dataset import get_dataloaders, SOS_IDX, EOS_IDX, PAD_IDX


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need" (§5.4).

    For each non-pad target token y:
        y_smooth = confidence · 1[y]  +  (smoothing / (V - 2)) · 1[≠ y, ≠ pad]
    Pad positions contribute zero loss.
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [N, V]   target: [N]
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            # Spread (smoothing) over V-2 non-{true,pad} classes.
            true_dist = torch.full_like(logits, self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0.0
            # Drop pad positions entirely from the loss.
            pad_mask = (target == self.pad_idx)
            true_dist[pad_mask] = 0.0

        loss = self.criterion(log_probs, true_dist)
        n_tokens = (~pad_mask).sum().clamp(min=1)
        return loss / n_tokens


# ══════════════════════════════════════════════════════════════════════
#  TRAIN / EVAL EPOCH
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    pad_idx: int = PAD_IDX,
    log_fn=None,
) -> float:
    """Run one epoch. Returns the average per-token loss across all batches."""
    if not hasattr(run_epoch, "global_step"):
        run_epoch.global_step = 0
    model.train(is_train)
    total_loss, n_batches = 0.0, 0

    pbar = tqdm(data_iter, desc=f"epoch {epoch_num} {'train' if is_train else 'eval'}")
    for src, tgt in pbar:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_in, pad_idx)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_in, src_mask, tgt_mask)            # [B, T-1, V]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)),
                           tgt_out.reshape(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # ABLATION 2.2 — Q/K grad norms (logged for the first 1k steps).
                if log_fn is not None and run_epoch.global_step < 1000:
                    layer0 = model.encoder.layers[0].self_attn
                    log_fn({
                        "grad_norm/W_Q": layer0.W_Q.weight.grad.norm().item(),
                        "grad_norm/W_K": layer0.W_K.weight.grad.norm().item(),
                        "step": run_epoch.global_step,
                    })
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                run_epoch.global_step += 1

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if log_fn is not None and is_train:
            log_fn({"train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"]})

    return total_loss / max(n_batches, 1)


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Token-by-token greedy decoding for a single source sentence (batch=1)."""
    model.eval()
    src, src_mask = src.to(device), src_mask.to(device)
    memory = model.encode(src, src_mask)

    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX)
        logits = model.decode(memory, src_mask, ys, tgt_mask)            # [1, t, V]
        next_token = logits[:, -1, :].argmax(-1, keepdim=True)           # [1, 1]
        ys = torch.cat([ys, next_token], dim=1)
        if next_token.item() == end_symbol:
            break
    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def _ids_to_tokens(ids: list[int], vocab, drop: set[int]) -> list[str]:
    out = []
    for i in ids:
        if i == EOS_IDX:
            break
        if i in drop:
            continue
        out.append(vocab.lookup_token(i))
    return out


def _corpus_bleu(hypotheses: list[list[str]], references: list[list[str]]) -> float:
    """Try sacrebleu first, fall back to nltk. Returns BLEU on a 0–100 scale."""
    try:
        import sacrebleu
        hyps = [" ".join(h) for h in hypotheses]
        refs = [[" ".join(r) for r in references]]
        return float(sacrebleu.corpus_bleu(hyps, refs).score)
    except ImportError:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs_wrapped = [[r] for r in references]
        return 100.0 * corpus_bleu(refs_wrapped, hypotheses,
                                    smoothing_function=SmoothingFunction().method1)


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """Corpus-level BLEU (0–100) via greedy decoding on the entire dataloader."""
    model.eval()
    drop = {SOS_IDX, PAD_IDX}
    hyps, refs = [], []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="bleu"):
            src, tgt = src.to(device), tgt.to(device)
            for i in range(src.size(0)):
                src_i = src[i:i + 1]
                src_mask = make_src_mask(src_i, PAD_IDX)
                pred = greedy_decode(model, src_i, src_mask, max_len,
                                     SOS_IDX, EOS_IDX, device=device)
                hyps.append(_ids_to_tokens(pred[0].tolist(), tgt_vocab, drop))
                refs.append(_ids_to_tokens(tgt[i].tolist(), tgt_vocab, drop))

    return _corpus_bleu(hyps, refs)


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def _extract_model_config(model: Transformer) -> dict:
    from model import PositionalEncoding
    layer = model.encoder.layers[0]
    return {
        "src_vocab_size": model.src_embed.num_embeddings,
        "tgt_vocab_size": model.tgt_embed.num_embeddings,
        "d_model":   model.d_model,
        "N":         len(model.encoder.layers),
        "num_heads": layer.self_attn.num_heads,
        "d_ff":      layer.ffn.linear1.out_features,
        "dropout":   layer.dropout.p,
        "pos_encoding": "sinusoidal" if isinstance(model.pos_enc, PositionalEncoding) else "learned",
        "attn_scale":   layer.self_attn.attn_scale,
    }


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config":         _extract_model_config(model),
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"]


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment(
    project: str = "da6401-a3",
    run_name: Optional[str] = None,
    config: Optional[dict] = None,
    use_wandb: bool = True,
) -> None:
    cfg = {
        "batch_size":   128,
        "d_model":      512,
        "N":            6,
        "num_heads":    8,
        "d_ff":         2048,
        "dropout":      0.1,
        "smoothing":    0.1,         # ABLATION 2.5: set 0.0 to disable
        "warmup_steps": 4000,
        "num_epochs":   20,
        "min_freq":     2,
        # Ablation toggles ─────────────────────────────────
        "use_noam":     True,        # ABLATION 2.1: False → fixed LR
        "fixed_lr":     1e-4,        # used when use_noam=False
        "attn_scale":   True,        # ABLATION 2.2: False → no 1/√dk
        "pos_encoding": "sinusoidal",# ABLATION 2.4: "learned" alternative
    }
    if config:
        cfg.update(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    if use_wandb:
        import wandb
        wandb.init(project=project, name=run_name, config=cfg)
        log_fn = wandb.log
    else:
        log_fn = None

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=cfg["batch_size"], min_freq=cfg["min_freq"],
    )
    print(f"|src| = {len(src_vocab)}   |tgt| = {len(tgt_vocab)}")

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg["d_model"], N=cfg["N"], num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"], dropout=cfg["dropout"],
        pos_encoding=cfg["pos_encoding"],
        attn_scale=cfg["attn_scale"],
    ).to(device)

    if cfg["use_noam"]:
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0,
                                     betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=cfg["d_model"],
                                  warmup_steps=cfg["warmup_steps"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["fixed_lr"],
                                     betas=(0.9, 0.98), eps=1e-9)
        scheduler = None
    loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=cfg["smoothing"])

    run_epoch.global_step = 0
    best_val = float("inf")
    for epoch in range(cfg["num_epochs"]):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                               epoch_num=epoch, is_train=True, device=device,
                               log_fn=log_fn)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None,
                             epoch_num=epoch, is_train=False, device=device)

        print(f"[epoch {epoch}] train={train_loss:.4f}  val={val_loss:.4f}")
        if log_fn is not None:
            log_fn({"epoch": epoch, "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, "checkpoint_best.pt")
        save_checkpoint(model, optimizer, scheduler, epoch, "checkpoint_last.pt")

    load_checkpoint("checkpoint_best.pt", model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"test BLEU = {bleu:.2f}")
    if log_fn is not None:
        log_fn({"test/bleu": bleu})


# ══════════════════════════════════════════════════════════════════════
#  ABLATION HELPERS
# ══════════════════════════════════════════════════════════════════════

ABLATIONS: dict[str, dict] = {
    # 2.1 Noam vs fixed LR
    "baseline":          {},
    "fixed_lr":          {"use_noam": False, "fixed_lr": 1e-4},
    # 2.2 Attention scaling
    "no_attn_scale":     {"attn_scale": False},
    # 2.4 Positional encoding
    "learned_pe":        {"pos_encoding": "learned"},
    # 2.5 Label smoothing
    "no_label_smooth":   {"smoothing": 0.0},
}


def run_all_ablations(num_epochs: int = 20, project: str = "da6401-a3") -> None:
    """Train every ablation in ABLATIONS sequentially, each as its own W&B run."""
    for name, overrides in ABLATIONS.items():
        cfg = {"num_epochs": num_epochs, **overrides}
        print(f"\n=== ablation: {name}  (cfg={cfg}) ===")
        run_training_experiment(project=project, run_name=name, config=cfg)


def get_encoder_attention(model: Transformer, layer_idx: int = -1) -> torch.Tensor:
    """Return attention weights [B, num_heads, S, S] from one encoder self-attn layer."""
    return model.encoder.layers[layer_idx].self_attn.last_attn_weights


if __name__ == "__main__":
    run_training_experiment()
