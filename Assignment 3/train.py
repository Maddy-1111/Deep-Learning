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
    total_loss = 0.0
    total_correct = 0     # § 2.1: token accuracy
    total_conf = 0.0      # § 2.5: prediction confidence
    total_tokens = 0
    n_batches = 0

    pbar = tqdm(data_iter, desc=f"epoch {epoch_num} {'train' if is_train else 'eval'}",
                mininterval=60, miniters=200, leave=False, ncols=80)
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
                # ABLATION 2.2 — Q/K grad norms, sampled every 50 steps (first 1k).
                if (log_fn is not None
                        and run_epoch.global_step < 1000
                        and run_epoch.global_step % 50 == 0):
                    layer0 = model.encoder.layers[0].self_attn
                    log_fn({
                        "grad_norm/W_Q": layer0.W_Q.weight.grad.norm().item(),
                        "grad_norm/W_K": layer0.W_K.weight.grad.norm().item(),
                        "step": run_epoch.global_step,
                    })
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                run_epoch.global_step += 1

        # accuracy + confidence — one extra sync per batch (token counts/correct/conf).
        with torch.no_grad():
            mask = (tgt_out != pad_idx)
            preds = logits.argmax(-1)
            correct = ((preds == tgt_out) & mask).sum().item()
            probs = F.softmax(logits, dim=-1)
            gold_prob = probs.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)
            conf_sum = gold_prob[mask].sum().item()
            n_tok = mask.sum().item()

        loss_val = loss.item()
        total_loss    += loss_val
        total_correct += correct
        total_conf    += conf_sum
        total_tokens  += n_tok
        n_batches += 1
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc  = total_correct / max(total_tokens, 1)
    avg_conf = total_conf    / max(total_tokens, 1)
    prefix = "train" if is_train else "val"
    if log_fn is not None:
        log_fn({
            f"{prefix}/epoch_loss":  avg_loss,
            f"{prefix}/accuracy":    avg_acc,
            f"{prefix}/confidence":  avg_conf,
            "train/lr": optimizer.param_groups[0]["lr"] if optimizer is not None else 0,
            "epoch": epoch_num,
        })
    return avg_loss


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
        for src, tgt in tqdm(test_dataloader, desc="bleu",
                              mininterval=30, leave=False, ncols=80):
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
        "src_vocab":            getattr(model, "src_vocab", None),
        "tgt_vocab":            getattr(model, "tgt_vocab", None),
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
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
        "d_model":      256,
        "N":            3,
        "num_heads":    8,
        "d_ff":         1024,
        "dropout":      0.3,
        "smoothing":    0.1,         # ABLATION 2.5: set 0.0 to disable
        "warmup_steps": 1000,
        "num_epochs":   15,
        "wandb_log_epochs": None,    # if set, stop logging to W&B after this many epochs (training continues)
        "min_freq":     2,
        # Ablation toggles ─────────────────────────────────
        "use_noam":     True,        # ABLATION 2.1: False → fixed LR
        "fixed_lr":     1e-4,        # used when use_noam=False
        "lr_scale":     0.5,         # multiplies the Noam base_lr (halves peak LR)
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
    # Attach so save_checkpoint persists them for autograder Transformer().infer().
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab

    if cfg["use_noam"]:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr_scale"],
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
    log_until = cfg["wandb_log_epochs"] if cfg["wandb_log_epochs"] is not None else cfg["num_epochs"]
    for epoch in range(cfg["num_epochs"]):
        epoch_log_fn = log_fn if epoch < log_until else None
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                               epoch_num=epoch, is_train=True, device=device,
                               log_fn=epoch_log_fn)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None,
                             epoch_num=epoch, is_train=False, device=device,
                             log_fn=epoch_log_fn)

        print(f"[epoch {epoch}] train={train_loss:.4f}  val={val_loss:.4f}")

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


def run_all_ablations(num_epochs: int = 20, project: str = "DA6401_Assignment_3", **common) -> None:
    """Train every ablation in ABLATIONS sequentially, each as its own W&B run.

    Extra kwargs (e.g. warmup_steps=1000) are applied to every run's config;
    per-ablation overrides in ABLATIONS still take precedence."""
    for name, overrides in ABLATIONS.items():
        cfg = {"num_epochs": num_epochs, **common, **overrides}
        print(f"\n=== ablation: {name}  (cfg={cfg}) ===")
        run_training_experiment(project=project, run_name=name, config=cfg)


def get_encoder_attention(model: Transformer, layer_idx: int = -1) -> torch.Tensor:
    """Return attention weights [B, num_heads, S, S] from one encoder self-attn layer."""
    return model.encoder.layers[layer_idx].self_attn.last_attn_weights


def log_attention_heatmaps(
    model: Transformer,
    german_sentence: str,
    src_vocab,
    layer_idx: int = -1,
    wandb_log: bool = True,
    run_name: str = "baseline",
) -> None:
    """§2.3 — render per-head attention heatmaps for the last encoder layer.

    Tokenises the input, runs one encoder forward pass, then plots one heatmap
    per head and (optionally) logs them to W&B as `attention/head_<h>`.
    """
    import matplotlib.pyplot as plt
    from dataset import _spacy, SOS_IDX, EOS_IDX

    device = next(model.parameters()).device
    model.eval()

    # tokenise → ids → encode
    de = _spacy("de_core_news_sm")
    toks = ["<sos>"] + [t.text.lower() for t in de.tokenizer(german_sentence)] + ["<eos>"]
    ids = [SOS_IDX] + [src_vocab.stoi.get(t, 0) for t in toks[1:-1]] + [EOS_IDX]
    src = torch.tensor([ids], dtype=torch.long, device=device)
    src_mask = make_src_mask(src, PAD_IDX)
    with torch.no_grad():
        model.encode(src, src_mask)
    attn = get_encoder_attention(model, layer_idx)[0]    # [H, S, S]

    H, S, _ = attn.shape
    fig, axes = plt.subplots(2, (H + 1) // 2, figsize=(3 * ((H + 1) // 2), 6))
    for h, ax in enumerate(axes.flat):
        if h >= H:
            ax.axis("off"); continue
        ax.imshow(attn[h].cpu(), cmap="viridis")
        ax.set_title(f"head {h}")
        ax.set_xticks(range(S)); ax.set_yticks(range(S))
        ax.set_xticklabels(toks, rotation=90, fontsize=6)
        ax.set_yticklabels(toks, fontsize=6)
    fig.suptitle(f"Encoder layer {layer_idx} attention — '{german_sentence}'")
    fig.tight_layout()

    if wandb_log:
        import wandb
        wandb.init(project="DA6401_Assignment_3", name=f"{run_name}_attn", reinit=True)
        wandb.log({"attention/heatmap": wandb.Image(fig)})
        wandb.finish()
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    run_training_experiment()
