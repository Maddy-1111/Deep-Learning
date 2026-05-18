"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘

Mask convention: True = MASK OUT (set to -inf before softmax).
"""

import math
import copy
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Special-token indices (kept in sync with dataset.py to avoid a circular import).
PAD_IDX, SOS_IDX, EOS_IDX = 1, 2, 3


# ══════════════════════════════════════════════════════════════════════
#  SCALED DOT-PRODUCT ATTENTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: bool = True,                       # ABLATION 2.2: toggle 1/√dk
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if scale:
        scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out, attn


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    # [B, S] → [B, 1, 1, S]; True where token is <pad>
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    # [B, T] → [B, 1, T, T]; True for <pad> OR future positions
    T = tgt.size(1)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)              # [B,1,1,T]
    causal = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=tgt.device), diagonal=1
    )                                                                  # [T,T]
    return pad_mask | causal                                           # → [B,1,T,T]


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        attn_scale: bool = True,             # ABLATION 2.2
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attn_scale = attn_scale

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights: Optional[torch.Tensor] = None  # for ABLATION 2.3

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, seq, d_model] → [B, num_heads, seq, d_k]
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = query.size(0)
        Q = self._split_heads(self.W_Q(query))
        K = self._split_heads(self.W_K(key))
        V = self._split_heads(self.W_V(value))

        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask, scale=self.attn_scale)
        self.last_attn_weights = attn.detach()
        out = self.dropout(out)

        # [B, num_heads, seq_q, d_k] → [B, seq_q, d_model]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_O(out)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── ABLATION 2.4: Learned positional embeddings ───────────────────────

class LearnedPositionalEncoding(nn.Module):
    """nn.Embedding-based positional encoding (drop-in replacement for the sinusoidal one)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # [1, seq]
        return self.dropout(x + self.pe(positions))


# ══════════════════════════════════════════════════════════════════════
#  POSITION-WISE FEED-FORWARD
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER / DECODER LAYERS  (Post-LayerNorm — paper-faithful)
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int,
        dropout: float = 0.1, attn_scale: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, attn_scale=attn_scale)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int,
        dropout: float = 0.1, attn_scale: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout, attn_scale=attn_scale)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, attn_scale=attn_scale)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    # === HOSTED WEIGHTS ============================================
    # Upload your best checkpoint.pt to Google Drive and put the file ID here.
    # Used only when the model is instantiated with no args (autograder path).
    WEIGHTS_GDRIVE_ID: str = "REPLACE_WITH_YOUR_GDRIVE_FILE_ID"
    WEIGHTS_LOCAL_PATH: str = "checkpoint.pt"
    # ===============================================================

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model: int = 256,
        N: int = 3,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.3,
        pos_encoding: str = "sinusoidal",   # ABLATION 2.4: "sinusoidal" | "learned"
        attn_scale: bool = True,            # ABLATION 2.2
        max_decode_len: int = 100,
    ) -> None:
        super().__init__()

        # Autograder path: no vocab sizes → download checkpoint and use its config.
        auto_load = (src_vocab_size is None or tgt_vocab_size is None)
        ckpt = self._download_and_load_checkpoint() if auto_load else None
        if auto_load:
            cfg = ckpt["model_config"]
            src_vocab_size = cfg["src_vocab_size"]
            tgt_vocab_size = cfg["tgt_vocab_size"]
            d_model      = cfg["d_model"]
            N            = cfg["N"]
            num_heads    = cfg["num_heads"]
            d_ff         = cfg["d_ff"]
            dropout      = cfg["dropout"]
            pos_encoding = cfg.get("pos_encoding", "sinusoidal")
            attn_scale   = cfg.get("attn_scale", True)

        self.d_model = d_model
        self.max_decode_len = max_decode_len

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        pe_cls = {"sinusoidal": PositionalEncoding,
                  "learned":    LearnedPositionalEncoding}[pos_encoding]
        self.pos_enc = pe_cls(d_model, dropout)

        self.encoder = Encoder(EncoderLayer(d_model, num_heads, d_ff, dropout, attn_scale), N)
        self.decoder = Decoder(DecoderLayer(d_model, num_heads, d_ff, dropout, attn_scale), N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Xavier init for all weight matrices (paper §5.4 / Annotated Transformer).
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Will be populated either by auto-load (below) or by the training script.
        self.src_vocab = None
        self.tgt_vocab = None
        self._spacy_de = None

        if auto_load:
            self.load_state_dict(ckpt["model_state_dict"])
            self.src_vocab = ckpt["src_vocab"]
            self.tgt_vocab = ckpt["tgt_vocab"]
            self._spacy_de = self._load_spacy_de()

    # ── auto-load helpers ──────────────────────────────────────────

    @classmethod
    def _download_and_load_checkpoint(cls) -> dict:
        if not os.path.exists(cls.WEIGHTS_LOCAL_PATH):
            import gdown
            url = f"https://drive.google.com/uc?id={cls.WEIGHTS_GDRIVE_ID}"
            gdown.download(url, cls.WEIGHTS_LOCAL_PATH, quiet=False)
        return torch.load(cls.WEIGHTS_LOCAL_PATH, map_location="cpu", weights_only=False)

    @staticmethod
    def _load_spacy_de():
        import spacy
        return spacy.load("de_core_news_sm",
                          disable=["parser", "tagger", "ner", "lemmatizer"])

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.generator(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ── End-to-end inference (autograder entry point) ──────────────
    @torch.no_grad()
    def infer(self, german_sentence: str) -> str:
        """Translate a single German sentence to English.

        Tokenises with spaCy → encodes → greedy-decodes → detokenises.
        Requires `self.src_vocab`, `self.tgt_vocab`, and `self._spacy_de`
        to be populated (auto-load handles this; the training script
        attaches them before calling save_checkpoint)."""
        assert self.src_vocab is not None and self.tgt_vocab is not None, \
            "Vocab not loaded — instantiate Transformer() to auto-load weights+vocabs."
        if self._spacy_de is None:
            self._spacy_de = self._load_spacy_de()

        device = next(self.parameters()).device
        self.eval()

        # tokenize + encode
        toks = [t.text.lower() for t in self._spacy_de.tokenizer(german_sentence)]
        ids = [SOS_IDX] + [self.src_vocab.stoi.get(t, 0) for t in toks] + [EOS_IDX]
        src = torch.tensor([ids], dtype=torch.long, device=device)
        src_mask = make_src_mask(src, PAD_IDX)

        # greedy decode
        memory = self.encode(src, src_mask)
        ys = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)
        for _ in range(self.max_decode_len - 1):
            tgt_mask = make_tgt_mask(ys, PAD_IDX)
            logits = self.decode(memory, src_mask, ys, tgt_mask)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
            if nxt.item() == EOS_IDX:
                break

        # detokenize
        out_tokens = []
        for i in ys[0, 1:].tolist():        # skip <sos>
            if i == EOS_IDX:
                break
            out_tokens.append(self.tgt_vocab.lookup_token(i))
        return " ".join(out_tokens)
