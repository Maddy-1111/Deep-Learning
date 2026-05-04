"""
dataset.py — Multi30k German→English loader for DA6401 Assignment 3.

Pipeline:
    HF `bentrevett/multi30k`  →  spaCy tokenize  →  Vocab built on train
    →  encode to ids w/ <sos>/<eos>  →  pad-collate for DataLoader.

Specials are fixed: <unk>=0, <pad>=1, <sos>=2, <eos>=3 (matches model.py).
Run once before first use:
    python -m spacy download de_core_news_sm
    python -m spacy download en_core_web_sm
"""

from collections import Counter
from functools import lru_cache, partial

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import spacy
from datasets import load_dataset


SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3


@lru_cache(maxsize=2)
def _spacy(name: str):
    return spacy.load(name, disable=["parser", "tagger", "ner", "lemmatizer"])


class Vocab:
    """Token↔id mapping built from a list of tokenised sentences."""

    def __init__(self, token_lists, min_freq: int = 2):
        counter: Counter[str] = Counter()
        for toks in token_lists:
            counter.update(toks)

        self.itos = list(SPECIALS) + sorted(
            (t for t, c in counter.items() if c >= min_freq and t not in SPECIALS),
            key=lambda t: (-counter[t], t),
        )
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, UNK_IDX) for t in tokens]

    def lookup_token(self, idx: int) -> str:
        return self.itos[idx]


class Multi30kDataset(Dataset):
    """
    Multi30k de→en wrapper. Yields `(src_ids, tgt_ids)` LongTensors with
    <sos>/<eos> wrapping. For non-train splits, pass the train vocabs.
    """

    def __init__(
        self,
        split: str = "train",
        src_vocab: Vocab | None = None,
        tgt_vocab: Vocab | None = None,
        min_freq: int = 2,
    ) -> None:
        ds = load_dataset("bentrevett/multi30k", split=split)
        de, en = _spacy("de_core_news_sm"), _spacy("en_core_web_sm")

        self.src_tokens = [[t.text.lower() for t in de.tokenizer(ex["de"])] for ex in ds]
        self.tgt_tokens = [[t.text.lower() for t in en.tokenizer(ex["en"])] for ex in ds]

        if src_vocab is None or tgt_vocab is None:
            if split != "train":
                raise ValueError("Pass src_vocab and tgt_vocab for non-train splits")
            src_vocab = Vocab(self.src_tokens, min_freq=min_freq)
            tgt_vocab = Vocab(self.tgt_tokens, min_freq=min_freq)
        self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab

        self.process_data()

    def build_vocab(self, min_freq: int = 2) -> None:
        self.src_vocab = Vocab(self.src_tokens, min_freq=min_freq)
        self.tgt_vocab = Vocab(self.tgt_tokens, min_freq=min_freq)

    def process_data(self) -> None:
        def _wrap(toks, vocab):
            return torch.tensor([SOS_IDX] + vocab.encode(toks) + [EOS_IDX], dtype=torch.long)
        self.src_ids = [_wrap(t, self.src_vocab) for t in self.src_tokens]
        self.tgt_ids = [_wrap(t, self.tgt_vocab) for t in self.tgt_tokens]

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        return self.src_ids[idx], self.tgt_ids[idx]


def collate_fn(batch, pad_idx: int = PAD_IDX):
    """Pad a batch of (src, tgt) variable-length tensors to a common length."""
    src_batch, tgt_batch = zip(*batch)
    src = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src, tgt


def get_dataloaders(batch_size: int = 128, min_freq: int = 2, num_workers: int = 0):
    """Return (train_loader, val_loader, test_loader, src_vocab, tgt_vocab)."""
    train_ds = Multi30kDataset("train", min_freq=min_freq)
    val_ds   = Multi30kDataset("validation", train_ds.src_vocab, train_ds.tgt_vocab)
    test_ds  = Multi30kDataset("test",       train_ds.src_vocab, train_ds.tgt_vocab)

    collate = partial(collate_fn, pad_idx=PAD_IDX)
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                    collate_fn=collate, num_workers=num_workers)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False), \
           train_ds.src_vocab, train_ds.tgt_vocab


if __name__ == "__main__":
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(batch_size=4)
    print(f"|src vocab| = {len(src_vocab)}   |tgt vocab| = {len(tgt_vocab)}")
    print(f"batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    src, tgt = next(iter(train_loader))
    print("src batch:", src.shape, "tgt batch:", tgt.shape)
    print("first src toks:", [src_vocab.lookup_token(i) for i in src[0].tolist()])
    print("first tgt toks:", [tgt_vocab.lookup_token(i) for i in tgt[0].tolist()])
