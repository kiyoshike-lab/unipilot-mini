from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<USER>", "<ASSISTANT>", "<SYSTEM>"]


class FoundationTokenizer:
    """Foundation-only byte-level BPE trained from scratch on licensed Base text."""

    def __init__(self, tokenizer: Tokenizer):
        self.backend = tokenizer
        self.special_tokens = list(SPECIAL_TOKENS)
        self.special_to_id = {
            token: int(tokenizer.token_to_id(token)) for token in self.special_tokens
        }

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<EOS>"]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.backend.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        return self.backend.decode(ids, skip_special_tokens=skip_special)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.backend.save(str(target), pretty=True)

    @classmethod
    def load(cls, path: str | Path) -> "FoundationTokenizer":
        return cls(Tokenizer.from_file(str(path)))


def train_tokenizer(texts, vocab_size: int) -> FoundationTokenizer:
    backend = Tokenizer(models.BPE(unk_token="<UNK>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    )
    backend.train_from_iterator(texts, trainer=trainer)
    tokenizer = FoundationTokenizer(backend)
    if tokenizer.vocab_size != vocab_size:
        raise RuntimeError(
            f"Foundation tokenizer did not reach requested vocab: {tokenizer.vocab_size}/{vocab_size}"
        )
    return tokenizer
