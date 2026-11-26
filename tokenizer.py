"""
tokenizer.py
Word-level tokenizer built only from this project's alert data (no pretrained vocabulary).

Example:
  "device=PLC-01 event=unauthorized_write"
  -> ["device", "=", "PLC-01", "event", "=", "unauthorized_write"]
IP addresses and names like PLC-01 stay whole, so each device and event is one token.

Special tokens: <PAD> padding, <UNK> unseen word, <BOS>/<EOS> start/end,
<SEP> separates the alert from the triage answer.
"""

import json
import re
from collections import Counter

PAD, UNK, BOS, EOS, SEP = "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"
SPECIALS = [PAD, UNK, BOS, EOS, SEP]

# Order matters: try IPs first, then word-like tokens (letters, digits, - _ .), then "="
TOKEN_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}|[A-Za-z0-9_\-.:]+|=|<[A-Z]+>")


class Tokenizer:
    def __init__(self, vocab=None):
        self.itos = vocab or list(SPECIALS)            # id -> token
        self.stoi = {t: i for i, t in enumerate(self.itos)}  # token -> id

    # ----- building -----
    @staticmethod
    def tokenize(text):
        """Split text into tokens (lowercasing is not used: PLC-01 and plc-01 differ in logs)."""
        return TOKEN_RE.findall(text)

    def build(self, texts, min_freq=1):
        """Create the vocabulary from training texts only."""
        counts = Counter(tok for t in texts for tok in self.tokenize(t))
        for tok, n in sorted(counts.items()):
            if n >= min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
        return self

    # converting 
    def encode(self, text, add_bos=False, add_eos=False):
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in self.tokenize(text)]
        if add_bos:
            ids = [self.stoi[BOS]] + ids
        if add_eos:
            ids = ids + [self.stoi[EOS]]
        return ids

    def decode(self, ids, skip_specials=True):
        toks = [self.itos[i] for i in ids]
        if skip_specials:
            toks = [t for t in toks if t not in SPECIALS]
        return " ".join(toks)

    def id(self, token):
        return self.stoi[token]

    def __len__(self):
        return len(self.itos)

    # saving 
    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.itos, fh, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(json.load(fh))


if __name__ == "__main__":
    tok = Tokenizer().build(["device=PLC-01 event=unauthorized_write src=10.0.9.77"])
    text = "device=PLC-01 event=unauthorized_write"
    print("tokens:", tok.tokenize(text))
    ids = tok.encode(text, add_bos=True, add_eos=True)
    print("ids:   ", ids)
    print("back:  ", tok.decode(ids))
