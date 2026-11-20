"""
train_transformer.py
Trains the small Transformer from scratch on the training alerts and keeps
the epoch with the best validation accuracy.

Example:  <BOS> alert fields <SEP> severity actions <EOS>
Loss is computed only on the answer after <SEP>.

Usage:  python src/train_transformer.py --epochs 40
"""
import argparse
import csv
import json
import os
import random
import time

import torch
import torch.nn as nn

import config as C
from log_parser import load_split, to_model_text
from model import TinyTransformerLM
from tokenizer import BOS, EOS, PAD, SEP, Tokenizer

TOKENIZER_PATH = os.path.join(C.MODEL_DIR, "tokenizer.json")
WEIGHTS_PATH = os.path.join(C.MODEL_DIR, "transformer.pt")
CONFIG_PATH = os.path.join(C.MODEL_DIR, "transformer_config.json")


def answer_text(row):
    """The target answer: severity first, then action codes."""
    return " ".join([row["severity"]] + row["actions"])


def make_sequence(tok, alert_text, answer=None):
    """
    Build token ids for one example. Returns (ids, answer_start), where
    answer_start is the index of the first answer token.
    """
    ids = [tok.id(BOS)] + tok.encode(alert_text) + [tok.id(SEP)]
    start = len(ids)
    if answer is not None:
        ids += tok.encode(answer) + [tok.id(EOS)]
    return ids, start


def batchify(examples, pad_id):
    """Pad a list of (ids, start) to the same length. Targets are -100 (ignored) outside the answer."""
    n = max(len(ids) for ids, _ in examples)
    x = torch.full((len(examples), n - 1), pad_id)
    y = torch.full((len(examples), n - 1), -100)
    for i, (ids, start) in enumerate(examples):
        t = torch.tensor(ids)
        x[i, :len(ids) - 1] = t[:-1]            # input: every token except the last
        tgt = t[1:].clone()                      # target: every token except the first
        tgt[:start - 1] = -100                   # only learn to predict the answer tokens
        y[i, :len(ids) - 1] = tgt
    return x, y


# Prediction helpers (also used by evaluate.py and triage.py)
def load_trained():
    tok = Tokenizer.load(TOKENIZER_PATH)
    with open(CONFIG_PATH) as fh:
        cfg = json.load(fh)
    model = TinyTransformerLM(len(tok), pad_id=tok.id(PAD), **cfg)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return tok, model


def predict(tok, model, fields, include_msg=True):
    """
    Run the Transformer on one parsed alert.
    Returns (severity, [actions], confidence) where confidence is the
    probability the model gave its severity token.
    """
    ids, _ = make_sequence(tok, to_model_text(fields, include_msg))
    out, probs = model.generate(ids, tok.id(EOS), return_probs=True)
    words = tok.decode(out).split()
    if not words or words[0] not in C.SEVERITIES:
        return "invalid", [], 0.0           # output validation: not a real severity
    actions = [w for w in words[1:] if w in C.ACTIONS]
    return words[0], actions, probs[0]


def severity_accuracy(tok, model, rows):
    correct = sum(predict(tok, model, r["fields"])[0] == r["severity"] for r in rows)
    return correct / len(rows)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    train = load_split(os.path.join(C.DATA_DIR, "train.csv"))
    val = load_split(os.path.join(C.DATA_DIR, "val.csv"))

    # 1) Tokenizer: vocabulary from TRAINING data only (val/test words may become <UNK>)
    tok = Tokenizer().build([to_model_text(r["fields"]) + " " + answer_text(r) for r in train])
    tok.save(TOKENIZER_PATH)
    print(f"Vocabulary size: {len(tok)}")

    # 2) Model with random initial weights
    cfg = dict(d_model=128, n_heads=4, n_layers=3, d_ff=256, max_len=64, dropout=0.1)
    model = TinyTransformerLM(len(tok), pad_id=tok.id(PAD), **cfg)
    print(f"Parameters: {model.num_parameters():,}")

    examples = [make_sequence(tok, to_model_text(r["fields"]), answer_text(r)) for r in train]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best_acc, log = -1.0, []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(examples)
        total, batches = 0.0, 0
        for i in range(0, len(examples), args.batch):
            x, y = batchify(examples[i:i + args.batch], tok.id(PAD))
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            batches += 1

        val_acc = severity_accuracy(tok, model, val)
        log.append({"epoch": epoch, "train_loss": round(total / batches, 4), "val_severity_acc": round(val_acc, 4)})
        print(f"epoch {epoch:3d}  loss {total / batches:.4f}  val acc {val_acc:.3f}")
        if val_acc > best_acc:                    # keep the best epoch, not the last
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)

    with open(CONFIG_PATH, "w") as fh:
        json.dump(cfg, fh, indent=1)
    with open(os.path.join(C.RESULTS_DIR, "training_log.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_severity_acc"])
        w.writeheader()
        w.writerows(log)
    print(f"\nBest validation accuracy: {best_acc:.3f}   training time: {time.time() - start_time:.0f}s")
    print(f"Saved model to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
