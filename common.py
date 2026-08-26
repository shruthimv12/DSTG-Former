# -*- coding: utf-8 -*-
"""Shared utilities for the IJIES revision experiments (paper 20265400).

Implements the strict evaluation protocol requested by Reviewer 2:
  * chronologically disjoint, capture-aware train/val/test partitions built
    BEFORE any sliding-window reuse can cross a partition boundary;
  * an explicit guard band that provably removes every packet-level overlap;
  * an option to mask the DNP3 application function code (payload byte 12),
    i.e. the field used to construct the ground-truth labels.
"""
import os, json, math, time
import numpy as np
import torch

BASE = os.environ.get("DNP3_ROOT",
                      os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results")
os.makedirs(OUT, exist_ok=True)

CLASS_NAMES = ["Disable Unsolicited", "Cold Restart", "Warm Restart",
               "Enumerate", "Info", "Initialize Data", "MITM-DoS", "Replay",
               "Stop Application", "Normal"]
NUM_CLASSES = 10

ATTACK_FC = {0: 0x15, 1: 0x0D, 2: 0x0E, 5: 0x0F, 8: 0x12}
FC_BYTE = 12                      # payload offset of the DNP3 function code
SEQ_LEN = 20
STRIDE = 5                        # window stride used when the dataset was built
GUARD = SEQ_LEN // STRIDE         # 4 windows -> provably packet-disjoint

ADDR_VOCAB = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18]
NUM_NODES = len(ADDR_VOCAB)
ADDR_LUT = torch.zeros(65536, dtype=torch.long)
for i, a in enumerate(ADDR_VOCAB):
    ADDR_LUT[a] = i


def semantic_relabel(X, y):
    y = y.clone()
    fc = X[:, :, FC_BYTE]
    for c, code in ATTACK_FC.items():
        m = y == c
        has_attack = (fc[m] == code).any(dim=1)
        idx = torch.nonzero(m, as_tuple=True)[0]
        y[idx[~has_attack]] = 9
    return y


def capture_blocks(y_raw):
    """Return [(start, end)] for each contiguous capture in the stored order."""
    y = y_raw.numpy()
    ch = np.nonzero(np.diff(y))[0] + 1
    b = [0] + ch.tolist() + [len(y)]
    return list(zip(b[:-1], b[1:]))


def chronological_split(blocks, frac=(0.70, 0.15, 0.15), guard=GUARD):
    """Chronologically disjoint partitions inside every capture.

    Windows are consumed in capture order; the guard band drops the `guard`
    windows that straddle each cut so that no packet can appear in two
    partitions.
    """
    tr, va, te = [], [], []
    for a, b in blocks:
        n = b - a
        c1 = a + int(round(frac[0] * n))
        c2 = a + int(round((frac[0] + frac[1]) * n))
        tr.append(np.arange(a, c1))
        va.append(np.arange(min(c1 + guard, c2), c2))
        te.append(np.arange(min(c2 + guard, b), b))
    return (np.concatenate(tr), np.concatenate(va), np.concatenate(te))


def verify_disjoint(tr, va, te, blocks, seq_len=SEQ_LEN, stride=STRIDE):
    """Assert that no raw packet is shared between two partitions.

    Packets are addressed capture-locally: window i of capture (a, b) covers
    packets [(i - a) * stride, (i - a) * stride + seq_len) of that capture, so
    windows from different captures can never collide.
    """
    starts = np.array([a for a, _ in blocks])
    span = max(b - a for a, b in blocks) * stride + seq_len + 1
    def mask(ids):
        m = np.zeros((len(blocks), span), dtype=bool)
        cap = np.searchsorted(starts, ids, side="right") - 1
        loc = ids - starts[cap]
        off = np.arange(seq_len)
        for c in range(0, len(ids), 200000):
            sl = slice(c, c + 200000)
            m[np.repeat(cap[sl], seq_len),
              (loc[sl, None] * stride + off[None, :]).ravel()] = True
        return m
    A, B, C = mask(tr), mask(va), mask(te)
    return {
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "packets_train": int(A.sum()), "packets_val": int(B.sum()),
        "packets_test": int(C.sum()),
        "overlap_train_val": int((A & B).sum()),
        "overlap_train_test": int((A & C).sum()),
        "overlap_val_test": int((B & C).sum()),
    }


def load_all(mask_fc=False, device="cuda"):
    d = torch.load(os.path.join(BASE, "dnp3_transformer_dataset.pt"))
    X, y_raw = d["data"], d["labels"].long()
    y = semantic_relabel(X, y_raw)
    dst = ADDR_LUT[(X[:, :, 4].long() + X[:, :, 5].long() * 256).clamp(0, 65535)]
    src = ADDR_LUT[(X[:, :, 6].long() + X[:, :, 7].long() * 256).clamp(0, 65535)]
    if mask_fc:
        X = X.clone()
        X[:, :, FC_BYTE] = 0
    blocks = capture_blocks(y_raw)
    tr, va, te = chronological_split(blocks)
    return X, y, src, dst, tr, va, te, blocks


def metrics(targs, preds, probs=None, extra=None):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, classification_report,
                                 confusion_matrix)
    cm = confusion_matrix(targs, preds, labels=list(range(NUM_CLASSES)))
    # macro false-positive rate
    fp = cm.sum(0) - np.diag(cm)
    tn = cm.sum() - cm.sum(0) - cm.sum(1) + np.diag(cm)
    fpr = np.divide(fp, fp + tn, out=np.zeros(NUM_CLASSES, float),
                    where=(fp + tn) > 0)
    res = {
        "test_accuracy": float(accuracy_score(targs, preds)),
        "precision_macro": float(precision_score(targs, preds, average="macro",
                                                 zero_division=0)),
        "recall_macro": float(recall_score(targs, preds, average="macro",
                                           zero_division=0)),
        "f1_macro": float(f1_score(targs, preds, average="macro",
                                   zero_division=0)),
        "f1_weighted": float(f1_score(targs, preds, average="weighted",
                                      zero_division=0)),
        "precision_weighted": float(precision_score(targs, preds,
                                    average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(targs, preds, average="weighted",
                                              zero_division=0)),
        "fpr_macro": float(fpr.mean()),
        "fpr_per_class": fpr.tolist(),
        "confusion_matrix": cm.tolist(),
        "report": classification_report(targs, preds,
                                        labels=list(range(NUM_CLASSES)),
                                        target_names=CLASS_NAMES,
                                        output_dict=True, zero_division=0),
    }
    if probs is not None:
        try:
            res["roc_auc_ovr_macro"] = float(roc_auc_score(
                targs, probs, multi_class="ovr", average="macro",
                labels=list(range(NUM_CLASSES))))
        except Exception as e:
            res["roc_auc_ovr_macro"] = None
    if extra:
        res.update(extra)
    return res


def save(name, res):
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    print("[saved]", name, flush=True)
