# -*- coding: utf-8 -*-
"""Zero-training controls requested by the reviewers:
  (a) verification that the strict partitions are packet-disjoint;
  (b) the trivial rule-based function-code classifier;
  (c) the same rule applied to the original random partition, for reference;
  (d) exact (unrounded) accuracy / weighted-F1 recomputation for the
      previously reported model, to answer Reviewer 1's query.
"""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

X, y, src, dst, tr, va, te, blocks = load_all(mask_fc=False, device="cpu")
print("blocks:", blocks, flush=True)

rep = verify_disjoint(tr, va, te, blocks)
print(json.dumps(rep, indent=1), flush=True)

# ---- class distribution per partition -------------------------------------
yn = y.numpy()
dist = {k: np.bincount(yn[v], minlength=NUM_CLASSES).tolist()
        for k, v in (("train", tr), ("val", va), ("test", te))}
dist["all"] = np.bincount(yn, minlength=NUM_CLASSES).tolist()

# ---- (b) rule-based function-code classifier -------------------------------
fc = X[:, :, FC_BYTE]
def rule_predict(ids):
    f = fc[torch.as_tensor(np.asarray(ids, dtype=np.int64))]
    pred = np.full(len(ids), 9, dtype=np.int64)      # default: Normal
    for c, code in ATTACK_FC.items():
        hit = (f == code).any(dim=1).numpy()
        pred[hit & (pred == 9)] = c
    return pred

pred_rule = rule_predict(te)
res_rule = metrics(yn[te], pred_rule,
                   extra={"config": "rule_based_fc", "params": 0,
                          "protocol": "strict_chronological",
                          "note": "predict class if its attack function code "
                                  "appears anywhere in the window, else Normal"})
save("rule_based_fc_strict", res_rule)
print("rule-based FC  acc=%.4f  macroF1=%.4f" %
      (res_rule["test_accuracy"], res_rule["f1_macro"]), flush=True)

# same rule on the original stratified-random partition
from sklearn.model_selection import train_test_split
idx = np.arange(len(yn))
tr0, tmp = train_test_split(idx, test_size=0.30, stratify=yn, random_state=42)
va0, te0 = train_test_split(tmp, test_size=0.50, stratify=yn[tmp],
                            random_state=42)
res_rule0 = metrics(yn[te0], rule_predict(te0),
                    extra={"config": "rule_based_fc", "params": 0,
                           "protocol": "random_window_level"})
save("rule_based_fc_random", res_rule0)
print("rule-based FC (random split) acc=%.4f macroF1=%.4f" %
      (res_rule0["test_accuracy"], res_rule0["f1_macro"]), flush=True)

# ---- window-overlap statistics of the ORIGINAL random partition ------------
starts = np.array([a for a, _ in blocks])
span = max(b - a for a, b in blocks) * STRIDE + SEQ_LEN + 1
seen = np.zeros((len(blocks), span), dtype=bool)
off = np.arange(SEQ_LEN)
def _loc(ids):
    cap = np.searchsorted(starts, ids, side="right") - 1
    return cap, ids - starts[cap]
cap_tr, loc_tr = _loc(tr0)
for c in range(0, len(tr0), 200000):
    sl = slice(c, c + 200000)
    seen[np.repeat(cap_tr[sl], SEQ_LEN),
         (loc_tr[sl, None] * STRIDE + off[None, :]).ravel()] = True
cap_te, loc_te = _loc(te0)
frac_shared = seen[np.repeat(cap_te, SEQ_LEN),
                   (loc_te[:, None] * STRIDE + off[None, :]).ravel()
                   ].reshape(len(te0), SEQ_LEN).mean(axis=1)
overlap_stats = {
    "protocol": "random_window_level",
    "test_windows": int(len(te0)),
    "mean_fraction_of_packets_also_in_train": float(frac_shared.mean()),
    "test_windows_fully_covered_by_train": int((frac_shared == 1.0).sum()),
    "test_windows_with_zero_overlap": int((frac_shared == 0.0).sum()),
}
save("overlap_random_split", overlap_stats)
print(json.dumps(overlap_stats, indent=1), flush=True)

save("strict_protocol", {"partition_report": rep, "class_distribution": dist,
                         "blocks": [[int(a), int(b)] for a, b in blocks],
                         "guard_windows": GUARD})

# ---- (d) exact metrics of the previously reported model --------------------
prev = json.load(open(os.path.join(BASE, "results", "no_pe.json")))
cm = np.array(prev["confusion_matrix"])
tot = cm.sum()
acc = np.trace(cm) / tot
supp = cm.sum(1)
f1s = []
for c in range(NUM_CLASSES):
    tp_ = cm[c, c]; fp_ = cm[:, c].sum() - tp_; fn_ = supp[c] - tp_
    p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
    r = tp_ / supp[c] if supp[c] else 0.0
    f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
wf1 = float(np.average(f1s, weights=supp))
save("previous_model_exact", {
    "protocol": "random_window_level (previously reported)",
    "test_accuracy_exact": float(acc), "f1_weighted_exact": wf1,
    "f1_macro_exact": float(np.mean(f1s)),
    "per_class_f1": f1s, "support": supp.tolist(),
    "note": "recomputed independently from the stored confusion matrix"})
print("previous model: acc=%.6f  weightedF1=%.6f  macroF1=%.6f" %
      (acc, wf1, np.mean(f1s)), flush=True)
