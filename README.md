# DSTG-Former: a graph-transformer for intrusion detection in DNP3 smart grids

Reference implementation for the paper *DSTG-Former: A Graph-Transformer for
Intrusion Detection in DNP3 Smart Grids* (International Journal of Intelligent
Engineering and Systems).

Everything reported in the paper is produced by the scripts in this
repository: the raw-PCAP preprocessing, the protocol-semantics-guided
relabeling, the construction of the sliding windows and of the partition
indices, the topology reconstruction, every model and ablation configuration,
the random seeds, the class-weight computation, the metric and
checkpoint-selection code, and the scripts that generate the tables and
figures.

## 1. Data

The dataset is the public **DNP3 Intrusion Detection Dataset**, IEEE DataPort,
doi:10.21227/s7h0-b081. Download it and keep the distributed directory layout
(one folder per scenario, each containing a `DNP3 PCAP Files` folder).

```
pip install -r requirements.txt
python preprocess_pcaps.py /path/to/DNP3_Intrusion_Detection_Dataset dnp3_transformer_dataset.pt
```

This performs the deep packet inspection (payloads beginning with the DNP3
synchronization bytes `0x05 0x64`), standardizes every payload to 64 bytes,
and groups consecutive payloads into windows of `T = 20` snapshots with a
stride of 5. Captures are processed one after another, so the stored order is
chronological within each capture. The result is a tensor of
`725,708 x 20 x 64` bytes plus one capture-level label per window.

The partition indices are **not** shipped as a file; they are regenerated
deterministically from this tensor by `common.py`, so they can be reproduced
exactly from the public data.

## 2. Evaluation protocol

`common.py` implements the protocol used for every number in the paper.

* `semantic_relabel` applies the protocol-semantics-guided relabeling: a
  window from a command-injection capture keeps its attack label only if at
  least one packet carries the corresponding DNP3 function code at payload
  offset 12 (`0x15`, `0x0D`, `0x0E`, `0x0F`, `0x12`); all other windows become
  `Normal`. This yields ten classes.
* `chronological_split` partitions the windows **inside every capture** into
  the first 70%, the next 15% and the last 15%, discarding the four windows
  that straddle each cut. A window spans exactly four strides, so no packet
  can appear in more than one partition.
* `verify_disjoint` proves this by expanding every window into its
  capture-local packet indices and intersecting the three sets. Run
  `python controls.py` to reproduce the verification, the trivial rule-based
  function-code detector, and the measurement of how much the previously used
  random partition leaks.
* `mask_fc=True` zeroes payload byte 12, that is, the field the labels are
  derived from. This is the leakage control.

## 3. Reproducing the paper

```
python controls.py                                                  # Table 2, protocol verification
python train_strict.py --name strict_main    --epochs 40 --seed 42  # main model
python train_strict.py --name strict_maskfc  --epochs 40 --seed 42 --mask-fc
python train_strict.py --name strict_no_gnn  --epochs 40 --seed 42 --variant no_gnn
python train_strict.py --name strict_seed7   --epochs 40 --seed 7
python train_strict.py --name strict_seed13  --epochs 40 --seed 13
python train_strict.py --name holdout_devices --epochs 40 --seed 42 --holdout-devices 16,17,18
python train_strict.py --name strict_no_transformer  --epochs 40 --seed 42 --variant no_transformer
python train_strict.py --name strict_random_topology --epochs 40 --seed 42 --variant random_topology
python train_strict.py --name strict_mean_pool       --epochs 40 --seed 42 --variant mean_pool
python train_strict.py --name strict_with_pe         --epochs 40 --seed 42 --variant with_pe
python baselines.py --which rf                                      # Table 5 baseline 1
python baselines.py --which cnn --epochs 25                         # Table 5 baseline 2
python bench.py                                                     # deployment benchmark
python extract_feats.py && python make_figs_rev.py                  # all figures
python resnums.py                                                   # every number quoted in the paper
```

Each run writes a JSON file into `results/` containing the full configuration,
the training history, the confusion matrix, the per-class report, the selected
epoch and the timing.

**Checkpoint selection.** After every epoch the macro F1-score is computed on
the validation partition; the parameters of the epoch with the highest
validation macro F1-score are stored and restored before the test partition is
evaluated exactly once. The validation figures quoted in the paper are those
of the selected checkpoint, not the maximum observed at an arbitrary epoch.

**Seeds.** `torch.manual_seed` and `numpy.random.seed` are set from the
`--seed` argument at the start of every run. The reported reference run uses
seed 42; the confidence intervals use seeds 42, 7 and 13.

## 4. Model

`train_dstg.py` holds `DSTGFormer`, the model used by all scripts. The
architecture is a byte-embedding front end, an edge-conditioned GNN over the
13-device communication graph recovered from the DNP3 link-layer addresses
(destination in bytes 4-5, source in bytes 6-7, little endian), and a
transformer encoder with attentive temporal pooling. Device identity is routed
exclusively through the graph branch: bytes 4-9 are masked in the semantic
branch, which makes the topology contribution explicit and measurable.

Default configuration: model dimension 128, node embedding dimension 64, eight
heads, three encoder layers, dropout 0.1, Adam with an initial learning rate of
3e-4 and cosine annealing, gradient clipping at 1.0, batch size 1024, mixed
precision, class-weighted cross entropy with weights inversely proportional to
the training frequency of each class. 2.88 million parameters.

## 5. Hardware

The reported runs use a single NVIDIA GeForce RTX 2070 SUPER (8 GB). One run
takes about 33 minutes. Do not run two trainings concurrently: the dataset is
held on the device in uint8 and each process needs roughly 4.5 GB.

## 6. Citation

Please cite the paper if you use this code.
