# -*- coding: utf-8 -*-
"""Collect every number quoted in the revised manuscript from the result files,
so that the text and the tables can never drift apart."""
import os, json
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")

CLASSES = ["Disable Unsolicited", "Cold Restart", "Warm Restart", "Enumerate",
           "Info", "Initialize Data", "MITM-DoS", "Replay",
           "Stop Application", "Normal"]


def load(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def pc(v, d=2):
    return "TBD" if v is None else ("%." + str(d) + "f") % (100.0 * v)


def f4(v):
    return "TBD" if v is None else "%.4f" % v


def get(r, k, default=None):
    return r.get(k, default) if r else None


def collect():
    N = {}
    R = {k: load(k) for k in
         ["strict_main", "strict_maskfc", "strict_no_gnn", "strict_seed7",
          "strict_seed13", "holdout_devices", "strict_no_transformer",
          "strict_random_topology", "strict_mean_pool", "strict_with_pe",
          "rule_based_fc_strict", "rule_based_fc_random", "baseline_rf",
          "baseline_cnn_bilstm", "overlap_random_split", "strict_protocol",
          "previous_model_exact", "deployment_benchmark"]}
    N["R"] = R
    m = R["strict_main"]
    for tag, r in [("", m), ("MASK", R["strict_maskfc"]),
                   ("NOGNN", R["strict_no_gnn"]), ("RULE", R["rule_based_fc_strict"]),
                   ("RF", R["baseline_rf"]), ("CNN", R["baseline_cnn_bilstm"]),
                   ("HOLD", R["holdout_devices"]), ("NOTR", R["strict_no_transformer"]),
                   ("RTOP", R["strict_random_topology"]), ("MEAN", R["strict_mean_pool"]),
                   ("PE", R["strict_with_pe"])]:
        N["ACC" + tag] = pc(get(r, "test_accuracy"))
        N["MF1" + tag] = pc(get(r, "f1_macro"))
        N["WF1" + tag] = pc(get(r, "f1_weighted"))
        N["AUC" + tag] = f4(get(r, "roc_auc_ovr_macro"))
        N["FPR" + tag] = pc(get(r, "fpr_macro"), 3)
        N["PAR" + tag] = ("%.2f" % (get(r, "params") / 1e6)) if get(r, "params") else "TBD"

    # exact, unrounded accuracy / weighted F1 of the previous (random) protocol
    pm = R["previous_model_exact"]
    N["OLD_ACC"] = "%.3f" % (100 * pm["test_accuracy_exact"]) if pm else "TBD"
    N["OLD_WF1"] = "%.3f" % (100 * pm["f1_weighted_exact"]) if pm else "TBD"
    N["OLD_MF1"] = "%.2f" % (100 * pm["f1_macro_exact"]) if pm else "TBD"
    if m:
        N["ACC_X"] = "%.3f" % (100 * m["test_accuracy"])
        N["WF1_X"] = "%.3f" % (100 * m["f1_weighted"])
        N["EPOCH"] = str(m.get("selected_epoch", "TBD"))
        N["VALACC"] = pc(m.get("val_accuracy_at_checkpoint"))
        N["VALMF1"] = pc(m.get("val_macro_f1_at_checkpoint"))
        N["TRAINMIN"] = "%.0f" % (m.get("train_time_s", 0) / 60.0)
        N["EPOCHS"] = str(m.get("epochs", 40))

    ov = R["overlap_random_split"]
    if ov:
        N["OVL_MEAN"] = "%.1f" % (100 * ov["mean_fraction_of_packets_also_in_train"])
        N["OVL_FULL"] = "%.1f" % (100.0 * ov["test_windows_fully_covered_by_train"]
                                  / ov["test_windows"])
    sp = R["strict_protocol"]
    if sp:
        pr = sp["partition_report"]
        N["NTR"] = "{:,}".format(pr["n_train"])
        N["NVA"] = "{:,}".format(pr["n_val"])
        N["NTE"] = "{:,}".format(pr["n_test"])

    # multi-seed statistics
    seeds = [R["strict_main"], R["strict_seed7"], R["strict_seed13"]]
    seeds = [s for s in seeds if s]
    if len(seeds) >= 2:
        a = np.array([s["test_accuracy"] for s in seeds]) * 100
        f = np.array([s["f1_macro"] for s in seeds]) * 100
        t = 4.303 if len(seeds) == 3 else 12.706          # t(0.975, n-1)
        N["SEED_N"] = str(len(seeds))
        N["SEED_ACC"] = "%.2f" % a.mean()
        N["SEED_ACC_CI"] = "%.2f" % (t * a.std(ddof=1) / np.sqrt(len(a)))
        N["SEED_MF1"] = "%.2f" % f.mean()
        N["SEED_MF1_CI"] = "%.2f" % (t * f.std(ddof=1) / np.sqrt(len(f)))
    else:
        for k in ["SEED_N", "SEED_ACC", "SEED_ACC_CI", "SEED_MF1", "SEED_MF1_CI"]:
            N[k] = "TBD"

    b = R["deployment_benchmark"]
    if b:
        g = b.get("cuda") or b.get("cpu")
        c = b.get("cpu", g)
        N["LAT_GPU"] = "%.2f" % g["latency_ms_mean"]
        N["LAT_GPU95"] = "%.2f" % g["latency_ms_p95"]
        N["LAT_CPU"] = "%.2f" % c["latency_ms_mean"]
        N["THR"] = "{:,.0f}".format(g["throughput_windows_per_s"])
        N["THR_PKT"] = "{:,.0f}".format(g["throughput_packets_per_s"])
        N["THR_CPU"] = "{:,.0f}".format(c["throughput_windows_per_s"])
        N["MEM"] = "%.0f" % g.get("peak_gpu_memory_MB", 0)
        N["MODELMB"] = "%.1f" % g["model_size_MB"]
    else:
        for k in ["LAT_GPU", "LAT_GPU95", "LAT_CPU", "THR", "THR_PKT",
                  "THR_CPU", "MEM", "MODELMB"]:
            N[k] = "TBD"

    # per-class rows of the strict model
    rows = []
    if m:
        rep = m["report"]
        for c in CLASSES:
            d = rep[c]
            rows.append([c, "%.2f" % (100 * d["precision"]),
                         "%.2f" % (100 * d["recall"]),
                         "%.2f" % (100 * d["f1-score"]),
                         "{:,}".format(int(d["support"]))])
    N["PERCLASS"] = rows

    # recall of the four classes the function-code rule cannot express
    if m:
        rep = m["report"]
        for key, lab in [("ENUM", "Enumerate"), ("INFO", "Info"),
                         ("MITM", "MITM-DoS"), ("REPL", "Replay"),
                         ("NORM", "Normal")]:
            N["REC_" + key] = "%.1f" % (100 * rep[lab]["recall"])
    if R["rule_based_fc_strict"]:
        rep = R["rule_based_fc_strict"]["report"]
        N["RULE_REC_ENUM"] = "%.1f" % (100 * rep["Enumerate"]["recall"])
        N["RULE_REC_INFO"] = "%.1f" % (100 * rep["Info"]["recall"])
    return N


if __name__ == "__main__":
    N = collect()
    for k in sorted(N):
        if k not in ("R", "PERCLASS"):
            print("%-14s %s" % (k, N[k]))
    for r in N["PERCLASS"]:
        print(r)
