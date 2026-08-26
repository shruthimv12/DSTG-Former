# -*- coding: utf-8 -*-
"""Step 1: turn the raw DNP3 Intrusion Detection Dataset PCAP files into the
window tensor used by every experiment.

Deep packet inspection keeps only transport-layer payloads that begin with the
DNP3 synchronization bytes 0x05 0x64.  Each payload is standardized to 64
bytes (truncate or zero-pad) and consecutive payloads are grouped into sliding
windows of 20 snapshots with a stride of 5.  Captures are processed one after
another, so the stored order is chronological inside every capture, which is
what the chronological partition in common.py relies on.

Usage:
    python preprocess_pcaps.py <dataset_root> dnp3_transformer_dataset.pt

<dataset_root> is the directory holding one subdirectory per scenario, each
with a "DNP3 PCAP Files" folder, exactly as distributed on IEEE DataPort
(doi:10.21227/s7h0-b081).
"""
import os, sys
import numpy as np
import torch
from scapy.all import PcapReader

SEQ_LENGTH = 20
PAYLOAD_SIZE = 64
STRIDE = 5


def process(base_path, output_path, seq_length=SEQ_LENGTH,
            payload_size=PAYLOAD_SIZE, stride=STRIDE):
    dataset, labels = [], []
    subfolders = sorted(f for f in os.listdir(base_path)
                        if os.path.isdir(os.path.join(base_path, f)))
    for label_idx, folder in enumerate(subfolders):
        folder_path = os.path.join(base_path, folder, "DNP3 PCAP Files")
        if not os.path.exists(folder_path):
            continue
        print("[*] %s -> class %d" % (folder, label_idx), flush=True)
        files = sorted(f for f in os.listdir(folder_path)
                       if f.lower().endswith((".pcap", ".pcapng")))
        for pcap_file in files:
            payloads = []
            try:
                for pkt in PcapReader(os.path.join(folder_path, pcap_file)):
                    raw = b""
                    if pkt.haslayer("TCP"):
                        raw = bytes(pkt["TCP"].payload)
                    elif pkt.haslayer("UDP"):
                        raw = bytes(pkt["UDP"].payload)
                    if raw.startswith(b"\x05\x64"):
                        payloads.append(list(raw[:payload_size]
                                             .ljust(payload_size, b"\x00")))
            except Exception as e:
                print("    [!] %s: %s" % (pcap_file, e))
                continue
            if len(payloads) < seq_length:
                print("    [!] %s: only %d payloads, skipped"
                      % (pcap_file, len(payloads)))
                continue
            print("    [+] %s: %d payloads" % (pcap_file, len(payloads)))
            for i in range(0, len(payloads) - seq_length + 1, stride):
                dataset.append(payloads[i:i + seq_length])
                labels.append(label_idx)

    if not dataset:
        raise SystemExit("no DNP3 sequences found")
    print("[*] %d sequences" % len(dataset), flush=True)
    torch.save({"data": torch.tensor(np.array(dataset, dtype=np.uint8),
                                     dtype=torch.uint8),
                "labels": torch.tensor(labels, dtype=torch.int16)},
               output_path)
    print("[+] saved", output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    process(sys.argv[1],
            sys.argv[2] if len(sys.argv) > 2 else "dnp3_transformer_dataset.pt")
