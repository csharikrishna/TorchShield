"""Plot generation script for cryptographic benchmark evaluation.

Generates:
1. results/throughput_comparison.png (Throughput across payload sizes)
2. results/latency_comparison.png (Latency scaling)
"""
import json
import os
import sys

def generate_plots():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot image generation.")
        return

    summary_path = "results/summary.json"
    if not os.path.exists(summary_path):
        print("results/summary.json not found; cannot plot.")
        return

    with open(summary_path, "r") as f:
        records = json.load(f)

    # Filter distinct configurations
    configs = {}
    for r in records:
        key = f"{r['primitive']} {r['backend']}"
        if r['N'] != "N/A":
            key += f" (N={r['N']}, R={r['R']})"
        configs.setdefault(key, []).append((r['workload_bytes'], r['enc_throughput_mib_s'], r['enc_latency_ms']))

    # Plot 1: Throughput
    plt.figure(figsize=(10, 6))
    for name, pts in configs.items():
        pts.sort(key=lambda x: x[0])
        sizes_kb = [x[0] / 1024 for x in pts]
        tputs = [x[1] for x in pts]
        plt.plot(sizes_kb, tputs, marker="o", label=name)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Workload Size (KiB)")
    plt.ylabel("Encryption Throughput (MiB/s) [Log Scale]")
    plt.title("Cryptographic Throughput: Standard AES-256-GCM vs AHCC Variants")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("results/throughput_comparison.png", dpi=300)
    plt.close()
    print("Plot saved: results/throughput_comparison.png")


if __name__ == "__main__":
    generate_plots()
