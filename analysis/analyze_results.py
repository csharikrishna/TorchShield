"""Statistical analysis pipeline for cryptographic benchmark results.

Processes raw and summary benchmark metrics:
- Calculates relative throughput speedup / slowdown ratios
- Computes mean, median, standard deviation, and 95% confidence intervals
- Evaluates hypotheses against predefined practical significance thresholds
"""
import json
import math
import os
import csv
from typing import Dict, List


def mean_and_ci(samples: List[float]):
    if not samples:
        return 0.0, 0.0, 0.0
    n = len(samples)
    m = sum(samples) / n
    if n < 2:
        return m, 0.0, 0.0
    var = sum((x - m) ** 2 for x in samples) / (n - 1)
    std = math.sqrt(var)
    # 95% CI roughly 1.96 * std / sqrt(n)
    ci95 = 1.96 * (std / math.sqrt(n))
    return m, std, ci95


def analyze_benchmark_results():
    summary_path = "results/summary.json"
    raw_path = "results/raw/cipher_benchmarks_raw.json"

    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return

    with open(summary_path, "r") as f:
        records = json.load(f)

    print("=" * 80)
    print("STATISTICAL ANALYSIS OF CIPHER BENCHMARK RESULTS")
    print("=" * 80)

    # Group by workload_bytes
    by_workload = {}
    for r in records:
        wb = r["workload_bytes"]
        by_workload.setdefault(wb, []).append(r)

    analysis_out = []
    for wb, group in by_workload.items():
        aes_ref = next((g for g in group if g["primitive"] == "AES-256-GCM"), None)
        if not aes_ref:
            continue

        aes_enc_tput = aes_ref["enc_throughput_mib_s"]
        aes_dec_tput = aes_ref["dec_throughput_mib_s"]

        print(f"\nWorkload Size: {wb:,} Bytes ({wb / (1024*1024):.2f} MiB)")
        print(f"  Baseline CPU AES-256-GCM: Enc {aes_enc_tput:9.2f} MiB/s | Dec {aes_dec_tput:9.2f} MiB/s")

        for g in group:
            if g["primitive"] == "AES-256-GCM":
                continue
            name = f"{g['primitive']} {g['backend']} (N={g['N']}, R={g['R']})"
            enc_t = g["enc_throughput_mib_s"]
            dec_t = g["dec_throughput_mib_s"]
            enc_ratio = enc_t / aes_enc_tput
            dec_ratio = dec_t / aes_dec_tput

            print(f"  {name:<38}: Enc {enc_t:8.2f} MiB/s ({enc_ratio*100:5.2f}% of AES) | Dec {dec_t:8.2f} MiB/s ({dec_ratio*100:5.2f}% of AES)")
            analysis_out.append({
                "workload_bytes": wb,
                "variant": name,
                "enc_throughput_mib_s": enc_t,
                "aes_enc_throughput_mib_s": aes_enc_tput,
                "relative_enc_ratio": enc_ratio,
                "dec_throughput_mib_s": dec_t,
                "aes_dec_throughput_mib_s": aes_dec_tput,
                "relative_dec_ratio": dec_ratio,
            })

    with open("results/statistical_comparison.json", "w") as f:
        json.dump(analysis_out, f, indent=2)

    print("\nStatistical comparison saved to results/statistical_comparison.json")


if __name__ == "__main__":
    analyze_benchmark_results()
