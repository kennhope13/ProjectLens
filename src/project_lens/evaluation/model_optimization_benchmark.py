"""Measure reranker latency and throughput for different inference batch sizes."""

from __future__ import annotations

import argparse
import time

import numpy as np
from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-v2-m3"
SAMPLE_PAIRS = [
    ("Giá máy bơm diesel là bao nhiêu?", "Máy bơm diesel Q=54m3/h; H=30mH2O. Đơn giá: 76.312.962 VND."),
    ("Thông số cáp điện C120", "Cáp điện C120 theo tiêu chuẩn TCVN, chiều dài 500m."),
    ("Trạm biến áp 110kV Phước Lý", "Hồ sơ dự án trạm biến áp 110kV Phước Lý."),
    ("Đầu đọc thẻ Hikvision", "Đầu đọc thẻ DS-K1107AM hỗ trợ thẻ Mifare và EM."),
]


def benchmark(batch_sizes: list[int], pair_count: int, runs: int, device: str) -> None:
    started = time.perf_counter()
    model = CrossEncoder(MODEL_NAME, device=device)
    load_seconds = time.perf_counter() - started
    pairs = (SAMPLE_PAIRS * ((pair_count + len(SAMPLE_PAIRS) - 1) // len(SAMPLE_PAIRS)))[:pair_count]

    print(f"Model: {MODEL_NAME} | device={device} | load={load_seconds:.2f}s")
    print(f"Pairs per run: {len(pairs)} | measured runs: {runs}")
    print("batch_size, p50_ms, p95_ms, pairs_per_second")

    results = []
    for batch_size in batch_sizes:
        model.predict(pairs, batch_size=batch_size, show_progress_bar=False)  # warm-up
        durations = []
        for _ in range(runs):
            start = time.perf_counter()
            model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            durations.append(time.perf_counter() - start)

        p50 = float(np.percentile(durations, 50))
        p95 = float(np.percentile(durations, 95))
        throughput = len(pairs) / float(np.mean(durations))
        results.append((batch_size, p50, p95, throughput))
        print(f"{batch_size}, {p50 * 1000:.1f}, {p95 * 1000:.1f}, {throughput:.2f}")

    baseline = results[0][3]
    fastest = max(results, key=lambda result: result[3])
    print(
        f"Fastest measured batch size: {fastest[0]} "
        f"({fastest[3] / baseline:.2f}x throughput vs batch size {results[0][0]})."
    )
    print("Use measurements from the target machine; this benchmark does not convert or quantize the model.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=32, help="Number of query-document pairs per run")
    parser.add_argument("--runs", type=int, default=5, help="Measured repetitions per batch size")
    parser.add_argument("--device", default="cpu", help="Sentence Transformers device, e.g. cpu or cuda")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    args = parser.parse_args()
    if args.pairs < 1 or args.runs < 1 or any(size < 1 for size in args.batch_sizes):
        parser.error("pairs, runs, and batch sizes must all be positive")
    benchmark(args.batch_sizes, args.pairs, args.runs, args.device)


if __name__ == "__main__":
    main()
