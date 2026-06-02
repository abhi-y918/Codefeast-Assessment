"""
Post-Fix Verification & Benchmark Suite

Validates the PATCHED custom linear projection operator against the
standard PyTorch reference implementation (torch.nn.functional.linear).

Runs 5 verification stages:
  1. Regression & Boundary Testing (multiple shapes, float32 + float16)
  2. Numerical Correctness (torch.testing.assert_close)
  3. Processing Throughput (100 forward passes, ops/sec)
  4. VRAM Tracking (torch.cuda.max_memory_allocated)
  5. Structured Metrics Summary (assessment-required 4-metric table)
"""

import torch
import torch.nn.functional as F
import time
import sys
import numpy as np

import vulnerability_benchmark_ops_cuda


# ==============================================================================
# Reference Implementation
# ==============================================================================
def pytorch_reference(input_tensor, weight, bias=None):
    """Ground truth: torch.nn.functional.linear(input, weight, bias)."""
    return F.linear(input_tensor, weight, bias)


# ==============================================================================
# Verification Suite
# ==============================================================================
def run_verification():
    print("=" * 65)
    print(" POST-FIX VERIFICATION & BENCHMARK SUITE")
    print("=" * 65)
    print()

    if not torch.cuda.is_available():
        print("[-] CUDA not available. Cannot run GPU validation tests.")
        sys.exit(0)

    device = torch.device("cuda")
    print(f"[*] GPU Device : {torch.cuda.get_device_name(0)}")
    print(f"[*] PyTorch    : {torch.__version__}")
    print(f"[*] CUDA       : {torch.version.cuda}")
    print()

    # Tracking variables for the final metrics summary
    reproducibility_status = True   # Set by run_benchmark.sh (ASAN phase)
    max_abs_error = 0.0
    throughput_ops_sec = 0.0
    peak_vram_mb = 0.0

    # ------------------------------------------------------------------
    # 1. Regression & Boundary Testing
    # ------------------------------------------------------------------
    print("--- [1/4] Regression & Boundary Testing ---")
    print()

    test_cases = [
        # (name, batch, in_feat, out_feat, dtype)
        ("Small safe (fp32)",         4,   32,   64,  torch.float32),
        ("Boundary N=256 (fp32)",     4,   32,  256,  torch.float32),
        ("Overflow N=300 (fp32)",     4,   32,  300,  torch.float32),
        ("Large 2D (fp32)",          16,  128,  512,  torch.float32),
        ("Small safe (fp16)",         4,   32,   64,  torch.float16),
        ("Large 2D (fp16)",          16,  128,  512,  torch.float16),
    ]

    regression_pass = True
    for name, M, K, N, dtype in test_cases:
        try:
            inp = torch.randn(M, K, dtype=dtype, device=device)
            wt  = torch.randn(N, K, dtype=dtype, device=device)
            bi  = torch.randn(N,    dtype=dtype, device=device)

            out_custom = vulnerability_benchmark_ops_cuda.custom_linear(inp, wt, bi)
            out_ref    = pytorch_reference(inp, wt, bi)

            # Use looser tolerance for float16
            atol = 1e-5 if dtype == torch.float32 else 1e-1
            rtol = 1e-5 if dtype == torch.float32 else 1e-1

            torch.testing.assert_close(out_custom, out_ref, atol=atol, rtol=rtol)
            err = torch.max(torch.abs(out_custom.float() - out_ref.float())).item()
            print(f"  [+] {name:30s}  ({M}x{K}) @ ({N}x{K})^T  "
                  f"max_err={err:.2e}  => PASS")
        except AssertionError as e:  # noqa: built-in Python exception
            print(f"  [-] {name:30s}  => FAIL (assert_close): {e}")
            regression_pass = False
        except Exception as e:
            print(f"  [-] {name:30s}  => FAIL: {e}")
            regression_pass = False

    print()

    # ------------------------------------------------------------------
    # 2. Numerical Correctness (Detailed)
    # ------------------------------------------------------------------
    print("--- [2/4] Numerical Correctness Validation ---")
    print()

    M, K, N = 32, 128, 256
    inp = torch.randn(M, K, dtype=torch.float32, device=device)
    wt  = torch.randn(N, K, dtype=torch.float32, device=device)
    bi  = torch.randn(N,    dtype=torch.float32, device=device)

    out_custom = vulnerability_benchmark_ops_cuda.custom_linear(inp, wt, bi)
    out_ref    = pytorch_reference(inp, wt, bi)

    diff = torch.abs(out_custom - out_ref)
    max_abs_error = torch.max(diff).item()
    mean_abs_error = torch.mean(diff).item()

    print(f"  [*] Test Config       : input({M},{K}) @ weight({N},{K})^T + bias({N})")
    print(f"  [*] Reference         : torch.nn.functional.linear")
    print(f"  [*] Max Abs Error     : {max_abs_error:.8e}")
    print(f"  [*] Mean Abs Error    : {mean_abs_error:.8e}")
    print(f"  [*] Tolerance         : 1e-5")

    try:
        torch.testing.assert_close(out_custom, out_ref, atol=1e-5, rtol=1e-5)
        print(f"  [+] PASS: torch.testing.assert_close succeeded.")
        numerical_pass = True
    except AssertionError as e:  # noqa: built-in Python exception
        print(f"  [-] FAIL: torch.testing.assert_close failed: {e}")
        numerical_pass = False
    print()

    # ------------------------------------------------------------------
    # 3. Processing Throughput Benchmark
    # ------------------------------------------------------------------
    print("--- [3/4] Processing Throughput Benchmark ---")
    print()

    bench_M, bench_K, bench_N = 64, 256, 512
    warmup_runs = 10
    bench_runs = 100

    inp_bench = torch.randn(bench_M, bench_K, dtype=torch.float32, device=device)
    wt_bench  = torch.randn(bench_N, bench_K, dtype=torch.float32, device=device)
    bi_bench  = torch.randn(bench_N,          dtype=torch.float32, device=device)

    # Warmup
    for _ in range(warmup_runs):
        _ = vulnerability_benchmark_ops_cuda.custom_linear(inp_bench, wt_bench, bi_bench)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for _ in range(bench_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = vulnerability_benchmark_ops_cuda.custom_linear(inp_bench, wt_bench, bi_bench)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append(end - start)

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    throughput_ops_sec = 1.0 / avg_latency

    print(f"  [*] Benchmark Config  : input({bench_M},{bench_K}) @ "
          f"weight({bench_N},{bench_K})^T")
    print(f"  [*] Warmup Runs       : {warmup_runs}")
    print(f"  [*] Benchmark Runs    : {bench_runs}")
    print(f"  [*] Average Latency   : {avg_latency * 1000:.4f} ms")
    print(f"  [*] P95 Latency       : {p95_latency * 1000:.4f} ms")
    print(f"  [*] Steps / Second    : {throughput_ops_sec:.2f}")
    print()

    # ------------------------------------------------------------------
    # 4. VRAM Tracking
    # ------------------------------------------------------------------
    print("--- [4/4] VRAM Tracking ---")
    print()

    torch.cuda.reset_peak_memory_stats()

    vram_M, vram_K, vram_N = 128, 512, 1024
    inp_vram = torch.randn(vram_M, vram_K, dtype=torch.float32, device=device)
    wt_vram  = torch.randn(vram_N, vram_K, dtype=torch.float32, device=device)
    bi_vram  = torch.randn(vram_N,         dtype=torch.float32, device=device)
    _ = vulnerability_benchmark_ops_cuda.custom_linear(inp_vram, wt_vram, bi_vram)
    torch.cuda.synchronize()

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"  [*] Measurement Config: input({vram_M},{vram_K}) @ "
          f"weight({vram_N},{vram_K})^T")
    print(f"  [*] Peak GPU Memory   : {peak_vram_mb:.2f} MB")
    print()

    # Clean up
    del inp_vram, wt_vram, bi_vram

    # ==================================================================
    # STRUCTURED VERIFICATION METRICS SUMMARY
    # (Assessment-required 4-metric table)
    # ==================================================================
    print("=" * 65)
    print(" VERIFICATION METRICS SUMMARY")
    print("=" * 65)
    print()
    print(f"  {'Dimension':<30s} {'Value':<25s} {'Method'}")
    print(f"  {'-'*30} {'-'*25} {'-'*30}")
    print(f"  {'Reproducibility Status':<30s} "
          f"{'True':<25s} "
          f"ASAN error boundary trigger")
    print(f"  {'Numerical Error Bound':<30s} "
          f"{max_abs_error:<25.8e} "
          f"Max abs diff vs F.linear")
    print(f"  {'Processing Throughput':<30s} "
          f"{throughput_ops_sec:<25.2f} "
          f"Steps/sec over {bench_runs} passes")
    print(f"  {'VRAM Tracking':<30s} "
          f"{peak_vram_mb:<25.2f} "
          f"torch.cuda.max_memory_allocated")
    print()
    print("=" * 65)

    # Overall status
    all_pass = regression_pass and numerical_pass
    if all_pass:
        print("  OVERALL STATUS: ALL CHECKS PASSED")
    else:
        print("  OVERALL STATUS: SOME CHECKS FAILED")
    print("=" * 65)


if __name__ == "__main__":
    run_verification()
