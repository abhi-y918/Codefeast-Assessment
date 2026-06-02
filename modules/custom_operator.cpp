#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <iostream>

// =============================================================================
// Custom Linear Projection Operator — C++ Binding Layer
//
// Implements: output = input @ weight^T + bias
// Equivalent to: torch.nn.functional.linear(input, weight, bias)
//
// This file contains an INTENTIONAL memory safety vulnerability in the
// vulnerable build path (#ifndef PATCH_APPLIED). A fixed-size stack buffer
// is used to store per-output-feature normalization metadata, which overflows
// when the number of output features (N) exceeds 256.
//
// The patched build path (#ifdef PATCH_APPLIED) replaces the static buffer
// with bounds-checked dynamic allocation via std::vector.
// =============================================================================

// Forward declaration of the CUDA kernel launch wrapper (defined in kernel.cu)
void launch_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int M, int N, int K,
    bool has_bias
);

// -----------------------------------------------------------------------------
// Main operator entry point
// -----------------------------------------------------------------------------
torch::Tensor custom_linear(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {

    // --- Tensor Property Validation ---
    TORCH_CHECK(input.device().is_cuda(),       "Input tensor must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(),      "Weight tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(),          "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(),         "Weight tensor must be contiguous");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat16,
        "Input tensor must be float32 or float16"
    );
    TORCH_CHECK(
        input.scalar_type() == weight.scalar_type(),
        "Input and weight tensors must have the same dtype"
    );

    // --- Dimensionality Validation ---
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (batch_size x in_features)");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D (out_features x in_features)");

    int M = input.size(0);   // batch size
    int K = input.size(1);   // in_features
    int N = weight.size(0);  // out_features

    TORCH_CHECK(M > 0 && K > 0, "Input tensor must not be empty");
    TORCH_CHECK(N > 0, "Weight tensor must have at least one output feature");

    TORCH_CHECK(weight.size(1) == K,
        "Weight in_features (", weight.size(1), ") must match input in_features (", K, ")");

    // --- Bias Validation ---
    bool has_bias = bias.defined() && bias.numel() > 0;
    if (has_bias) {
        TORCH_CHECK(bias.device().is_cuda(),    "Bias tensor must be a CUDA tensor");
        TORCH_CHECK(bias.is_contiguous(),       "Bias tensor must be contiguous");
        TORCH_CHECK(bias.dim() == 1,            "Bias must be 1D (out_features)");
        TORCH_CHECK(bias.size(0) == N,
            "Bias size (", bias.size(0), ") must match out_features (", N, ")");
    }

    // =========================================================================
    // HOST-SIDE WEIGHT METADATA PRE-COMPUTATION
    //
    // Before launching the CUDA kernel, we pre-compute per-output-feature
    // normalization scaling factors (Xavier/He-style 1/sqrt(K)) on the host.
    // This metadata buffer is where the vulnerability resides.
    // =========================================================================

#ifndef PATCH_APPLIED
    // =====================================================================
    // VULNERABLE IMPLEMENTATION
    //
    // BUG: Fixed-size stack buffer assumes N (out_features) <= 256.
    //      When a weight matrix with more than 256 output features is
    //      passed from Python, the loop writes past the end of the buffer,
    //      corrupting adjacent stack memory.
    //
    // Vulnerability Type : Stack Buffer Overflow (CWE-121)
    // Trigger Condition  : weight.size(0) > 256
    // ASAN Detection     : stack-buffer-overflow
    // =====================================================================
    float feature_scales[256];

    for (int i = 0; i < N; i++) {
        // Compute per-output-feature Xavier normalization factor.
        // When N > 256, this writes out-of-bounds on the stack.
        feature_scales[i] = 1.0f / sqrtf(static_cast<float>(K));
    }

    // Volatile read prevents the compiler from optimizing away the buffer
    volatile float _sink = feature_scales[0];
    (void)_sink;

#else
    // =====================================================================
    // SECURE (PATCHED) IMPLEMENTATION
    //
    // Fix 1: TORCH_CHECK bounds validation rejects excessively large N
    //        to prevent heap exhaustion from adversarial weight shapes.
    // Fix 2: std::vector dynamically allocates exactly N elements on the
    //        heap, eliminating the fixed-capacity assumption entirely.
    // =====================================================================
    TORCH_CHECK(N <= 100000,
        "Too many output features (N=", N, "). Max supported: 100,000.");

    std::vector<float> feature_scales(N);

    for (int i = 0; i < N; i++) {
        feature_scales[i] = 1.0f / sqrtf(static_cast<float>(K));
    }

    volatile float _sink = feature_scales[0];
    (void)_sink;
#endif

    // --- Allocate output tensor ---
    auto output = torch::empty({M, N}, input.options());

    // --- Launch CUDA kernel ---
    launch_linear_forward(input, weight, bias, output, M, N, K, has_bias);

    return output;
}

// =============================================================================
// Python Bindings (pybind11)
// =============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_linear", &custom_linear,
          "Custom linear projection operator (CUDA) — output = input @ weight^T + bias",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = torch::Tensor()  // bias is optional, defaults to undefined
    );
}
