#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>

// =============================================================================
// Linear Projection CUDA Kernel (Naive GEMM)
//
// Computes: output = input @ weight^T + bias
//   input:  (M, K)   — batch of input vectors
//   weight: (N, K)   — weight matrix (N output neurons, K input features)
//   bias:   (N,)     — optional bias vector
//   output: (M, N)   — projected output
//
// Supports: float32 and float16 via AT_DISPATCH_FLOATING_TYPES_AND_HALF.
// Accumulation is done in float32 for numerical stability even with float16
// inputs, following mixed-precision best practices.
// =============================================================================

template <typename scalar_t>
__global__ void linear_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int M, int N, int K,
    bool has_bias
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch dimension (M)
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output feature dimension (N)

    if (row < M && col < N) {
        // Accumulate dot product in float32 for precision (important for float16)
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += static_cast<float>(input[row * K + k])
                 * static_cast<float>(weight[col * K + k]);
        }
        if (has_bias) {
            acc += static_cast<float>(bias[col]);
        }
        output[row * N + col] = static_cast<scalar_t>(acc);
    }
}

// =============================================================================
// Host-side kernel launch wrapper
// Dispatches the templated kernel based on the tensor's scalar type.
// =============================================================================
void launch_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int M, int N, int K,
    bool has_bias
) {
    const int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "linear_forward_cuda", [&] {
            linear_forward_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(),
                M, N, K,
                has_bias
            );
        }
    );

    // Synchronize and propagate CUDA errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err)
        );
    }
}
