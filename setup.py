import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ==============================================================================
# Build Configuration for Custom Linear Projection Operator
#
# Compiles custom_operator.cpp (host) and kernel.cu (device) into a single
# Python-importable extension module: vulnerability_benchmark_ops_cuda
#
# Key flags:
#   -fsanitize=address     → Enable AddressSanitizer on host code
#   -fno-omit-frame-pointer → Preserve stack frames for meaningful ASAN traces
#   -DPATCH_APPLIED=1      → Activates the secure (patched) code path
# ==============================================================================

is_patched = os.environ.get("PATCH_APPLIED", "0") == "1"

# ASAN flags are injected ONLY into the host compiler (cxx), not nvcc.
# NVCC (CUDA compiler) does not support -fsanitize; ASAN hooks are sufficient
# on the host-side C++ code where the vulnerability resides.
extra_compile_args = {
    'cxx': ['-g', '-O0', '-fsanitize=address', '-fno-omit-frame-pointer'],
    'nvcc': ['-g', '-O0']
}

if is_patched:
    extra_compile_args['cxx'].append('-DPATCH_APPLIED=1')

setup(
    name='vulnerability_benchmark_ops',
    ext_modules=[
        CUDAExtension(
            name='vulnerability_benchmark_ops_cuda',
            sources=[
                'modules/custom_operator.cpp',
                'modules/kernel.cu',
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=['-fsanitize=address']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
