# Optional RTX 4090 CUDA backend — manual validation required

## Assumed target and execution policy
Target: RTX 4090, 24 GB VRAM, SM 8.9; user-reported KMD 617.14 / CUDA UMD 13.4; 64 GB DDR5 and Ryzen 9 7900X. These are assumptions, **not verified compatibility claims**. CUDA Toolkit (`nvcc`, headers, runtime development libraries) is a separate prerequisite; UMD version is not proof of Toolkit installation. Use a Toolkit supporting SM89 and a supported host compiler.

No CUDA configuration, compilation, hardware probing, GPU tests or GPU training was performed by the assistant. This source implementation is **uncompiled/unexecuted CUDA code** pending manual validation. Do not infer that the CPU validation results establish CUDA correctness.

## Implementation scope
`cfwx_cuda` owns device parameters, parameter gradients, Adam moments, stream, token buffer and reverse-mode tape. CPU code constructs graph topology and initialization, uploads tokens, and reads loss; the mathematical forward/backward, token-mean batch reduction, clipping and AdamW execute on GPU. There is no CPU numerical fallback.

The model is embeddings + fixed sinusoidal positions + causal projected multi-head attention + pre-LayerNorm residuals + GELU FFN + final LayerNorm + vocabulary head. Joint batch/sequence graph has analytic gradients including shared Q/K/V paths. Dropout masks are explicitly stored for backward. Adam uses one step per entire batch update.

**Performance limit:** forward/backward tape interpreter uses one CUDA thread for ordered scalar operations. This is a correctness-oriented device execution baseline, not an RTX-4090-optimized backend. It does not implement fused batched GEMM, cuBLAS/Tensor Cores, FlashAttention, mixed precision or GPU KV/checkpoint interoperability. Device residency alone is not a claim of speedup. Double precision is deliberately chosen for gradient comparisons; RTX 4090 FP64 throughput is low. Start with the small defaults, not a production-sized model.

CPU `cfwx_run` checkpoint and KV support remains available separately. CUDA parameters have a distinct layout; CPU checkpoint files must not be loaded into CUDA models. CUDA copy APIs support exporting/importing parameter values for tests; optimizer checkpoint/resume is not yet included in the CUDA ABI. The dataset runner reports complete validation batches and explicit excluded-tail count, not full held-out evaluation.

## Build manually on your machine
Use a **fresh build directory**. CPU CMake default and vcpkg integration are preserved. CUDA is OFF by default; CPU CI never enables it. CMake >=3.24 required only for CUDA mode. GNU-only flags are restricted to C compilation so they are not passed to nvcc.

```sh
cmake -S . -B build-cuda -DCFWX_USE_VCPKG=OFF \
  -DCFWX_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCFWX_CUDA_ARCHITECTURES=89
cmake --build build-cuda --parallel 12
```

For Windows native CUDA, choose an nvcc-supported MSVC toolchain. Existing main application uses POSIX getopt/Linux ptrace, so the full repository still targets Linux/WSL; this CUDA addition does not port the legacy CLI to Windows. Linux/WSL Toolkit setup is your responsibility. Do not combine CPU ASan build flags with CUDA; use Compute Sanitizer instead.

## Manual validation order

```sh
# Nothing GPU-related is automatically run during CPU builds.
./build-cuda/Release/cfwx_cuda_manual --test

# Optional explicit GPU CTest registration
cmake -S . -B build-cuda -DCFWX_USE_VCPKG=OFF -DCFWX_CUDA=ON \
  -DCFWX_REGISTER_GPU_TESTS=ON -DCFWX_CUDA_ARCHITECTURES=89
ctest --test-dir build-cuda -L gpu --output-on-failure

compute-sanitizer --tool memcheck ./build-cuda/Release/cfwx_cuda_manual --test
compute-sanitizer --tool initcheck ./build-cuda/Release/cfwx_cuda_manual --test
./build-cuda/Release/cfwx_cuda_manual data/tiny.txt 200
```

The manual test checks every tiny-model parameter gradient against central finite differences executed on the GPU, then asserts substantial loss reduction. It does **not** currently test CPU/CUDA parity, checkpoint persistence, or dropout statistical correctness; add these before production use. Tests are not registered unless requested explicitly.

## Resource and error handling
Default tiny dataset configuration: context=8, batch=4, width=8, heads=2, hidden=16, depth=1; node budget 8 million, device buffer budget 2 GiB. Hard budgets reject oversized graphs before tape allocation. Parameters/moments also count against budget. Host graph memory is additional to device allocations; 64 GB RAM does not remove graph complexity limits. Scalar graph growth is quadratic in context for attention and large for dense products.

API synchronizes at the call boundary and reports CUDA API/launch/numerical failures. Numerical update failure may leave some tensors updated; the model becomes poisoned and must be destroyed. This is explicitly not transactional GPU optimizer rollback. API calls for a model are single-threaded; host logger synchronization is unchanged. No device-property probing is performed. No fast-math option is enabled.
