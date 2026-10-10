# CFWx
Educational C17 numerical foundations for ML/DL experiments. Not a production ML framework.

## Build and test (Linux)
Requires C17 compiler and CMake >= 3.16. No mandatory vcpkg or cmocka dependency.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/cfwx
cmake -S . -B build-asan -DCFWX_SANITIZERS=ON
cmake --build build-asan --parallel
ctest --test-dir build-asan --output-on-failure
# Optional when Valgrind is installed
cmake --build build --target valgrind
```

Select vcpkg externally using `-DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake` on first configure. CLI uses POSIX getopt; core has no OS-specific calls. VSCode settings remain user-specific.

## Foundations
- Contiguous row-major vectors, matrices, tensors (rank 1-8); overflow-checked allocation, bounds-checked indexing, metadata-only reshape.
- Compensated dot product, vector addition/scaling, matrix product/transpose.
- Affine neuron and batched dense layer; reproducible local SplitMix64 Xavier-uniform initialization; explicit input/weight/bias gradients; SGD.
- Stable sigmoid, ReLU, exact erf-based GELU and derivatives; tanh derivative; stable softmax.
- MSE, categorical cross entropy from logits, binary cross entropy from logits, gradients.
- Discounted returns with zero terminal bootstrap.
- Per-row layer normalization (Welford population variance), caller-supplied gamma/beta.
- Single-head scaled dot-product attention with optional causal mask and probabilities.

Use public `.h` headers. Old empty `.c` files under `include/maths/` now forward to the corresponding headers.

## Contracts
Zero-initialize owners with `{0}`. Constructors require empty destinations and commit allocations only on success. Matching `_free` functions reset owners and may be called repeatedly. Never shallow-copy owners or fabricate shape/stride metadata. Positive dimensions only. Caller preallocates outputs. Buffers must be disjoint, including partial overlap, except documented exact in-place cases (softmax, scaling, returns, normalization). Operations return `cfwx_status`; errors may leave outputs partially modified. SGD validates all updates before mutation.

Basic vector elementwise arithmetic and transpose follow IEEE-754 and can propagate NaN/Inf. Neural operations generally reject non-finite inputs/results; extreme finite inputs can still overflow. This is double-precision reference code, not BLAS-optimized. Softmax only accepts finite logits; attention handles masking internally.

Dense: W `[inputs,outputs]`, X `[batch,inputs]`, Y `[batch,outputs]`. Backward overwrites gradients without averaging; the loss controls reduction. MSE and BCE divide by scalar element count. Categorical CE takes a single label/vector. Derivatives take preactivation values; ReLU derivative at zero is zero. GELU is exact erf-based, not tanh-approximated.

Attention: Q `[queries,d]`, K `[keys,d]`, V `[keys,value_dim]`. Causal mode requires equal query/key lengths and masks key index above query index. Layer norm divides variance by feature count, not feature count minus one.

## Tests and limits
CTest runner uses checks active under NDEBUG. Tests cover container operations, overflow/bounds, reference matrix products, stable logits, activation and categorical-loss finite differences, all dense weight/bias/input finite differences, SGD, causal attention and constant-row normalization.

Not implemented: broadcasting/views, autodiff graph, normalization/attention backward, multi-head projections, full Transformer blocks, dropout, Adam, serialization, GPU or parallel training. Existing logger is global and not thread-safe; initialization RNG is noncryptographic. Separate mutable objects per thread.
