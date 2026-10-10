# CFWx
Educational C17 numerical foundations for ML/DL experiments. Not a production ML framework.

## Build and test (Linux)
The original vcpkg integration remains enabled by default, using `/opt/vcpkg/scripts/buildsystems/vcpkg.cmake`. An explicitly supplied `CMAKE_TOOLCHAIN_FILE` takes precedence. GNU C17 extensions, the original directory/module layout, compiler flags and Debug/Release output directories are retained.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/Debug/cfwx
# For environments without vcpkg: use system packages instead
cmake -S . -B build-system -DCFWX_USE_VCPKG=OFF
cmake --build build-system --parallel
ctest --test-dir build-system --output-on-failure
# Optional sanitizer build
cmake -S . -B build-asan -DCFWX_SANITIZERS=ON
cmake --build build-asan --parallel
ctest --test-dir build-asan --output-on-failure
```

Cmocka is discovered through its CMake configuration (vcpkg by default), with a system pkg-config fallback. When available, `cfwx_test_runner` executes the cmocka tests registered as `unit_tests`. The separate `cfwx_numerical_test_runner` retains the full numerical regression suite registered as `numerical_foundations`; it can run without cmocka. Missing cmocka generates an explicit warning. `BUILD_TESTING=OFF` disables both test executables.

Main retains the original argument parsing, debugger detection, logger initialization and cleanup. It delegates the experiment to `numerical_runner(&logger)` in `src/experiments/numerical_runner.c`, which checks softmax reference values and normalization and returns a process-compatible success/failure code. Runner failures are logged and propagated after cleanup. The original `args.c` has been restored behavior (subsequently reformatted); its pre-existing unterminated long-option table and potential filename-termination defect remain and should be addressed separately. Debugger detection also retains its original `PTRACE_TRACEME` side effects.

`PROJECT_PATH` is configurable and defaults to the checkout directory rather than the original placeholder, so logger initialization works without source edits. CLI uses POSIX getopt and Linux ptrace; the numerical core has no OS-specific calls. VSCode settings remain user-specific.

## Custom targets
- `welp`: prints target help.
- `start`: copies the template to `PROJECT_PATH`. Set this to an external destination; self-copy and descendant destinations are rejected. Generated build trees, `.git`, binaries and logs are excluded.
- `valgrind`: available when installed; runs the actual application target, writes `log/valgrind.log`, and reports memory-check failures through its exit status.
- `clear`: clears `bin` and `log`, recreating project directories. It deliberately does not delete the active build tree or `lib` contents. Use the generated `clean` target for build outputs, or remove the build directory from an external shell for a complete reset.

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
Numerical tests remain active under NDEBUG. Tests cover container operations, overflow/bounds, reference matrix products, stable logits, activation and categorical-loss finite differences, all dense weight/bias/input finite differences, SGD, causal attention and constant-row normalization. Cmocka additionally exercises the experiment runner and extreme-logit normalization.

Not implemented: broadcasting/views, autodiff graph, normalization/attention backward, multi-head projections, full Transformer blocks, dropout, Adam, serialization, GPU or parallel training. Existing logger is global and not thread-safe; initialization RNG is noncryptographic. Separate mutable objects per thread.

## Named experiment logging and code style
The application injects an `experiment_logger` named `numerical.softmax` into the runner.
INFO records lifecycle/results, DEBUG records reference values and ERROR records failed checks.
All five levels have independent per-context thresholds and an optional custom sink.
See [CODE_STYLEGUIDE.md](CODE_STYLEGUIDE.md) and `.clang-format`.
The legacy FILE/localtime backend remains single-threaded.

## End-to-end Transformer training
The repository now includes complete analytic attention, normalization and composite-block
backward passes, a stacked causal language model, global gradient clipping and AdamW.
Run `cfwx_train` for the built-in overfit regression or provide a small byte dataset.
See [TRAINING.md](TRAINING.md) for commands, reduction/tokenization semantics and limitations,
and [ARCHITECTURE.md](ARCHITECTURE.md) for the repository/dependency overview.
The previous forward-only limitation is superseded by `models/training.h`.

## Runtime lifecycle update
The previous no-checkpoint, single-window, dropout-disabled and no-KV-cache
limitations are superseded by [RUNTIME.md](RUNTIME.md) and `models/runtime.h`.
Use `cfwx_run` for dropout-enabled mini-batch training, complete held-out evaluation,
checkpoint/resume and incremental cached generation. Earlier descriptions refer to
the original `cfwx_train` runner. CPU micro-batch accumulation is implemented;
fused batching and GPU execution remain explicitly unsupported rather than mislabeled.

## Doxygen HTML reference
Run `doxygen Doxyfile` from the repository root and open `build-docs/html/index.html`.
The additive CMake `docs` target runs the same configuration. Documentation includes
module groups, public contracts, internal definitions, source browsing, model/state
records, tests and multi-page architecture/training/runtime/GPU design guides.
See `docs/doxygen/README.md`. CUDA is parsed as source only; no GPU build or test is
required or performed by documentation generation.
