# CFWx repository architecture

## Purpose
An educational C17 numerical and Transformer playground. Public numerical APIs are independent of logging and return `cfwx_status`. Experiments orchestrate data, model lifecycle, diagnostics and validation using injected `experiment_logger` contexts. The original application startup, vcpkg/cmocka integration and custom CMake targets are preserved.

## Repository map

| Location | Responsibility |
|---|---|
| `include/cfwx.h` | Shared statuses and allocation-count overflow checks |
| `include/datastructures/`, `src/datastructures/` | Owning contiguous vectors, matrices, tensors and affine neurons |
| `include/maths/`, `src/maths/` | Activations/derivatives, stable losses, softmax and discounted returns |
| `models/ann.h`, `src/models/ann.c` | Dense forward/backward, reproducible initialization, SGD |
| `models/attention.h`, `src/models/attention.c` | Raw head attention and LayerNorm forward |
| `models/transformer.h`, `src/models/transformer.c` | Modular forward composition: positions, embeddings, masks, projections, FFNs, residual blocks |
| `models/training.h`, `src/models/training.c` | Analytic reverse-mode tape, composite backward, parameter registries, AdamW, clipping, causal LM |
| `include/util/`, `src/util/` | Original argument parsing/messages; backend logging and named contexts |
| `include/experiments/`, `src/experiments/` | Softmax, causal-LM wiring, cross-attention and actual dataset-training runners |
| `test/` | Dependency-free numerical/regression/gradient tests, cmocka smoke tests, example/training executable drivers |
| `scripts/` | Idempotent source formatting and target-registration maintenance |
| `cmake/` | Safe template-copy custom target script |
| `.github/workflows/` | Trusted-branch style/integration, build/test/sanitizer validation |
| `.vscode/` | User-specific editor/debug configuration |
| `CODE_STYLEGUIDE.md`, `.clang-format` | Allman formatting, ownership and logging conventions |
| `TRANSFORMER_BLOCKS.md`, `TRAINING.md` | Block contracts and actual dataset training instructions |

Header references in the table under `models/` mean `include/models/`. Existing `.c` forwarding files under `include/maths/` are compatibility artifacts; include `.h` in new code. `log/`, `bin/` and build directories contain generated runtime/build artifacts, not source modules. Do not commit generated logs; some original logs remain historically tracked.

## Dependency graph

```text
cfwx_core
  datastructures -> maths -> dense/head kernels -> Transformer forward composition
                                                   |
                                                   +-> analytic training tape / AdamW / LM

cfwx_logger
  process backend + experiment contexts / custom sink

application and example/training drivers
  -> experiment runners -> cfwx_core
                        -> cfwx_logger
```

Numerical kernels never depend on the logger or CLI parser. Experimental code logs returned errors, retaining status propagation and cleanup. CLI process termination remains confined to the legacy argument/message layer. The standalone `cfwx_train` driver avoids the original parser's known long-option defect.

## Forward and reverse execution
Model owners hold persistent parameter arrays. Forward composites allocate temporary row-major matrices, call explicit kernels and release temporaries. Training reconstructs the same mathematical graph as a local tape of owned nodes; each node records parent pointers, values and adjoints. Nodes are reverse-traversed, adding adjoints into parents so shared branches and residual connections work correctly. Leaf adjoints are copied into caller-owned gradient models only after numerical validation. The tape is released after each call.

Operations: leaves, addition/product, matrix multiplication, bias broadcast, activation, normalization, attention, head slice/join and embedding gather. Parameter enumeration orders the model/gradient registries identically. Clipping and AdamW operate on borrowed registry storage. The optimizer separately owns first/second moments and a shared update step.

The LM explicitly combines trainable embeddings, fixed sinusoidal positions, one or more causal pre-LN blocks, a final learned LayerNorm and an untied vocabulary head. All trainable leaves receive analytic gradients; no finite differences are used by the trainer.

## Ownership and shape rules
- Owners are zero-initialized; matching `_free` resets them. Do not shallow-copy live owners.
- Matrices are positive-size contiguous row-major double arrays; tensors have explicit shape/stride metadata.
- Sequence layout is `[tokens,features]`; batches are handled by callers, not implicit 3-D views.
- Outputs and gradients are caller-allocated and shape checked. Documented in-place forward primitives are exceptions to the default no-alias contract.
- Model and gradient owners must be architecture-matched and disjoint. Registries borrow their storage.
- Experiment names and custom sink userdata are borrowed. The sink message is valid during the callback only.
- Local RNG makes initialization/dropout reproducible; the global FILE/localtime logger backend is not thread-safe.

## Build targets
`cfwx_core` and `cfwx_logger` are static libraries. `cfwx` retains original startup; `cfwx_train` accepts a dataset path/step count. With testing enabled, numerical, logger, Transformer forward/examples, analytic-gradient and training regressions are registered with CTest. cmocka additionally registers `unit_tests` when available. Test checks remain active under NDEBUG.

CMake still defaults to vcpkg; pass `CFWX_USE_VCPKG=OFF` in environments without it. Debug/Release target output paths are under the corresponding build directory/configuration. Custom targets `welp`, `start`, `clear` and optional `valgrind` remain. Style/integration scripts modify relevant sources/target lists idempotently and leave generated logs untouched.

## Extension points
Add a new numerical primitive with a documented forward contract, analytic vector-Jacobian product and finite-difference test. To integrate it into composite training, add a tape operation or compose existing differentiable operations. Add dataset experiments as named-logger runners rather than modifying kernels or replacing startup.

Priorities beyond this implementation: cached forward workspaces, token-weighted mini-batches, full held-out evaluation, tokenizer/vocabulary persistence, parameter/optimizer checkpoints, scheduler, generation sampling and KV cache. These are not silently claimed as implemented.

## Runtime lifecycle update
The previous no-checkpoint, single-window, dropout-disabled and no-KV-cache
limitations are superseded by [RUNTIME.md](RUNTIME.md) and `models/runtime.h`.
Use `cfwx_run` for dropout-enabled mini-batch training, complete held-out evaluation,
checkpoint/resume and incremental cached generation. Earlier descriptions refer to
the original `cfwx_train` runner. CPU micro-batch accumulation is implemented;
fused batching and GPU execution remain explicitly unsupported rather than mislabeled.
