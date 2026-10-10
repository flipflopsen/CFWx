# Developer workflow and contribution guide

## Reading order
[Setup](setup.md) -> [architecture](../ARCHITECTURE.md) -> [code style](../CODE_STYLEGUIDE.md) -> the relevant API header -> generated Doxygen file/module page. Mathematical correctness, memory safety, runtime reproducibility and performance are separate acceptance criteria.

## Change boundaries
Public interfaces belong in `include/`; numerical implementations in corresponding `src/` modules. Model kernels return status codes and do not own logger/CLI state. Experiments orchestrate datasets/models and borrow `experiment_logger`. Tests exercise scientific invariants and failure paths. CMake changes are additive: preserve original package integration and custom targets.

Every Markdown documentation file belongs under `docs/`, including overview/readme documents. Do not regenerate a root README.md or move Doxygen `.dox` inputs back into docs/. Doxygen assets belong under `doxygen_docs/`, while generated HTML/XML are ignored under `doxygen_docs/generated/`.

## Ownership and APIs
Document shapes, borrowed/owned lifetimes, exact/partial aliasing, reduction, RNG consumption and failed-output mutation. Owners are zero-initialized and matching free resets them. New backward APIs must propagate every trainable parameter/input path; use finite differences on nontrivial non-square shapes, not only identity tests. Check fully masked/ignored inputs, tiny variances, extreme logits, overflow, zero/invalid dimensions and allocation cleanup.

For optimizer code validate shared step counting, clipping order, bias correction, decay policy and checkpoint/restored state. For cache code check exact exhaustion, capacity one, sliding wraps, absolute positions, rollback/reset and model invalidation. Do not claim truncated-context recomputation equals incremental sliding caching.

## Formatting and documentation
Use repository `.clang-format` and Doxygen conventions. Existing source maintenance scripts use conservative text transforms, not a C/C++ parser. Run them only intentionally, inspect the complete diff and confirm no executable semantics changed. Never rerun historical migration scripts just to format a file.

```sh
find include src test -type f \( -name '*.h' -o -name '*.c' -o -name '*.cu' \) \
  -exec clang-format -i {} +
python scripts/check_docs_layout.py
doxygen doxygen_docs/Doxyfile
python scripts/check_doxygen.py
```

Doxygen warning logs are not hidden. Header contracts are canonical; type-aware generic internal comments are not enough for new scientific operations. Add precise formulas and assumptions. New source/group/page links should be checked in HTML, not merely counted in XML.

## Validation checklist
1. Build/test Debug and Release using a fresh CPU-only tree.
2. Run ASan/UBSan and appropriate memory/failure tests.
3. Validate analytic gradients with centered finite differences over parameters and inputs.
4. Run lifecycle round-trip/resume/cache tests for stateful changes.
5. Check Markdown containment/links and regenerate Doxygen.
6. Review git diff for artifacts, logs, checkpoints, unintended text-transform changes and secrets.
7. Describe untested platforms/backends accurately in the commit/PR.

CUDA code remains manual-only by user instruction. Do not add CUDA execution to CPU/documentation CI. A GPU change can be source-reviewed and documented without being compiled locally; clearly state that boundary. cuBLAS/Tensor Core/FlashAttention performance requires actual local manual benchmarks and numeric comparisons. CPU pass results never establish CUDA correctness.

## Source layout and maintenance workflows
The project includes historical scripts for target registration and source migration. These are repository maintenance tools, not required at program runtime. Read their guards and run only the intended current operation. Documentation workflows must use the canonical paths and must not recreate deleted root Markdown. CI source-write jobs can race collaborator pushes: never force-push automatically.

## Reproducibility and scientific reporting
Record compiler/backend/precision, dataset split/fingerprint/tokenizer mapping, architecture, seed/RNG policy, optimizer/scheduler/decay groups, checkpoint step and evaluation token count. Report limitations of synthetic corpora and overlap. Separate pre-update batch loss, complete validation loss and generative quality. Atomic fused reductions can change floating-point ordering. File durability/checksum is not authenticity or distributed writer coordination.
