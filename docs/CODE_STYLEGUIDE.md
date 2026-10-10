# CFWx code style guide

## Scope and formatting
All project-owned C sources and headers, including tests and compatibility forwarding files, follow `.clang-format` (clang-format 22.1.0). Use four spaces, LF, no tabs, a final newline and a 100-column soft limit. Allman braces start on a separate line. Never collapse functions, blocks, loops or guards into one line. Put one statement per line and separate validation, computation, output assignment and return with blank lines. Short unbraced guards/loops are allowed, but their controlled statement goes on the next line. Preserve include order/grouping; pointer stars attach to variables (`double *value`).

```sh
python -m pip install clang-format==22.1.0
python scripts/apply_code_style.py
```

The maintenance script applies the format, purpose/algorithm comments and logger build dependencies idempotently. The maintenance workflow accepts only the explicit `apply-cfwx-code-style` dispatch for `feature/implement_basics`; it builds/tests before committing formatting. It does not run untrusted pull-request code.

## Naming, types and numerical contracts
Use snake_case and the `cfwx_` prefix for public numerical APIs; preserve existing legacy interfaces. Use `size_t` for indices/dimensions, explicit enum status comparisons (`status != CFWX_OK`) and floating-point constants (`0.0`). Validate before dereferencing, indexing or allocation. Formatting must not remove or reorder numerical failure checks. In particular, retain cross-entropy's non-finite loss check before mutating the optional gradient.

Owners begin as `{0}`. Headers document ownership, borrowed lifetimes, dimensions, aliasing, reduction and failure behavior. Constructors commit only on success. Errors may partially modify numerical outputs unless documented otherwise. Numerical kernels remain independent of application logging: they return statuses and the experiment logs the failure with its injected context.

## Comments and readability
Each C file/header starts with its purpose. Header comments state public contracts; implementation comments explain algorithms, stability transformations, ownership, cleanup and test invariants. Prefer explaining *why* over narrating assignments. Keep comments accurate when algorithms change. Preserve license/legal text and data. CMake uses four-space indentation and comments describing dependencies; retain vcpkg/cmocka discovery, module directories and custom targets. JSON, configuration, logs and legal text do not follow C brace rules and are not blindly reformatted.

Example:

```c
/* Shifted log-sum-exp avoids overflow in the exponential terms. */
double max = x[0];
for (size_t i = 0; i < n; i++)
{
    if (!isfinite(x[i]))
        return CFWX_NUMERIC;

    if (x[i] > max)
        max = x[i];
}
```

## Injected experiment logging
Every experiment runner accepts `const experiment_logger *logger`. The caller provides a nonempty experiment name, minimum level, and optional sink/user data. The experiment borrows the context and must not open/close the backend or terminate the process. Pass the same context explicitly to experiment helpers that log; avoid hidden experiment globals.

```c
experiment_logger logger;
if (experiment_logger_init(&logger, "numerical.softmax", LOG_DEBUG, NULL, NULL) != 0)
{
    return EXIT_FAILURE;
}
int result = numerical_runner(&logger);
```

Names and user data must outlive logger calls. A custom sink receives complete formatted messages valid only during the callback and must copy retained data. A NULL sink uses the console/file backend with the experiment name on every line. Each experiment threshold is independent of the global application threshold. Formatting/allocation failures drop a log message but never change numerical results.

- **DEBUG:** inputs, tolerances, intermediate/reference values; potentially verbose.
- **INFO:** start, milestones and successful completion.
- **WARNING:** recoverable anomaly or degraded operation; do not invent warnings on healthy runs.
- **ERROR:** failed operation/check; return failure and let the caller clean up.
- **FATAL:** the application cannot initialize/continue; logging does not implicitly exit or abort.

Use `EXP_LOG_DEBUG`, `EXP_LOG_INFO`, `EXP_LOG_WARNING`, `EXP_LOG_ERROR` and `EXP_LOG_FATAL` with the context. Never use arbitrary input as a printf format string or log secrets/entire datasets. The legacy global FILE/localtime backend is not thread-safe. Injection alone does not make it thread-safe; custom sink synchronization belongs to the caller.

## Validation
Run numerical and logger tests in Debug and Release, plus ASan/UBSan builds. Run cmocka tests when available. Verify experiment names, formatted arguments, all levels, independent thresholds, invalid contexts, and formatter idempotence. State explicitly if an actual vcpkg toolchain or other dependency could not be tested. Generated runtime logs are never committed as formatting changes.
