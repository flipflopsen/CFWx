"""Apply the documented C style and purpose/algorithm comments without changing algorithms.

Run from the repository root with clang-format 22.1.0 available on PATH.
This script is idempotent; it deliberately excludes logs, licenses and data.
"""
from pathlib import Path
import re
import subprocess

PURPOSE = {
    "include/cfwx.h": "Shared numerical statuses and overflow-safe allocation contracts.",
    "src/datastructures/vector.c": "Owning vector operations with compensated dot-product summation.",
    "src/datastructures/matrix.c": "Row-major matrix reference kernels with caller-owned outputs.",
    "src/datastructures/tensor.c": "Contiguous tensor storage and element-count-preserving metadata reshape.",
    "src/datastructures/neuron.c": "Affine neuron evaluation; activation selection belongs to the caller.",
    "src/maths/activation_functions.c": "Stable activations and max-shifted softmax for finite logits.",
    "src/maths/loss_functions.c": "Logit-based losses, explicit gradients and scalar-element reductions.",
    "src/maths/rewards.c": "Backward discounted-return recurrence with zero terminal bootstrap.",
    "src/models/ann.c": "Dense forward/backward kernels, Xavier initialization and SGD updates.",
    "src/models/attention.c": "Single-head attention and population layer normalization; no backward pass yet.",
    "src/util/args.c": "Legacy POSIX argument parser; help/version retain their process-exit behavior.",
    "src/util/messages.c": "CLI usage, help and version messages for the project template.",
    "test/main.c": "Numerical regression and finite-difference checks, active even under NDEBUG.",
}
ALGORITHMS = {
    "cfwx_matrix_init": "Validate the empty owner and allocation size before committing storage.",
    "cfwx_vector_init": "Construct storage before transferring ownership to the caller.",
    "cfwx_vector_dot": "Kahan correction reduces low-order rounding losses in the accumulated dot product.",
    "layout": "Traverse dimensions backwards to construct contiguous row-major strides.",
    "cfwx_tensor_init": "Use temporary metadata so a failed constructor leaves the owner unchanged.",
    "cfwx_tensor_reshape": "Preserve both the element count and the existing allocation.",
    "cfwx_sigmoid": "Choose the exponent's sign to avoid overflow for large positive/negative inputs.",
    "cfwx_gelu": "Exact erf-based GELU; handle infinite limits before multiplying by zero.",
    "cfwx_softmax": "Subtract the largest logit; probabilities are invariant to a common shift.",
    "cfwx_mse": "Mean reduction includes every scalar element; its gradient carries the same factor.",
    "cfwx_cross_entropy_logits": "Use shifted log-sum-exp; the categorical gradient is softmax minus one-hot.",
    "cfwx_bce_logits": "Stable softplus avoids exponential overflow; targets may be fractional in [0,1].",
    "cfwx_discounted_returns": "Reverse traversal supports exact in-place operation and zero terminal bootstrap.",
    "rng": "SplitMix64 uses caller-local state, not the process-global random generator.",
    "cfwx_dense_init": "Xavier uniform variance is 2/(fan_in + fan_out); biases start at zero.",
    "cfwx_dense_backward": "Overwrite dX=dY*W^T, dW=X^T*dY and db=sum(dY); loss controls batch reduction.",
    "cfwx_sgd_step": "Prevalidate every update before mutation to avoid partially applied SGD steps.",
    "cfwx_attention": "Reuse one score row; causal masking permits only key indices <= query index.",
    "cfwx_layer_norm": "Welford accumulation avoids subtracting large moments; variance divides by feature count.",
    "containers": "Exercise ownership reset, allocation overflow, matrix reference products and tensor bounds.",
    "maths": "Check stable extreme logits and derivatives against centered finite differences.",
    "objective": "A linear output objective isolates the dense layer's vector-Jacobian product.",
    "dense": "Perturb weights, biases and inputs independently to verify all backward gradients.",
    "attention": "Check uniform/nonuniform attention, causal zeros and constant-row normalization.",
}
paths = sorted(p for root in ("include", "src", "test") for p in Path(root).rglob("*")
               if p.suffix in (".c", ".h"))
for path in paths:
    text = path.read_text()
    if not text.startswith("/*"):
        purpose = PURPOSE.get(str(path), "Public declarations for " + path.stem.replace("_", " ") + ".")
        if path.parent == Path("include/maths") and path.suffix == ".c":
            purpose = "Compatibility forwarding file only; use the corresponding .h header in new code."
        text = "/* " + purpose + " */\n\n" + text

    # Numerical statuses are enums, not booleans. Legacy pointer tests are untouched.
    if str(path).startswith(("src/datastructures/", "src/maths/", "src/models/")):
        text = re.sub(r"if\s*\(\s*s\s*\)", "if (s != CFWX_OK)", text)
        text = re.sub(r"if\s*\(\s*!s\s*\)", "if (s == CFWX_OK)", text)
        # Declaration-only zero initializers; do not rewrite integer indexing/RNG constants.
        text = re.sub(r"(double\s+[A-Za-z_]\w*\s*=\s*)0\b(?![.\w])", r"\g<1>0.0", text)

    # Comment function bodies once. Run formatter first to make signatures unambiguous.
    path.write_text(text)
subprocess.run(["clang-format", "-i", *map(str, paths)], check=True)
for path in paths:
    text = path.read_text()
    for name, comment in ALGORITHMS.items():
        pattern = r"(\b" + re.escape(name) + r"\([^;{}]*\)\n\{\n)(?!    /\* " + re.escape(comment) + r" \*/)"
        text = re.sub(pattern, lambda m: m[1] + "    /* " + comment + " */\n\n", text)
    # Separate top-level functions and include/declaration sections for readability.
    text = re.sub(r"\n}\n(?=[A-Za-z_])", "\n}\n\n", text)
    text = re.sub(r'(#include[^\n]*\n)(?=(?:static|cfwx_status|double|typedef)\b)', r"\1\n", text)
    path.write_text(text)
subprocess.run(["clang-format", "-i", *map(str, paths)], check=True)

# Preserve CMake discovery/custom targets while injecting logger/test dependencies.
p = Path("CMakeLists.txt")
s = p.read_text()
if "add_library(cfwx_logger" not in s:
    s = s.replace("# Main executable\n", """# Separate logging from numerical kernels; experiments depend on both.
add_library(cfwx_logger STATIC ${SRCDIR}/util/logger.c)
target_include_directories(cfwx_logger PUBLIC "${CMAKE_SOURCE_DIR}/include")
list(REMOVE_ITEM SOURCES "${SRCDIR}/util/logger.c")

# Main executable
""")
    s = s.replace("target_link_libraries(${BINARY} PRIVATE cfwx_core)",
                  "target_link_libraries(${BINARY} PRIVATE cfwx_core cfwx_logger)")
    s = s.replace("PRIVATE cfwx_core ${CFWX_CMOCKA_TARGET}",
                  "PRIVATE cfwx_core cfwx_logger ${CFWX_CMOCKA_TARGET}")
    s = s.replace("foreach(target cfwx_core ${BINARY}", "foreach(target cfwx_core cfwx_logger ${BINARY}")
if "NAME experiment_logging" not in s:
    s = s.replace("# Custom targets for additional functionality", """# Verify named logger injection even when cmocka is not installed.
if(BUILD_TESTING)
    add_executable(${BINARY}_logger_test_runner ${TESTDIR}/logger_test.c ${SOURCES_EXPERIMENTS})
    target_link_libraries(${BINARY}_logger_test_runner PRIVATE cfwx_core cfwx_logger)
    add_test(NAME experiment_logging COMMAND ${BINARY}_logger_test_runner)
    if(CFWX_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${BINARY}_logger_test_runner PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        set_property(TARGET ${BINARY}_logger_test_runner APPEND_STRING PROPERTY LINK_FLAGS " -fsanitize=address,undefined")
    endif()
endif()

# Custom targets for additional functionality""")
p.write_text(s)
p = Path("README.md")
s = p.read_text().replace("numerical_runner()", "numerical_runner(&logger)")
s = s.replace("restored verbatim", "restored behavior (subsequently reformatted)")
if "## Named experiment logging" not in s:
    s += """
## Named experiment logging and code style
The application injects an `experiment_logger` named `numerical.softmax` into the runner.
INFO records lifecycle/results, DEBUG records reference/intermediate values, and ERROR records
failed checks. All five levels are supported, with independent per-context thresholds and an
optional custom sink for capture or alternate routing. See [CODE_STYLEGUIDE.md](CODE_STYLEGUIDE.md)
and `.clang-format`. The legacy FILE/localtime backend is not thread-safe.
"""
p.write_text(s)
