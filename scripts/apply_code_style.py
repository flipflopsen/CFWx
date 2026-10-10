## @file scripts/apply_code_style.py
# @brief apply code style source/document maintenance tool.
# @ingroup maintenance
# @details Textual maintenance only; does not itself validate numerical correctness.

"""Idempotent project-owned C formatting/comment maintenance. Run at repository root."""
from pathlib import Path
import re
import subprocess

COMMENTS = {
    "cfwx_vector_dot": "Kahan correction reduces low-order rounding loss in summation.",
    "layout": "Reverse dimension traversal constructs contiguous row-major strides.",
    "cfwx_tensor_init": "Temporary metadata leaves ownership unchanged on allocation failure.",
    "cfwx_tensor_reshape": "Reshape preserves the allocation and total element count.",
    "cfwx_sigmoid": "Choose the exponent sign to avoid overflow for extreme inputs.",
    "cfwx_gelu": "Exact erf-based GELU; handle infinite limits before multiplying by zero.",
    "cfwx_softmax": "A common logit shift preserves probabilities and avoids exponential overflow.",
    "cfwx_mse": "Mean scalar-element reduction also determines the gradient scaling.",
    "cfwx_cross_entropy_logits": "Shifted log-sum-exp is stable; gradient is softmax minus one-hot.",
    "cfwx_bce_logits": "Stable softplus handles large logits; fractional targets are permitted in [0,1].",
    "cfwx_discounted_returns": "Reverse traversal supports in-place returns with zero terminal bootstrap.",
    "rng": "SplitMix64 uses local state instead of the process-global random generator.",
    "cfwx_dense_init": "Xavier uniform variance is 2/(fan_in + fan_out); biases start at zero.",
    "cfwx_dense_backward": "Overwrite dX=dY*W^T, dW=X^T*dY and db=sum(dY); loss controls reduction.",
    "cfwx_sgd_step": "Prevalidate all updates to avoid partially applied parameter mutations.",
    "cfwx_attention": "Reuse one score row; causal masking allows key indices <= query index.",
    "cfwx_layer_norm": "Welford accumulation computes population variance without subtracting large moments.",
    "containers": "Verify ownership reset, overflow checks, reference products and tensor bounds.",
    "maths": "Check extreme logits and derivatives against centered finite differences.",
    "objective": "A linear output objective isolates the dense vector-Jacobian product.",
    "dense": "Perturb weights, biases and inputs to check all backward gradients.",
    "attention": "Check attention reference values, causal zeros and constant-row normalization.",
}
paths = sorted(p for root in ("include", "src", "test") for p in Path(root).rglob("*")
               if p.suffix in (".c", ".h"))
for p in paths:
    s = p.read_text()
    if not s.startswith("/*"):
        purpose = "Public contracts for " if p.suffix == ".h" else "Implementation and checks for "
        purpose += p.stem.replace("_", " ") + "."
        if p.parent == Path("include/maths") and p.suffix == ".c":
            purpose = "Compatibility forwarding file; new code should include the corresponding .h header."
        s = "/* " + purpose + " */\n\n" + s
    if str(p).startswith(("src/datastructures/", "src/maths/", "src/models/")):
        s = re.sub(r"if\s*\(\s*s\s*\)", "if (s != CFWX_OK)", s)
        s = re.sub(r"if\s*\(\s*!s\s*\)", "if (s == CFWX_OK)", s)
        s = re.sub(r"(double\s+[A-Za-z_]\w*\s*=\s*)0\b(?![.\w])", r"\g<1>0.0", s)
    p.write_text(s)
subprocess.run(["clang-format", "-i", *map(str, paths)], check=True)
for p in paths:
    s = p.read_text()
    for name, comment in COMMENTS.items():
        # Normalize comment whitespace because clang-format can wrap it.
        if comment in " ".join(s.split()):
            continue
        pattern = r"(\b" + re.escape(name) + r"\([^;{}]*\)\n\{\n)"
        s = re.sub(pattern, lambda m: m[1] + "    /* " + comment + " */\n\n", s)
    s = re.sub(r"\n}\n(?=[A-Za-z_])", "\n}\n\n", s)
    s = re.sub(r'(#include[^\n]*\n)(?=(?:static|cfwx_status|double|typedef)\b)', r"\1\n", s)
    p.write_text(s)
subprocess.run(["clang-format", "-i", *map(str, paths)], check=True)

p = Path("CMakeLists.txt")
s = p.read_text()
if "add_library(cfwx_logger" not in s:
    s = s.replace("# Main executable\n", """# Logging stays separate from backend-independent numerical kernels.
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
    s = s.replace("# Custom targets for additional functionality", """# Verify logger injection even without cmocka.
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
INFO records lifecycle/results, DEBUG records reference values and ERROR records failed checks.
All five levels have independent per-context thresholds and an optional custom sink.
See [CODE_STYLEGUIDE.md](CODE_STYLEGUIDE.md) and `.clang-format`.
The legacy FILE/localtime backend remains single-threaded.
"""
p.write_text(s)
