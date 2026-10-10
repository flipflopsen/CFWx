## @file scripts/integrate_runtime.py
# @brief integrate runtime source/document maintenance tool.
# @ingroup maintenance
# @details Textual maintenance only; does not itself validate numerical correctness.

"""Idempotently attach stored dropout masks to analytic training tape and add targets."""
from pathlib import Path
p=Path('src/models/training.c');s=p.read_text()
if 'cfwx_language_model_loss_training' not in s:
 s=s.replace('#include "models/training.h"','#include "models/training.h"\n#include "models/runtime.h"')
 s=s.replace('    EMBED\n','    EMBED,\n    DROPOUT\n')
 s=s.replace('    double epsilon;','    double epsilon;\n    double *dropout_mask;')
 s=s.replace('    cfwx_status status;','    cfwx_status status;\n    double dropout;\n    uint64_t *rng;')
 s=s.replace('        free(t->nodes[i]);','        free(t->nodes[i]->dropout_mask);\n        free(t->nodes[i]);')
 marker='static node *block('
 index=s.index(marker)
 s=s[:index]+'''/* Store multipliers on the actual graph: backward must reuse the forward mask. */
static node *training_dropout(tape *t, node *x)
{
    if (!x || t->dropout == 0.0) return x;
    node *n = allocate(t, x->value.rows, x->value.cols, DROPOUT);
    if (!n) return NULL;
    n->a = x;
    n->dropout_mask = calloc(x->value.rows * x->value.cols, sizeof(double));
    if (!n->dropout_mask) { t->status = CFWX_ALLOC; return NULL; }
    cfwx_matrix mask = {x->value.rows, x->value.cols, n->dropout_mask};
    t->status = cfwx_dropout(&x->value, t->dropout, true, t->rng, &n->value, &mask);
    return n;
}

'''+s[index:]
 old='''    node *residual = binary(
        t, x, multihead(t, ln1, ln1, &layer->attention, g ? &g->attention : NULL, mask, causal),
        ADD);'''
 new='''    node *attention_branch = multihead(t, ln1, ln1, &layer->attention,
                                      g ? &g->attention : NULL, mask, causal);
    node *residual = binary(t, x, training_dropout(t, attention_branch), ADD);'''
 if old not in s:raise RuntimeError('block attention match failed')
 s=s.replace(old,new)
 old='''    return binary(t, residual, feedforward(t, ln2, &layer->feedforward, g ? &g->feedforward : NULL),
                  ADD);'''
 new='''    node *feedforward_branch = feedforward(t, ln2, &layer->feedforward,
                                         g ? &g->feedforward : NULL);
    return binary(t, residual, training_dropout(t, feedforward_branch), ADD);'''
 if old not in s:raise RuntimeError('block FFN match failed')
 s=s.replace(old,new)
 s=s.replace('    if (n->op == LEAF)\n        return;','''    if (n->op == LEAF)
        return;
    if (n->op == DROPOUT)
    {
        for (size_t i = 0; i < rows * cols; i++)
            a->grad.data[i] += n->grad.data[i] * n->dropout_mask[i];
        return;
    }''')
 start=s.index('cfwx_status cfwx_language_model_loss(')
 old=s[start:]
 old=old.replace('cfwx_status cfwx_language_model_loss(', 'cfwx_status cfwx_language_model_loss_training(',1)
 old=old.replace('size_t length, double *loss,','size_t length, double dropout, uint64_t *rng, double *loss,',1)
 old=old.replace('    if (!targets || !loss)','    if (!targets || !loss || !isfinite(dropout) || dropout < 0.0 || dropout >= 1.0 ||\n        (dropout > 0.0 && !rng))',1)
 old=old.replace('    tape t = {0};','    tape t = {0};\n    t.dropout = dropout;\n    t.rng = rng;',1)
 s=s[:start]+old+'''
cfwx_status cfwx_language_model_loss(const cfwx_language_model *model,
    const size_t *tokens, const size_t *targets, size_t length, double *loss,
    cfwx_language_model *gradient)
{
    return cfwx_language_model_loss_training(model, tokens, targets, length,
                                            0.0, NULL, loss, gradient);
}
'''
 p.write_text(s)
p=Path('CMakeLists.txt');s=p.read_text()
if 'NAME transformer_runtime' not in s:
 s+='''
# Lifecycle regressions and extended training/generation driver.
add_executable(cfwx_run test/runtime_driver.c)
target_link_libraries(cfwx_run PRIVATE cfwx_core cfwx_logger)
if(BUILD_TESTING)
    add_executable(cfwx_runtime_test_runner test/runtime_test.c)
    target_link_libraries(cfwx_runtime_test_runner PRIVATE cfwx_core)
    add_test(NAME transformer_runtime COMMAND cfwx_runtime_test_runner)
endif()
if(CFWX_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    foreach(target cfwx_run cfwx_runtime_test_runner)
        if(TARGET ${target})
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS " -fsanitize=address,undefined")
        endif()
    endforeach()
endif()
'''
 p.write_text(s)
