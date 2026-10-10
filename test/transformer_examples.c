/**
 * @file test/transformer_examples.c
 * @brief transformer examples interfaces and implementation.
 * @ingroup tests
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Example driver supplies identities and owns the borrowed logging contexts. */
#include "experiments/transformer_runners.h"
#include <stdlib.h>

/**
 * @anchor api_test_transformer_examples_c_main
 * @brief Run this executable driver and propagate its success/failure status.
 * @ingroup tests
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @return Computed value or status described by the adjacent contract.
 */
int main(void)
{
    experiment_logger language, cross;
    if (experiment_logger_init(&language, "transformer.causal_lm", LOG_DEBUG, NULL, NULL) != 0 ||
        experiment_logger_init(&cross, "transformer.cross_attention", LOG_DEBUG, NULL, NULL) != 0)
        return EXIT_FAILURE;
    if (transformer_language_runner(&language) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    return transformer_cross_attention_runner(&cross);
}
