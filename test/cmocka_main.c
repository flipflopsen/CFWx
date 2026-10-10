/**
 * @file test/cmocka_main.c
 * @brief cmocka main interfaces and implementation.
 * @ingroup tests
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** cmocka smoke tests for injected experiments and stable extreme logits. */

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <setjmp.h>
#include <stdlib.h>
#include <cmocka.h>

#include "experiments/numerical_runner.h"
#include "maths/activation_functions.h"

/**
 * @anchor api_test_cmocka_main_c_numerical_experiment_success
 * @brief Perform the numerical experiment success operation.
 * @ingroup tests
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param state Experiment state including vocabulary, RNG, dataset fingerprint and hyperparameters.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
static void numerical_experiment_success(void **state)
{
    (void)state;
    experiment_logger logger;
    assert_int_equal(
        experiment_logger_init(&logger, "test.numerical.softmax", LOG_WARNING, NULL, NULL), 0);
    assert_int_equal(numerical_runner(&logger), EXIT_SUCCESS);
    assert_int_equal(numerical_runner(NULL), EXIT_FAILURE);
}

/**
 * @anchor api_test_cmocka_main_c_softmax_extreme_logits
 * @brief Perform the softmax extreme logits operation.
 * @ingroup tests
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param state Experiment state including vocabulary, RNG, dataset fingerprint and hyperparameters.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
static void softmax_extreme_logits(void **state)
{
    (void)state;
    const double logits[] = {1000.0, 1001.0, 1002.0};
    double probabilities[3];
    assert_int_equal(cfwx_softmax(logits, 3, probabilities), CFWX_OK);
    assert_float_equal(probabilities[2], 0.6652409557748218, 1e-12);
    assert_float_equal(probabilities[0] + probabilities[1] + probabilities[2], 1.0, 1e-12);
}

/**
 * @anchor api_test_cmocka_main_c_main
 * @brief Run this executable driver and propagate its success/failure status.
 * @ingroup tests
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @return Computed value or status described by the adjacent contract.
 */
int main(void)
{
    const struct CMUnitTest tests[] = {cmocka_unit_test(numerical_experiment_success),
                                       cmocka_unit_test(softmax_extreme_logits)};
    return cmocka_run_group_tests(tests, NULL, NULL);
}
