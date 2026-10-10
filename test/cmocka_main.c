#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <setjmp.h>
#include <stdlib.h>
#include <cmocka.h>

#include "experiments/numerical_runner.h"
#include "maths/activation_functions.h"

static void numerical_experiment_success(void **state)
{
    (void)state;
    assert_int_equal(numerical_runner(), EXIT_SUCCESS);
}

static void softmax_extreme_logits(void **state)
{
    (void)state;
    const double logits[] = {1000.0, 1001.0, 1002.0};
    double probabilities[3];
    assert_int_equal(cfwx_softmax(logits, 3, probabilities), CFWX_OK);
    assert_float_equal(probabilities[2], 0.6652409557748218, 1e-12);
    assert_float_equal(probabilities[0] + probabilities[1] + probabilities[2], 1.0, 1e-12);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(numerical_experiment_success),
        cmocka_unit_test(softmax_extreme_logits)
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
