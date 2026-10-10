/**
 * @file src/experiments/numerical_runner.c
 * @brief numerical runner interfaces and implementation.
 * @ingroup experiments
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Softmax experiment: independent reference values and normalization checks. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "experiments/numerical_runner.h"
#include "maths/activation_functions.h"

/**
 * @anchor api_src_experiments_numerical_runner_c_numerical_runner
 * @brief Perform the numerical runner operation.
 * @ingroup experiments
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see numerical_runner include/experiments/numerical_runner.h
 */
int numerical_runner(const experiment_logger *logger)
{
    /* Do not construct or take ownership of the experiment's logger. */
    if (!logger || !logger->experiment_name || !logger->experiment_name[0] ||
        logger->minimum_level < LOG_DEBUG || logger->minimum_level > LOG_FATAL)
        return EXIT_FAILURE;

    EXP_LOG_INFO(logger, "Starting softmax reference experiment");
    EXP_LOG_DEBUG(logger, "Input logits: [1.0, 2.0, 3.0]; tolerance: 1e-12");

    const double logits[] = {1.0, 2.0, 3.0};
    const double expected[] = {0.09003057317038046, 0.24472847105479764, 0.6652409557748218};
    double probabilities[3];
    cfwx_status status = cfwx_softmax(logits, 3, probabilities);
    if (status != CFWX_OK)
    {
        EXP_LOG_ERROR(logger, "Softmax evaluation failed: status %d", (int)status);
        return EXIT_FAILURE;
    }

    /* Check each reference independently; a sum of one alone is insufficient. */
    double sum = 0.0;
    for (size_t i = 0; i < 3; i++)
    {
        if (!isfinite(probabilities[i]) || probabilities[i] < 0.0 || probabilities[i] > 1.0 ||
            fabs(probabilities[i] - expected[i]) > 1e-12)
        {
            EXP_LOG_ERROR(logger, "Softmax reference check failed at index %zu", i);
            return EXIT_FAILURE;
        }

        EXP_LOG_DEBUG(logger, "probability[%zu]=%.17g; expected=%.17g", i, probabilities[i],
                      expected[i]);
        sum += probabilities[i];
    }
    if (fabs(sum - 1.0) > 1e-12)
    {
        EXP_LOG_ERROR(logger, "Softmax normalization check failed: sum=%.17g", sum);
        return EXIT_FAILURE;
    }

    EXP_LOG_INFO(logger, "PASS softmax reference and normalization checks");
    printf("CFWx numerical experiment\n"
           "softmax([1,2,3]) = [%.6f, %.6f, %.6f]\n"
           "PASS softmax reference and normalization checks\n",
           probabilities[0], probabilities[1], probabilities[2]);

    return EXIT_SUCCESS;
}
