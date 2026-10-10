#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "experiments/numerical_runner.h"
#include "maths/activation_functions.h"

int numerical_runner(void)
{
    const double logits[] = {1.0, 2.0, 3.0};
    const double expected[] = {
        0.09003057317038046,
        0.24472847105479764,
        0.6652409557748218
    };
    double probabilities[3];
    cfwx_status status = cfwx_softmax(logits, 3, probabilities);
    if (status != CFWX_OK)
    {
        fprintf(stderr, "Softmax experiment failed: status %d\n", (int)status);
        return EXIT_FAILURE;
    }

    double sum = 0.0;
    for (size_t i = 0; i < 3; ++i)
    {
        if (!isfinite(probabilities[i]) || probabilities[i] < 0.0 ||
            probabilities[i] > 1.0 || fabs(probabilities[i] - expected[i]) > 1e-12)
        {
            fprintf(stderr, "Softmax reference check failed at index %zu\n", i);
            return EXIT_FAILURE;
        }
        sum += probabilities[i];
    }
    if (fabs(sum - 1.0) > 1e-12)
    {
        fprintf(stderr, "Softmax normalization check failed\n");
        return EXIT_FAILURE;
    }

    printf("CFWx numerical experiment\n"
           "softmax([1,2,3]) = [%.6f, %.6f, %.6f]\n"
           "PASS softmax reference and normalization checks\n",
           probabilities[0], probabilities[1], probabilities[2]);
    return EXIT_SUCCESS;
}
