/**
 * @file test/train_transformer.c
 * @brief train transformer interfaces and implementation.
 * @ingroup tests
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Standalone runner bypasses the legacy CLI parser; main startup remains unchanged. */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include "experiments/training_runner.h"
/**
 * @anchor api_test_train_transformer_c_main
 * @brief Run this executable driver and propagate its success/failure status.
 * @ingroup tests
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param argc Number of process arguments including argv[0].
 * @param argv Process argument strings; the legacy parser may terminate on help/error.
 * @return Computed value or status described by the adjacent contract.
 */
int main(int argc, char **argv)
{
    if (argc > 3)
    {
        fprintf(stderr, "Usage: %s [dataset-file [steps]]\n", argv[0]);
        return EXIT_FAILURE;
    }
    size_t steps = 200;
    if (argc == 3)
    {
        char *end;
        errno = 0;
        unsigned long value = strtoul(argv[2], &end, 10);
        if (errno || !argv[2][0] || *end || argv[2][0] == '-' || value == 0 || value > 100000)
            return EXIT_FAILURE;
        steps = (size_t)value;
    }
    experiment_logger logger;
    if (experiment_logger_init(&logger, "transformer.training", LOG_INFO, NULL, NULL) != 0)
        return EXIT_FAILURE;
    return transformer_training_runner(&logger, argc >= 2 ? argv[1] : NULL, steps);
}
