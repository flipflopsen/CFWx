/* Numerical experiment entry point and injected logger lifetime contract. */

#ifndef CFWX_NUMERICAL_RUNNER_H
#define CFWX_NUMERICAL_RUNNER_H

#include "util/logger.h"

/* The logger is borrowed. Invalid/NULL contexts fail; the caller owns backend
 * initialization and cleanup. The runner returns EXIT_SUCCESS/EXIT_FAILURE. */
int numerical_runner(const experiment_logger *logger);

#endif
