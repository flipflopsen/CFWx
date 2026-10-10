/**
 * @file include/experiments/numerical_runner.h
 * @brief numerical runner interfaces and implementation.
 * @ingroup experiments
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Numerical experiment entry point and injected logger lifetime contract. */

#ifndef CFWX_NUMERICAL_RUNNER_H
#define CFWX_NUMERICAL_RUNNER_H

#include "util/logger.h"

/** The logger is borrowed. Invalid/NULL contexts fail; the caller owns backend
 * initialization and cleanup. The runner returns EXIT_SUCCESS/EXIT_FAILURE. */
/**
 * @anchor api_include_experiments_numerical_runner_h_numerical_runner
 * @brief Perform the numerical runner operation.
 * @ingroup experiments
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * The logger is borrowed. Invalid/NULL contexts fail; the caller owns backend initialization and cleanup. The runner returns EXIT_SUCCESS/EXIT_FAILURE.
 * @param logger Borrowed initialized experiment logger; name/user data must outlive all calls.
 * @return Computed value or status described by the adjacent contract.
 */
int numerical_runner(const experiment_logger *logger);

#endif
