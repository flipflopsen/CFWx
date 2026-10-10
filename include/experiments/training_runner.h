/**
 * @file include/experiments/training_runner.h
 * @brief training runner interfaces and implementation.
 * @ingroup experiments
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Train a tiny byte language model with an injected experiment logger. */
#ifndef CFWX_TRAINING_RUNNER_H
#define CFWX_TRAINING_RUNNER_H
#include <stddef.h>
#include "util/logger.h"
/** NULL dataset selects deterministic built-in corpus. External file is binary
 * byte-tokenized and limited to 1 MiB. steps must be positive. */
/**
 * @anchor api_include_experiments_training_runner_h_transformer_training_runner
 * @brief Perform the transformer training runner operation.
 * @ingroup experiments
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * NULL dataset selects deterministic built-in corpus. External file is binary byte-tokenized and limited to 1 MiB. steps must be positive.
 * @param logger Borrowed initialized experiment logger; name/user data must outlive all calls.
 * @param dataset Optional binary byte-dataset path; NULL selects the built-in corpus where supported.
 * @param steps Positive number of requested optimizer updates.
 * @return Computed value or status described by the adjacent contract.
 */
int transformer_training_runner(const experiment_logger *logger, const char *dataset, size_t steps);
#endif
