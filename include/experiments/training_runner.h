/* Train a tiny byte language model with an injected experiment logger. */
#ifndef CFWX_TRAINING_RUNNER_H
#define CFWX_TRAINING_RUNNER_H
#include <stddef.h>
#include "util/logger.h"
/* NULL dataset selects deterministic built-in corpus. External file is binary
 * byte-tokenized and limited to 1 MiB. steps must be positive. */
int transformer_training_runner(const experiment_logger *logger, const char *dataset, size_t steps);
#endif
