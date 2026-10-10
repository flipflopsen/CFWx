/* Named, caller-logged examples of Transformer building-block composition. */
#ifndef CFWX_TRANSFORMER_RUNNERS_H
#define CFWX_TRANSFORMER_RUNNERS_H
#include "util/logger.h"
/* Borrow initialized experiment contexts; return EXIT_* after releasing all
 * owned temporaries. Names are supplied by the caller, never hardcoded sinks. */
int transformer_language_runner(const experiment_logger *logger);
int transformer_cross_attention_runner(const experiment_logger *logger);
#endif
