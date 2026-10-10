/**
 * @file include/experiments/transformer_runners.h
 * @brief transformer runners interfaces and implementation.
 * @ingroup experiments
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Named, caller-logged examples of Transformer building-block composition. */
#ifndef CFWX_TRANSFORMER_RUNNERS_H
#define CFWX_TRANSFORMER_RUNNERS_H
#include "util/logger.h"
/** Borrow initialized experiment contexts; return EXIT_* after releasing all
 * owned temporaries. Names are supplied by the caller, never hardcoded sinks. */
/**
 * @anchor api_include_experiments_transformer_runners_h_transformer_language_runner
 * @brief Perform the transformer language runner operation.
 * @ingroup experiments
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Borrow initialized experiment contexts; return EXIT_* after releasing all owned temporaries. Names are supplied by the caller, never hardcoded sinks.
 * @param logger Borrowed initialized experiment logger; name/user data must outlive all calls.
 * @return Computed value or status described by the adjacent contract.
 */
int transformer_language_runner(const experiment_logger *logger);
/**
 * @anchor api_include_experiments_transformer_runners_h_transformer_cross_attention_runner
 * @brief Perform the transformer cross attention runner operation.
 * @ingroup experiments
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param logger Borrowed initialized experiment logger; name/user data must outlive all calls.
 * @return Computed value or status described by the adjacent contract.
 */
int transformer_cross_attention_runner(const experiment_logger *logger);
#endif
