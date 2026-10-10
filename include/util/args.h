/**
 * @file include/util/args.h
 * @brief args interfaces and implementation.
 * @ingroup utilities
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Public contracts for args. */

#ifndef ARGS_H
#define ARGS_H

#include <stdbool.h>
#include <getopt.h>

/** Max size of a file name */
/** @def FILE_NAME_SIZE
 * @brief file name size macro; see expansion and module contract.
 */
#define FILE_NAME_SIZE 512

/** Defines the command line allowed options struct */
struct options
{
    bool help;
    bool version;
    bool use_colors;
    char file_name[FILE_NAME_SIZE];
};

/** Exports options as a global type */
/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef struct options options_t;

/** Public functions section */
/**
 * @anchor api_include_util_args_h_options_parser
 * @brief Perform the options parser operation.
 * @ingroup utilities
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Public functions section
 * @param argc Number of process arguments including argv[0].
 * @param argv Process argument strings; the legacy parser may terminate on help/error.
 * @param options Caller-owned command-line option record.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void options_parser(int argc, char *argv[], options_t *options); /**< Borrowed pointer or owned storage as specified by the containing type. */

#endif // ARGS_H
