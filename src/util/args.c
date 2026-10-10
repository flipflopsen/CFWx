/**
 * @file src/util/args.c
 * @brief args interfaces and implementation.
 * @ingroup utilities
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Implementation and checks for args. */

#include <stdlib.h>
#include <string.h>

#include "util/messages.h"
#include "util/args.h"

/**
 * @anchor api_src_util_args_c_set_default_options
 * @brief Perform the set default options operation.
 * @ingroup utilities
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param options Caller-owned command-line option record.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
static void set_default_options(options_t *options)
{
    options->help = false;
    options->version = false;
    options->use_colors = true;
}

/**
 * @anchor api_src_util_args_c_switch_options
 * @brief Perform the switch options operation.
 * @ingroup utilities
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param arg arg argument; interpretation follows this operation's implementation and contract.
 * @param options Caller-owned command-line option record.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void switch_options(int arg, options_t *options)
{
    switch (arg)
    {
    case 'h':
        options->help = true;
        help();
        exit(EXIT_SUCCESS);

    case 'v':
        options->version = true;
        version();
        exit(EXIT_SUCCESS);

    case 0:
        options->use_colors = false;
        break;

    case '?':
        usage();
        exit(EXIT_FAILURE);

    default:
        usage();
        abort();
    }
}

/**
 * Tries to get the file name. Otherwise, gets stdin
 */
/**
 * @anchor api_src_util_args_c_get_file_name
 * @brief Perform the get file name operation.
 * @ingroup utilities
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Tries to get the file name. Otherwise, gets stdin
 * @param argc Number of process arguments including argv[0].
 * @param argv Process argument strings; the legacy parser may terminate on help/error.
 * @param options Caller-owned command-line option record.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void get_file_name(int argc, char *argv[], options_t *options)
{

    /* If there is more arguments, probably, it is an input file */
    if (optind < argc)
    {
        strncpy(options->file_name, argv[optind++], FILE_NAME_SIZE - 1);

        /* Otherwise, assumes stdin as the input file */
    }
    else
    {
        strncpy(options->file_name, "-", FILE_NAME_SIZE - 1);
    }
}

/**
 * Public function that loops until command line options were parsed
 */
/**
 * @anchor api_src_util_args_c_options_parser
 * @brief Perform the options parser operation.
 * @ingroup utilities
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see options_parser include/util/args.h
 */
void options_parser(int argc, char *argv[], options_t *options)
{
    set_default_options(options);

    int arg;

    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"version", no_argument, 0, 'v'},
        {"no-colors", no_argument, 0, 0},
    };

    while (true)
    {
        int option_index = 0;
        arg = getopt_long(argc, argv, "hvt:", long_options, &option_index);

        if (arg == -1)
            break;

        switch_options(arg, options);
    }

    get_file_name(argc, argv, options);
}
