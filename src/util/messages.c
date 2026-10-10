/**
 * @file src/util/messages.c
 * @brief messages interfaces and implementation.
 * @ingroup utilities
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Implementation and checks for messages. */

#include <stdio.h>

#include "util/colors.h"
#include "util/messages.h"

/**
 * Help message
 */
/**
 * @anchor api_src_util_messages_c_help
 * @brief Perform the help operation.
 * @ingroup utilities
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see help include/util/messages.h
 */
void help(void)
{
    fprintf(stdout, BLUE __PROGRAM_NAME__ "\n\n" NO_COLOR);
    usage();
    description();
    options();
    author();
    version();
}

/**
 * Usage message
 */
/**
 * @anchor api_src_util_messages_c_usage
 * @brief Perform the usage operation.
 * @ingroup utilities
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see usage include/util/messages.h
 */
void usage(void)
{
    fprintf(stdout, BROWN "Usage: " NO_COLOR);
    fprintf(stdout, "%s [options] input file\n\n", __PROGRAM_NAME__);
}

/**
 * Description message
 */
/**
 * @anchor api_src_util_messages_c_description
 * @brief Perform the description operation.
 * @ingroup utilities
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see description include/util/messages.h
 */
void description(void)
{
    fprintf(stdout, BROWN "Description: " NO_COLOR);
    fprintf(stdout, "Write here what you want to be your project description."
                    " Observe that you can break a string inside a fprintf\n");
}

/**
 * Options message
 */
/**
 * @anchor api_src_util_messages_c_options
 * @brief Perform the options operation.
 * @ingroup utilities
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see options include/util/messages.h
 */
void options(void)
{
    fprintf(stdout, BROWN "Options:\n\n" NO_COLOR);
    fprintf(stdout, GRAY "\t-v|--version\n" NO_COLOR "\t\tPrints %s version\n\n", __PROGRAM_NAME__);
    fprintf(stdout, GRAY "\t-h|--help\n" NO_COLOR "\t\tPrints this help message\n\n");
    fprintf(stdout, GRAY "\t--no-color\n" NO_COLOR "\t\tDoes not use colors for printing\n\n");
}

/**
 * Author message
 */
/**
 * @anchor api_src_util_messages_c_author
 * @brief Perform the author operation.
 * @ingroup utilities
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see author include/util/messages.h
 */
void author(void)
{
    fprintf(stdout, BROWN "Written by: " GRAY "%s\n\n" NO_COLOR, __PROGRAM_AUTHOR__);
}

/**
 * Version message
 */
/**
 * @anchor api_src_util_messages_c_version
 * @brief Perform the version operation.
 * @ingroup utilities
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see version include/util/messages.h
 */
void version(void)
{
    fprintf(stdout, __PROGRAM_NAME__ " version: " GRAY "%s\n" NO_COLOR, __PROGRAM_VERSION__);
}
