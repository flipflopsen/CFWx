/**
 * @file include/util/messages.h
 * @brief messages interfaces and implementation.
 * @ingroup utilities
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Public contracts for messages. */

#ifndef MESSAGES_H
#define MESSAGES_H

/** @def __PROGRAM_NAME__
 * @brief   program name   macro; see expansion and module contract.
 */
#define __PROGRAM_NAME__ "CFWx - C Framework"
/** @def __PROGRAM_VERSION__
 * @brief   program version   macro; see expansion and module contract.
 */
#define __PROGRAM_VERSION__ "0.0.1"
/** @def __PROGRAM_AUTHOR__
 * @brief   program author   macro; see expansion and module contract.
 */
#define __PROGRAM_AUTHOR__ "Flip"

/**
 * @anchor api_include_util_messages_h_help
 * @brief Perform the help operation.
 * @ingroup utilities
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void help(void);
/**
 * @anchor api_include_util_messages_h_usage
 * @brief Perform the usage operation.
 * @ingroup utilities
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void usage(void);
/**
 * @anchor api_include_util_messages_h_version
 * @brief Perform the version operation.
 * @ingroup utilities
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void version(void);
/**
 * @anchor api_include_util_messages_h_description
 * @brief Perform the description operation.
 * @ingroup utilities
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void description(void);
/**
 * @anchor api_include_util_messages_h_options
 * @brief Perform the options operation.
 * @ingroup utilities
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void options(void);
/**
 * @anchor api_include_util_messages_h_author
 * @brief Perform the author operation.
 * @ingroup utilities
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void author(void);

#endif // MESSAGES_H
