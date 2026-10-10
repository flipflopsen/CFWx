/**
 * @file include/util/logger.h
 * @brief logger interfaces and implementation.
 * @ingroup logging
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Application backend and borrowed named experiment logger contracts. */

#ifndef LOGGER_H
#define LOGGER_H

#include <stdarg.h>
#include <stdbool.h>

/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef enum
{
    LOG_DEBUG = 0, /**< log debug enum choice. */
    LOG_INFO, /**< log info enum choice. */
    LOG_WARNING, /**< log warning enum choice. */
    LOG_ERROR, /**< log error enum choice. */
    LOG_FATAL /**< log fatal enum choice. */
} LogLevel;

/** The legacy console/file backend is process-global and not thread-safe. */
/**
 * @anchor api_include_util_logger_h_logger_init
 * @brief Construct and initialize logger state.
 * @ingroup logging
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * The legacy console/file backend is process-global and not thread-safe.
 * @param log_folder log_folder argument; interpretation follows this operation's implementation and contract.
 * @param log_filename log_filename argument; interpretation follows this operation's implementation and contract.
 * @param debug debug argument; interpretation follows this operation's implementation and contract.
 * @return Computed value or status described by the adjacent contract.
 */
int logger_init(const char *log_folder, const char *log_filename, bool debug);
/**
 * @anchor api_include_util_logger_h_logger_cleanup
 * @brief Perform the logger cleanup operation.
 * @ingroup logging
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void logger_cleanup(void);
/**
 * @anchor api_include_util_logger_h_logger_set_level
 * @brief Perform the logger set level operation.
 * @ingroup logging
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param level Valid logging severity from DEBUG through FATAL.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void logger_set_level(LogLevel level);
/**
 * @anchor api_include_util_logger_h_logger_log
 * @brief Emit an application-attributed record through the legacy shared backend.
 * @ingroup logging
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param level Valid logging severity from DEBUG through FATAL.
 * @param file Call-site filename or internal stream handle, according to the declared type.
 * @param line Source line number for diagnostic attribution.
 * @param fmt Trusted printf-style format string; remaining variadic arguments must match it.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void logger_log(LogLevel level, const char *file, int line, const char *fmt, ...);

/** @def LOG_DEBUG
 * @brief log debug macro; see expansion and module contract.
 */
#define LOG_DEBUG(...) logger_log(LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
/** @def LOG_INFO
 * @brief log info macro; see expansion and module contract.
 */
#define LOG_INFO(...) logger_log(LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
/** @def LOG_WARNING
 * @brief log warning macro; see expansion and module contract.
 */
#define LOG_WARNING(...) logger_log(LOG_WARNING, __FILE__, __LINE__, __VA_ARGS__)
/** @def LOG_ERROR
 * @brief log error macro; see expansion and module contract.
 */
#define LOG_ERROR(...) logger_log(LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
/** @def LOG_FATAL
 * @brief log fatal macro; see expansion and module contract.
 */
#define LOG_FATAL(...) logger_log(LOG_FATAL, __FILE__, __LINE__, __VA_ARGS__)

/** Sink message storage is valid only during the callback; copy retained data.
 * Names and user_data are borrowed and must outlive calls to the context.
 * A custom sink can capture messages in tests or route them to another backend. */
typedef void (*logger_sink_fn)(void *user_data, const char *experiment_name, LogLevel level,
                               const char *file, int line, const char *message);

/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef struct
{
    const char *experiment_name; /**< Borrowed pointer or owned storage as specified by the containing type. */
    LogLevel minimum_level; /**< State/dimension/policy value; see the containing type contract. */
    logger_sink_fn sink; /**< State/dimension/policy value; see the containing type contract. */
    void *user_data; /**< Borrowed pointer or owned storage as specified by the containing type. */
} experiment_logger;

/** NULL sink selects the legacy backend; thresholds are independent of its
 * global application threshold. Returns -1 for invalid arguments, leaving
 * the destination unchanged. Context initialization does not open log files. */
/**
 * @anchor api_include_util_logger_h_experiment_logger_init
 * @brief Initialize a borrowed named logger context and optional sink.
 * @ingroup logging
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * NULL sink selects the legacy backend; thresholds are independent of its global application threshold. Returns -1 for invalid arguments, leaving the destination unchanged. Context initialization does not open log files.
 * @param logger Borrowed initialized experiment logger; name/user data must outlive all calls.
 * @param name Borrowed name/identity string; consult lifetime requirements of this interface.
 * @param minimum_level Minimum emitted severity; independent of the application-wide threshold.
 * @param sink Optional message callback; NULL selects the legacy backend.
 * @param user_data Borrowed callback state, not owned or freed by the logger.
 * @return Computed value or status described by the adjacent contract.
 */
int experiment_logger_init(experiment_logger *logger, const char *name, LogLevel minimum_level,
                           logger_sink_fn sink, void *user_data);
/**
 * @anchor api_include_util_logger_h_experiment_logger_log
 * @brief Filter and route an experiment-attributed formatted record.
 * @ingroup logging
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param logger Borrowed initialized experiment logger; name/user data must outlive all calls.
 * @param level Valid logging severity from DEBUG through FATAL.
 * @param file Call-site filename or internal stream handle, according to the declared type.
 * @param line Source line number for diagnostic attribution.
 * @param fmt Trusted printf-style format string; remaining variadic arguments must match it.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void experiment_logger_log(const experiment_logger *logger, LogLevel level, const char *file,
                           int line, const char *fmt, ...);

#define EXP_LOG_DEBUG(logger, ...)                                                                 \
    experiment_logger_log(logger, LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define EXP_LOG_INFO(logger, ...)                                                                  \
    experiment_logger_log(logger, LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define EXP_LOG_WARNING(logger, ...)                                                               \
    experiment_logger_log(logger, LOG_WARNING, __FILE__, __LINE__, __VA_ARGS__)
#define EXP_LOG_ERROR(logger, ...)                                                                 \
    experiment_logger_log(logger, LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define EXP_LOG_FATAL(logger, ...)                                                                 \
    experiment_logger_log(logger, LOG_FATAL, __FILE__, __LINE__, __VA_ARGS__)

#endif
