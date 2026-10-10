/* Application backend and borrowed named experiment logger contracts. */

#ifndef LOGGER_H
#define LOGGER_H

#include <stdarg.h>
#include <stdbool.h>

typedef enum
{
    LOG_DEBUG = 0,
    LOG_INFO,
    LOG_WARNING,
    LOG_ERROR,
    LOG_FATAL
} LogLevel;

/* The legacy console/file backend is process-global and not thread-safe. */
int logger_init(const char *log_folder, const char *log_filename, bool debug);
void logger_cleanup(void);
void logger_set_level(LogLevel level);
void logger_log(LogLevel level, const char *file, int line, const char *fmt, ...);

#define LOG_DEBUG(...) logger_log(LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...) logger_log(LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARNING(...) logger_log(LOG_WARNING, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) logger_log(LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_FATAL(...) logger_log(LOG_FATAL, __FILE__, __LINE__, __VA_ARGS__)

/* Sink message storage is valid only during the callback; copy retained data.
 * Names and user_data are borrowed and must outlive calls to the context.
 * A custom sink can capture messages in tests or route them to another backend. */
typedef void (*logger_sink_fn)(void *user_data, const char *experiment_name, LogLevel level,
                               const char *file, int line, const char *message);

typedef struct
{
    const char *experiment_name;
    LogLevel minimum_level;
    logger_sink_fn sink;
    void *user_data;
} experiment_logger;

/* NULL sink selects the legacy backend; thresholds are independent of its
 * global application threshold. Returns -1 for invalid arguments, leaving
 * the destination unchanged. Context initialization does not open log files. */
int experiment_logger_init(experiment_logger *logger, const char *name, LogLevel minimum_level,
                           logger_sink_fn sink, void *user_data);
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
