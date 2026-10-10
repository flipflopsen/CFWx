/**
 * @file src/util/logger.c
 * @brief logger interfaces and implementation.
 * @ingroup logging
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Legacy console/file backend with explicitly injected experiment contexts. */

#include "util/logger.h"
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdio.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
/** @def mkdir
 * @brief mkdir macro; see expansion and module contract.
 */
#define mkdir(path, mode) _mkdir(path)
#else
#include <unistd.h>
#endif

/** Backend lifecycle belongs to the application, not to experiments. */
static FILE *log_file = NULL;
static LogLevel current_level = LOG_DEBUG;
static const char *level_strings[] = {"DEBUG", "INFO", "WARNING", "ERROR", "FATAL"};
static const char *level_colors[] = {
    "\x1b[36m", /* DEBUG: cyan */
    "\x1b[32m", /* INFO: green */
    "\x1b[33m", /* WARNING: yellow */
    "\x1b[31m", /* ERROR: red */
    "\x1b[35m"  /* FATAL: magenta */
};
static const char *color_reset = "\x1b[0m";

/**
 * @anchor api_src_util_logger_c_valid_level
 * @brief Perform the valid level operation.
 * @ingroup logging
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param level Valid logging severity from DEBUG through FATAL.
 * @return Computed value or status described by the adjacent contract.
 */
static bool valid_level(LogLevel level)
{
    return level >= LOG_DEBUG && level <= LOG_FATAL;
}

/**
 * @anchor api_src_util_logger_c_get_timestamp
 * @brief Perform the get timestamp operation.
 * @ingroup logging
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param buffer buffer argument; interpretation follows this operation's implementation and contract.
 * @param size Positive number of elements in the requested storage.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
static void get_timestamp(char *buffer, size_t size)
{
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);

    /* Clock failures must not leave an uninitialized timestamp. */
    if (!tm_info || strftime(buffer, size, "%Y-%m-%d %H:%M:%S", tm_info) == 0)
        snprintf(buffer, size, "unknown-time");
}

/**
 * @anchor api_src_util_logger_c_get_filename
 * @brief Perform the get filename operation.
 * @ingroup logging
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param path File path used by the operation; checkpoint publication requires a writable parent directory.
 * @return Pointer described by the operation contract; ownership is not implied.
 */
static const char *get_filename(const char *path)
{
    if (!path)
        return "unknown";

    const char *filename = strrchr(path, '/');
    if (!filename)
        filename = strrchr(path, '\\');

    return filename ? filename + 1 : path;
}

/**
 * @anchor api_src_util_logger_c_logger_init
 * @brief Construct and initialize logger state.
 * @ingroup logging
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see logger_init include/util/logger.h
 */
int logger_init(const char *log_folder, const char *log_filename, bool debug)
{
    /* Do not leak an already initialized backend on a repeated init call. */
    if (!log_folder || !log_filename || log_file)
        return -1;

    const char *project_root = PROJECT_ROOT;
    size_t folder_len = strlen(project_root) + strlen(log_folder) + 2;
    char *folder_path = malloc(folder_len);
    if (!folder_path)
        return -1;

    snprintf(folder_path, folder_len, "%s/%s", project_root, log_folder);
    if (mkdir(folder_path, 0755) != 0 && errno != EEXIST)
    {
        perror("mkdir() error");
        free(folder_path);
        return -1;
    }

    const char *suffix = debug ? "_debug.log" : ".log";
    size_t full_len = strlen(folder_path) + strlen(log_filename) + strlen(suffix) + 2;
    char *full_path = malloc(full_len);
    if (!full_path)
    {
        free(folder_path);
        return -1;
    }

    snprintf(full_path, full_len, "%s/%s%s", folder_path, log_filename, suffix);
    log_file = fopen(full_path, "a");
    if (!log_file)
    {
        fprintf(stderr, "Failed to open log file %s: %s\n", full_path, strerror(errno));
        free(folder_path);
        free(full_path);
        return -1;
    }

    fprintf(log_file, "[INFO] Logger initialized - writing to %s\n", full_path);
    fflush(log_file);
    free(folder_path);
    free(full_path);

    return 0;
}

/**
 * @anchor api_src_util_logger_c_logger_cleanup
 * @brief Perform the logger cleanup operation.
 * @ingroup logging
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see logger_cleanup include/util/logger.h
 */
void logger_cleanup(void)
{
    if (log_file && log_file != stdout && log_file != stderr)
    {
        LOG_INFO("Logger shutting down");
        fclose(log_file);
        log_file = NULL;
    }
}

/**
 * @anchor api_src_util_logger_c_logger_set_level
 * @brief Perform the logger set level operation.
 * @ingroup logging
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see logger_set_level include/util/logger.h
 */
void logger_set_level(LogLevel level)
{
    if (valid_level(level))
    {
        current_level = level;
        LOG_INFO("Log level set to %s", level_strings[level]);
    }
}

/** Independent context thresholds are checked before reaching this shared sink.
 * va_copy is required because console and file each consume the arguments. */
static void logger_write(LogLevel level, const char *name, const char *file, int line,
                         const char *fmt, va_list args)
{
    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    const char *filename = get_filename(file);
    va_list copy;

    va_copy(copy, args);
    fprintf(stderr, "%s[%s %s %s %s:%d]%s ", level_colors[level], timestamp, level_strings[level],
            name, filename, line, color_reset);
    vfprintf(stderr, fmt, copy);
    fprintf(stderr, "\n");
    fflush(stderr);
    va_end(copy);

    if (log_file)
    {
        va_copy(copy, args);
        fprintf(log_file, "[%s %s %s %s:%d] ", timestamp, level_strings[level], name, filename,
                line);
        vfprintf(log_file, fmt, copy);
        fprintf(log_file, "\n");
        fflush(log_file);
        va_end(copy);
    }
}

/**
 * @anchor api_src_util_logger_c_logger_log
 * @brief Emit an application-attributed record through the legacy shared backend.
 * @ingroup logging
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see logger_log include/util/logger.h
 */
void logger_log(LogLevel level, const char *file, int line, const char *fmt, ...)
{
    /* Validate before indexing level/color arrays. */
    if (!valid_level(level) || level < current_level || !fmt)
        return;

    va_list args;
    va_start(args, fmt);
    logger_write(level, "application", file, line, fmt, args);
    va_end(args);
}

/**
 * @anchor api_src_util_logger_c_experiment_logger_init
 * @brief Initialize a borrowed named logger context and optional sink.
 * @ingroup logging
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see experiment_logger_init include/util/logger.h
 */
int experiment_logger_init(experiment_logger *logger, const char *name, LogLevel minimum_level,
                           logger_sink_fn sink, void *user_data)
{
    if (!logger || !name || !name[0] || !valid_level(minimum_level))
        return -1;

    *logger = (experiment_logger){name, minimum_level, sink, user_data};
    return 0;
}

/**
 * @anchor api_src_util_logger_c_experiment_logger_log
 * @brief Filter and route an experiment-attributed formatted record.
 * @ingroup logging
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see experiment_logger_log include/util/logger.h
 */
void experiment_logger_log(const experiment_logger *logger, LogLevel level, const char *file,
                           int line, const char *fmt, ...)
{
    if (!logger || !logger->experiment_name || !logger->experiment_name[0] ||
        !valid_level(logger->minimum_level) || !valid_level(level) ||
        level < logger->minimum_level || !fmt)
        return;

    va_list args;
    va_start(args, fmt);
    if (!logger->sink)
    {
        logger_write(level, logger->experiment_name, file, line, fmt, args);
    }
    else
    {
        /* Two-pass formatting avoids silently truncating a captured message.
         * Allocation/formatting failures drop the log, never alter results. */
        va_list copy;
        va_copy(copy, args);
        int length = vsnprintf(NULL, 0, fmt, copy);
        va_end(copy);
        if (length >= 0)
        {
            char *message = malloc((size_t)length + 1);
            if (message)
            {
                int written = vsnprintf(message, (size_t)length + 1, fmt, args);
                if (written == length)
                    logger->sink(logger->user_data, logger->experiment_name, level, file, line,
                                 message);
                free(message);
            }
        }
    }
    va_end(args);
}
