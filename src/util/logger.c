#include "util/logger.h"
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdio.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <unistd.h>
#endif

// Global logger state
static FILE *log_file = NULL;
static LogLevel current_level = LOG_DEBUG;
static const char *level_strings[] = {
    "DEBUG", "INFO", "WARNING", "ERROR", "FATAL"};

// ANSI color codes for console output
static const char *level_colors[] = {
    "\x1b[36m", // DEBUG - Cyan
    "\x1b[32m", // INFO - Green
    "\x1b[33m", // WARNING - Yellow
    "\x1b[31m", // ERROR - Red
    "\x1b[35m"  // FATAL - Magenta
};
static const char *color_reset = "\x1b[0m";

// Get current timestamp string
static void get_timestamp(char *buffer, size_t size)
{
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(buffer, size, "%Y-%m-%d %H:%M:%S", tm_info);
}

// Extract filename from full path
static const char *get_filename(const char *path)
{
    const char *filename = strrchr(path, '/');
    if (!filename)
    {
        filename = strrchr(path, '\\');
    }
    return filename ? filename + 1 : path;
}

int logger_init(const char *log_folder, const char *log_filename, bool debug)
{
    const char *project_root = PROJECT_ROOT;

    // Full folder path
    size_t folder_len = strlen(project_root) + 1 + strlen(log_folder) + 1; // +1 for '/' and +1 for '\0'
    char *folder_path = malloc(folder_len);
    if (!folder_path)
        return -1;
    snprintf(folder_path, folder_len, "%s/%s", project_root, log_folder);

    // Create folder if it doesn't exist
    if (mkdir(folder_path, 0755) != 0 && errno != EEXIST)
    {
        perror("mkdir() error");
        free(folder_path);
        return -1;
    }

    // Full filename
    const char *suffix = debug ? "_debug.log" : ".log";
    size_t filename_len = strlen(log_filename) + strlen(suffix) + 1;
    char *filename = malloc(filename_len);
    if (!filename)
    {
        free(folder_path);
        return -1;
    }
    snprintf(filename, filename_len, "%s%s", log_filename, suffix);

    // Full path
    size_t full_len = strlen(folder_path) + 1 + strlen(filename) + 1;
    char *full_path = malloc(full_len);
    if (!full_path)
    {
        free(folder_path);
        free(filename);
        return -1;
    }
    snprintf(full_path, full_len, "%s/%s", folder_path, filename);

    // Open log file
    log_file = fopen(full_path, "a");
    if (!log_file)
    {
        fprintf(stderr, "Failed to open log file %s: %s\n", full_path, strerror(errno));
        free(folder_path);
        free(filename);
        free(full_path);
        return -1;
    }

    fprintf(log_file, "[INFO] Logger initialized - writing to %s\n", full_path);
    fflush(log_file);

    // Clean up
    free(folder_path);
    free(filename);
    free(full_path);

    return 0;
}

void logger_cleanup(void)
{
    if (log_file && log_file != stdout && log_file != stderr)
    {
        LOG_INFO("Logger shutting down");
        fclose(log_file);
        log_file = NULL;
    }
}

void logger_set_level(LogLevel level)
{
    if (level >= LOG_DEBUG && level <= LOG_FATAL)
    {
        current_level = level;
        LOG_INFO("Log level set to %s", level_strings[level]);
    }
}

void logger_log(LogLevel level, const char *file, int line, const char *fmt, ...)
{
    if (level < current_level)
    {
        return;
    }

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));

    const char *filename = get_filename(file);

    va_list args;

    // Log to console with colors
    va_start(args, fmt);
    fprintf(stderr, "%s[%s %s %s:%d]%s ",
            level_colors[level], timestamp, level_strings[level],
            filename, line, color_reset);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    fflush(stderr);
    va_end(args);

    // Log to file (without colors)
    if (log_file)
    {
        va_start(args, fmt);
        fprintf(log_file, "[%s %s %s:%d] ",
                timestamp, level_strings[level], filename, line);
        vfprintf(log_file, fmt, args);
        fprintf(log_file, "\n");
        fflush(log_file);
        va_end(args);
    }
}
