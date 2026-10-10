/* Verify named experiment routing, all levels, filtering and logger injection. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "experiments/numerical_runner.h"

#define CHECK(condition)                                                                           \
    do                                                                                             \
    {                                                                                              \
        if (!(condition))                                                                          \
        {                                                                                          \
            fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition);                           \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while (0)

typedef struct
{
    unsigned counts[5];
    char name[64];
    char message[256];
    int line;
} capture;

/* Copy borrowed data during the callback; no process-global test state. */
static void capture_message(void *data, const char *name, LogLevel level, const char *file,
                            int line, const char *message)
{
    capture *result = data;
    (void)file;
    result->counts[level]++;
    snprintf(result->name, sizeof(result->name), "%s", name);
    snprintf(result->message, sizeof(result->message), "%s", message);
    result->line = line;
}

int main(void)
{
    capture first = {0}, second = {0};
    experiment_logger a, b;
    CHECK(experiment_logger_init(&a, "experiment.alpha", LOG_DEBUG, capture_message, &first) == 0);
    CHECK(experiment_logger_init(&b, "experiment.beta", LOG_WARNING, capture_message, &second) ==
          0);

    /* Experiment thresholds must not depend on the application's global level. */
    logger_set_level(LOG_FATAL);
    EXP_LOG_DEBUG(&a, "debug %d", 42);
    CHECK(strcmp(first.message, "debug 42") == 0);
    EXP_LOG_INFO(&a, "info");
    EXP_LOG_WARNING(&a, "warning");
    EXP_LOG_ERROR(&a, "error");
    EXP_LOG_FATAL(&a, "fatal");
    for (size_t i = 0; i < 5; i++)
        CHECK(first.counts[i] == 1);
    CHECK(strcmp(first.name, "experiment.alpha") == 0 && first.line > 0);

    EXP_LOG_DEBUG(&b, "filtered");
    EXP_LOG_INFO(&b, "filtered");
    EXP_LOG_WARNING(&b, "warning %d", 7);
    CHECK(second.counts[LOG_DEBUG] == 0 && second.counts[LOG_INFO] == 0);
    CHECK(second.counts[LOG_WARNING] == 1);
    CHECK(strcmp(second.name, "experiment.beta") == 0);
    CHECK(strcmp(second.message, "warning 7") == 0);

    /* Invalid initialization preserves the previous context. */
    CHECK(experiment_logger_init(&a, "", LOG_INFO, NULL, NULL) == -1);
    CHECK(strcmp(a.experiment_name, "experiment.alpha") == 0);
    CHECK(experiment_logger_init(NULL, "name", LOG_INFO, NULL, NULL) == -1);
    experiment_logger_log(&a, (LogLevel)99, __FILE__, __LINE__, "ignored");
    logger_log((LogLevel)99, __FILE__, __LINE__, "ignored");
    CHECK(numerical_runner(NULL) == EXIT_FAILURE);
    CHECK(numerical_runner(&a) == EXIT_SUCCESS);
    CHECK(first.counts[LOG_INFO] == 3 && first.counts[LOG_DEBUG] == 5);
    puts("PASS named logger injection, independent filtering, levels and runner");

    return EXIT_SUCCESS;
}
