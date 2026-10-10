/* Application lifecycle: parse options, initialize logging, run experiment, clean up. */

#include <sys/ptrace.h>
#include <stdio.h>
#include <stdlib.h>

#include "util/args.h"
#include "util/logger.h"
#include "experiments/numerical_runner.h"

/* Preserve the original debugger detection; PTRACE_TRACEME has process side effects. */
int is_debugger_attached(void)
{
    return (ptrace(PTRACE_TRACEME, 0, 1, 0) == -1);
}

int main(int argc, char *argv[])
{
    options_t options;
    options_parser(argc, argv, &options);

    bool isDebuggerAttached = is_debugger_attached();
    if (logger_init("log", "cfwx_app", isDebuggerAttached) != 0)
    {
        fprintf(stderr, "Failed to initialize logger\n");
        return EXIT_FAILURE;
    }

    if (isDebuggerAttached)
    {
        LOG_INFO("Debugger detected - Setting LogLevel to DEBUG");
        logger_set_level(LOG_DEBUG);
    }
    else
    {
        LOG_INFO("No debugger detected - Setting LogLevel to INFO");
        logger_set_level(LOG_INFO);
    }

    /* Identity and threshold are injected; backend cleanup remains caller-owned. */
    experiment_logger numerical_logger;
    if (experiment_logger_init(&numerical_logger, "numerical.softmax",
                               isDebuggerAttached ? LOG_DEBUG : LOG_INFO, NULL, NULL) != 0)
    {
        LOG_FATAL("Failed to initialize numerical experiment logger");
        logger_cleanup();
        return EXIT_FAILURE;
    }

    int experiment_status = numerical_runner(&numerical_logger);
    if (experiment_status != EXIT_SUCCESS)
        LOG_ERROR("Numerical experiment failed");

    /* Propagate failure only after releasing the application backend. */
    logger_cleanup();
    return experiment_status;
}
