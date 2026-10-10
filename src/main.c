#include <sys/ptrace.h>
#include <stdio.h>
#include <stdlib.h>

#include "util/args.h"
#include "util/logger.h"
#include "experiments/numerical_runner.h"

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

    // Main logic: delegate the selected experiment to its runner.
    LOG_INFO("Running CFWx main logic\n");
    int experiment_status = numerical_runner();
    if (experiment_status != EXIT_SUCCESS)
    {
        LOG_ERROR("Numerical experiment failed");
    }

    // Cleanup
    logger_cleanup();

    // Exit
    return experiment_status;
}
