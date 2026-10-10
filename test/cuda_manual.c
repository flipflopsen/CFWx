/**
 * @file test/cuda_manual.c
 * @brief cuda manual interfaces and implementation.
 * @ingroup gpu
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** MANUAL ONLY: not built or executed in CPU CI. Named logger remains on host. */
#include "gpu/cuda_training.h"
#include "util/logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>

/**
 * @anchor api_test_cuda_manual_c_test
 * @brief Perform the test operation.
 * @ingroup gpu
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param logger Borrowed initialized experiment logger; name/user data must outlive all calls.
 * @return Computed value or status described by the adjacent contract.
 * @warning CUDA source remains manually validated only. Doxygen parses this
 * symbol without running nvcc, CUDA kernels or hardware probes.
 */
static int test(const experiment_logger *logger)
{
    cfwx_cuda_config c = {3, 4, 2, 6, 1, 3, 2, 42, 0.0, 1000000, 512 * 1024 * 1024};
    cfwx_cuda_model *m = NULL;
    char error[256];
    if (cfwx_cuda_create(&c, &m, error, sizeof(error)))
    {
        EXP_LOG_ERROR(logger, "%s", error);
        return 1;
    }
    const size_t ids[] = {0, 1, 2, 1, 2, 0}, targets[] = {1, 2, 0, 2, 0, 1};
    const char *fused = getenv("CFWX_CUDA_FUSED_BATCH");
    if (fused && strcmp(fused, "1") == 0)
        cfwx_cuda_set_fused_batching(m, 1);
    size_t count = cfwx_cuda_parameter_count(m);
    double *p = malloc(count * sizeof(double)), *g = malloc(count * sizeof(double));
    int result = 1;
    double initial, loss;
    if (!p || !g)
        goto cleanup;
    if (cfwx_cuda_run(m, ids, targets, 1, 0, 1, 0, &initial) ||
        cfwx_cuda_get_parameters(m, p, count) || cfwx_cuda_get_gradients(m, g, count))
        goto cleanup;
    /* Every scalar parameter: analytic GPU gradient vs GPU finite differences.
     * No dropout ensures identical graph evaluations without RNG state resets. */
    for (size_t i = 0; i < count; i++)
    {
        double old = p[i], plus, minus;
        p[i] = old + 1e-5;
        if (cfwx_cuda_set_parameters(m, p, count) ||
            cfwx_cuda_run(m, ids, targets, 0, 0, 1, 0, &plus))
            goto cleanup;
        p[i] = old - 1e-5;
        if (cfwx_cuda_set_parameters(m, p, count) ||
            cfwx_cuda_run(m, ids, targets, 0, 0, 1, 0, &minus))
            goto cleanup;
        p[i] = old;
        double numeric = (plus - minus) / 2e-5;
        if (fabs(g[i] - numeric) > 2e-4 * (1 + fabs(numeric)))
        {
            EXP_LOG_ERROR(logger, "Gradient mismatch index=%zu analytic=%.9g numerical=%.9g", i,
                          g[i], numeric);
            goto cleanup;
        }
    }
    if (cfwx_cuda_set_parameters(m, p, count))
        goto cleanup;
    for (size_t i = 0; i < 200; i++)
        if (cfwx_cuda_run(m, ids, targets, 1, 0.01, 1, 0, &loss))
            goto cleanup;
    if (cfwx_cuda_run(m, ids, targets, 0, 0, 1, 0, &loss))
        goto cleanup;
    if (!(loss < initial * 0.5))
    {
        EXP_LOG_ERROR(logger, "Loss reduction failed %.9f -> %.9f", initial, loss);
        goto cleanup;
    }
    EXP_LOG_INFO(logger, "PASS %zu parameter gradients and device training %.9f -> %.9f", count,
                 initial, loss);
    result = 0;
cleanup:
    if (result)
        EXP_LOG_ERROR(logger, "Manual CUDA test failed: %s", cfwx_cuda_error(m));
    free(p);
    free(g);
    cfwx_cuda_free(m);
    return result;
}
/**
 * @anchor api_test_cuda_manual_c_train
 * @brief Perform the train operation.
 * @ingroup gpu
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param logger Borrowed initialized experiment logger; name/user data must outlive all calls.
 * @param path File path used by the operation; checkpoint publication requires a writable parent directory.
 * @param steps Positive number of requested optimizer updates.
 * @return Computed value or status described by the adjacent contract.
 * @warning CUDA source remains manually validated only. Doxygen parses this
 * symbol without running nvcc, CUDA kernels or hardware probes.
 */
static int train(const experiment_logger *logger, const char *path, size_t steps)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        EXP_LOG_ERROR(logger, "Cannot open dataset");
        return 1;
    }
    unsigned char *data = malloc(1048577);
    size_t n = data ? fread(data, 1, 1048577, f) : 0;
    int error = ferror(f);
    fclose(f);
    if (!data || error || n < 64 || n > 1048576)
    {
        free(data);
        return 1;
    }
    size_t map[256], vocab = 0, split = n * 4 / 5;
    for (size_t i = 0; i < 256; i++)
        map[i] = SIZE_MAX;
    for (size_t i = 0; i < split; i++)
        if (map[data[i]] == SIZE_MAX)
            map[data[i]] = vocab++;
    size_t unknown = vocab++;
    cfwx_cuda_config c = {vocab, 8, 2, 16, 1, 8, 4, 42, 0.1, 8000000, 2ULL * 1024 * 1024 * 1024};
    cfwx_cuda_model *m = NULL;
    char message[256];
    if (cfwx_cuda_create(&c, &m, message, sizeof(message)))
    {
        EXP_LOG_ERROR(logger, "%s", message);
        free(data);
        return 1;
    }
    const char *fused = getenv("CFWX_CUDA_FUSED_BATCH");
    if (fused && strcmp(fused, "1") == 0)
        cfwx_cuda_set_fused_batching(m, 1);
    size_t ids[32], targets[32];
    double loss = 0;
    int result = 0;
    for (size_t step = 0; step < steps; step++)
    {
        for (size_t b = 0; b < 4; b++)
            for (size_t t = 0; t < 8; t++)
            {
                size_t start = ((step * 4 + b) * 8) % (split - 8);
                ids[b * 8 + t] = map[data[start + t]];
                targets[b * 8 + t] = map[data[start + t + 1]];
            }
        if (cfwx_cuda_run(m, ids, targets, 1, 0.01, 1, 0, &loss))
        {
            result = 1;
            break;
        }
        if (step % 25 == 0 || step + 1 == steps)
            EXP_LOG_INFO(logger, "CUDA step=%zu joint token-mean loss=%.9f", step + 1, loss);
    }
    /* Score every held-out target. Fixed-shape backend pads final batch and
     * evaluates each target in a replicated single-target context is unsupported:
     * report only complete context batches and explicitly count excluded tail. */
    double sum = 0.0;
    size_t scored = 0;
    for (size_t start = split; start + 32 < n && !result; start += 32)
    {
        for (size_t i = 0; i < 32; i++)
        {
            size_t a = map[data[start + i]], b = map[data[start + i + 1]];
            ids[i] = a == SIZE_MAX ? unknown : a;
            targets[i] = b == SIZE_MAX ? unknown : b;
        }
        if (cfwx_cuda_run(m, ids, targets, 0, 0, 1, 0, &loss))
        {
            result = 1;
            break;
        }
        sum += loss * 32;
        scored += 32;
    }
    if (!result)
        EXP_LOG_INFO(logger, "CUDA validation loss=%.9f scored=%zu excluded_tail=%zu",
                     scored ? sum / (double)scored : 0.0, scored, n - split - 1 - scored);
    if (result)
        EXP_LOG_ERROR(logger, "CUDA failure: %s", cfwx_cuda_error(m));
    cfwx_cuda_free(m);
    free(data);
    return result;
}
/**
 * @anchor api_test_cuda_manual_c_main
 * @brief Run this executable driver and propagate its success/failure status.
 * @ingroup gpu
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param argc Number of process arguments including argv[0].
 * @param argv Process argument strings; the legacy parser may terminate on help/error.
 * @return Computed value or status described by the adjacent contract.
 * @warning CUDA source remains manually validated only. Doxygen parses this
 * symbol without running nvcc, CUDA kernels or hardware probes.
 */
int main(int argc, char **argv)
{
    experiment_logger logger;
    if (experiment_logger_init(&logger, "transformer.cuda.manual", LOG_INFO, NULL, NULL))
        return 1;
    if (argc == 2 && !strcmp(argv[1], "--test"))
        return test(&logger);
    if (argc == 3)
    {
        char *end;
        errno = 0;
        unsigned long steps = strtoul(argv[2], &end, 10);
        if (!errno && argv[2][0] && argv[2][0] != '-' && !*end && steps && steps <= 100000)
            return train(&logger, argv[1], (size_t)steps);
    }
    fprintf(stderr, "Usage: %s --test | DATASET STEPS\n", argv[0]);
    return 1;
}
