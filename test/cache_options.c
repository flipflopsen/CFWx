/**
 * @file test/cache_options.c
 * @brief cache options interfaces and implementation.
 * @ingroup tests
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Fixed/sliding cache limits, absolute positions, failed append rollback and reset. */
#include "models/runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        if (!(x))                                                                                  \
        {                                                                                          \
            fprintf(stderr, "FAIL %d %s\n", __LINE__, #x);                                         \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
/** @def OK
 * @brief ok macro; see expansion and module contract.
 */
#define OK(x) CHECK((x) == CFWX_OK)
/**
 * @anchor api_test_cache_options_c_main
 * @brief Run this executable driver and propagate its success/failure status.
 * @ingroup tests
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @return Computed value or status described by the adjacent contract.
 */
int main(void)
{
    cfwx_language_model m = {0};
    OK(cfwx_language_model_init(&m, 4, 4, 2, 6, 1, 42));
    cfwx_kv_cache sliding = {0}, fixed = {0}, full = {0};
    OK(cfwx_kv_cache_init_mode(&sliding, &m, 2, CFWX_CACHE_SLIDING));
    OK(cfwx_kv_cache_init(&fixed, &m, 2));
    OK(cfwx_kv_cache_init(&full, &m, 4));
    cfwx_matrix a = {0}, b = {0};
    OK(cfwx_matrix_init(&a, 1, 4));
    OK(cfwx_matrix_init(&b, 1, 4));
    for (size_t i = 0; i < 2; i++)
    {
        OK(cfwx_language_model_decode(&sliding, i, &a));
        OK(cfwx_language_model_decode(&full, i, &b));
        for (size_t j = 0; j < 4; j++)
            CHECK(fabs(a.data[j] - b.data[j]) < 1e-8);
        OK(cfwx_language_model_decode(&fixed, i, &b));
    }
    CHECK(cfwx_language_model_decode(&fixed, 2, &b) == CFWX_INVALID);
    for (size_t i = 2; i < 9; i++)
        OK(cfwx_language_model_decode(&sliding, i % 4, &a));
    CHECK(sliding.length == 2 && sliding.position == 9);
    double keys[8];
    memcpy(keys, sliding.keys[0].data, sizeof(keys));
    double saved = m.head.weights.data[0];
    m.head.weights.data[0] = NAN;
    CHECK(cfwx_language_model_decode(&sliding, 0, &a) != CFWX_OK);
    CHECK(sliding.length == 2 && sliding.position == 9);
    CHECK(memcmp(keys, sliding.keys[0].data, sizeof(keys)) == 0);
    m.head.weights.data[0] = saved;
    OK(cfwx_language_model_decode(&sliding, 0, &a));
    CHECK(sliding.position == 10);
    cfwx_kv_cache_reset(&sliding);
    CHECK(!sliding.length && !sliding.position);
    OK(cfwx_language_model_decode(&sliding, 0, &a));
    cfwx_kv_cache_free(&sliding);
    cfwx_kv_cache_free(&fixed);
    cfwx_kv_cache_free(&full);
    cfwx_matrix_free(&a);
    cfwx_matrix_free(&b);
    cfwx_language_model_free(&m);
    puts("PASS configurable cache eviction absolute position rollback and reset");
    return 0;
}
