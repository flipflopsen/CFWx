/* Dropout finite differences, token-weighted batching, KV equivalence and exact resume. */
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
#define OK(x) CHECK((x) == CFWX_OK)
#define CLOSE(a, b) CHECK(isfinite(a) && fabs((a) - (b)) < 1e-7 * (1 + fabs((double)(b))))
int main(void)
{
    cfwx_language_model m = {0}, g = {0}, other = {0}, g2 = {0};
    OK(cfwx_language_model_init(&m, 4, 4, 2, 6, 2, 42));
    OK(cfwx_language_model_init(&g, 4, 4, 2, 6, 2, 0));
    OK(cfwx_language_model_init(&g2, 4, 4, 2, 6, 2, 0));
    cfwx_parameters p = {0}, grad = {0}, grad2 = {0};
    OK(cfwx_parameters_language_model(&p, &m));
    OK(cfwx_parameters_language_model(&grad, &g));
    OK(cfwx_parameters_language_model(&grad2, &g2));
    size_t ids[] = {0, 1, 2, 0, 3}, labels[] = {1, 2, 0, 3, 1};
    uint64_t rng = 17;
    double loss;
    OK(cfwx_language_model_loss_training(&m, ids, labels, 5, 0.2, &rng, &loss, &g));
    /* Same RNG resets yield identical masks for both central-difference samples. */
    for (size_t a = 0; a < p.count; a++)
    {
        size_t j = p.size[a] / 2;
        double original = p.data[a][j], plus, minus;
        uint64_t r1 = 17, r2 = 17;
        p.data[a][j] = original + 1e-6;
        OK(cfwx_language_model_loss_training(&m, ids, labels, 5, 0.2, &r1, &plus, NULL));
        p.data[a][j] = original - 1e-6;
        OK(cfwx_language_model_loss_training(&m, ids, labels, 5, 0.2, &r2, &minus, NULL));
        p.data[a][j] = original;
        CLOSE(grad.data[a][j], (plus - minus) / 2e-6);
        CHECK(r1 == r2);
    }
    const size_t *inputs[] = {ids, ids + 2}, *targets[] = {labels, labels + 2};
    size_t lengths[] = {5, 3};
    OK(cfwx_language_model_batch(&m, inputs, targets, lengths, 2, 0, 0, NULL, &loss, &g));
    double first, second;
    OK(cfwx_language_model_loss(&m, ids, labels, 5, &first, &g2));
    double saved = grad2.data[0][0];
    OK(cfwx_language_model_loss(&m, ids + 2, labels + 2, 3, &second, &g2));
    CLOSE(loss, (first * 5 + second * 3) / 8);
    CLOSE(grad.data[0][0], (saved * 5 + grad2.data[0][0] * 3) / 8);
    size_t scored;
    double evaluated;
    OK(cfwx_language_model_evaluate(&m, ids, 5, 3, &evaluated, &scored));
    CHECK(scored == 4);
    OK(cfwx_language_model_loss(&m, ids, ids + 1, 3, &first, NULL));
    OK(cfwx_language_model_loss(&m, ids + 3, ids + 4, 1, &second, NULL));
    CLOSE(evaluated, (first * 3 + second) / 4);
    cfwx_kv_cache cache = {0};
    OK(cfwx_kv_cache_init(&cache, &m, 5));
    cfwx_matrix logits = {0};
    OK(cfwx_matrix_init(&logits, 1, 4));
    for (size_t i = 0; i < 5; i++)
    {
        cfwx_matrix full = {0};
        OK(cfwx_matrix_init(&full, i + 1, 4));
        OK(cfwx_language_model_decode(&cache, ids[i], &logits));
        OK(cfwx_language_model_forward(&m, ids, i + 1, &full));
        for (size_t j = 0; j < 4; j++)
            CLOSE(logits.data[j], full.data[i * 4 + j]);
        cfwx_matrix_free(&full);
    }
    CHECK(cfwx_language_model_decode(&cache, 0, &logits) == CFWX_INVALID);
    cfwx_kv_cache_reset(&cache);
    CHECK(cache.length == 0);
    cfwx_adamw optimizer = {0}, loaded = {0};
    OK(cfwx_adamw_init(&optimizer, &p));
    OK(cfwx_adamw_step(&optimizer, &p, &grad, 0.01, 0.9, 0.999, 1e-8, 0));
    cfwx_training_state state = {0};
    state.rng = 777;
    state.context = 5;
    state.batch = 2;
    state.unknown = 3;
    state.dropout = 0.2;
    state.rate = 0.01;
    state.beta1 = 0.9;
    state.beta2 = 0.999;
    state.epsilon = 1e-8;
    state.clip = 1;
    state.dataset_hash = 123;
    for (size_t i = 0; i < 256; i++)
        state.byte_map[i] = SIZE_MAX;
    state.byte_map['a'] = 0;
    state.byte_map['b'] = 1;
    state.byte_map['c'] = 2;
    OK(cfwx_checkpoint_save("runtime-test.ckpt", &m, &optimizer, &state));
    cfwx_training_state restored = {0};
    OK(cfwx_checkpoint_load("runtime-test.ckpt", &other, &loaded, &restored));
    CHECK(restored.rng == state.rng && loaded.step == optimizer.step);
    cfwx_parameters op = {0};
    OK(cfwx_parameters_language_model(&op, &other));
    for (size_t a = 0; a < p.count; a++)
        CHECK(!memcmp(p.data[a], op.data[a], p.size[a] * sizeof(double)));
    uint64_t r1 = state.rng, r2 = restored.rng;
    OK(cfwx_language_model_loss_training(&m, ids, labels, 5, 0.2, &r1, &first, &g));
    OK(cfwx_language_model_loss_training(&other, ids, labels, 5, 0.2, &r2, &second, &g2));
    CHECK(first == second && r1 == r2);
    OK(cfwx_adamw_step(&optimizer, &p, &grad, 0.01, 0.9, 0.999, 1e-8, 0));
    OK(cfwx_adamw_step(&loaded, &op, &grad2, 0.01, 0.9, 0.999, 1e-8, 0));
    for (size_t a = 0; a < p.count; a++)
        CHECK(!memcmp(p.data[a], op.data[a], p.size[a] * sizeof(double)));
    FILE *f = fopen("runtime-test.ckpt", "r+b");
    CHECK(f);
    CHECK(fseek(f, 100, SEEK_SET) == 0);
    int original = fgetc(f);
    CHECK(original != EOF);
    CHECK(fseek(f, 100, SEEK_SET) == 0);
    CHECK(fputc(original ^ 1, f) != EOF);
    fclose(f);
    cfwx_language_model invalid = {0};
    cfwx_adamw invalid_o = {0};
    CHECK(cfwx_checkpoint_load("runtime-test.ckpt", &invalid, &invalid_o, &restored) != CFWX_OK);
    CHECK(!invalid.embeddings.data && !invalid_o.count);
    remove("runtime-test.ckpt");
    size_t sampled;
    uint64_t oldrng = r1;
    OK(cfwx_sample_logits(logits.data, 4, 0, 0, &r1, &sampled));
    CHECK(r1 == oldrng);
    OK(cfwx_sample_logits(logits.data, 4, 1, 1, &r1, &sampled));
    cfwx_matrix_free(&logits);
    cfwx_kv_cache_free(&cache);
    cfwx_adamw_free(&optimizer);
    cfwx_adamw_free(&loaded);
    cfwx_language_model_free(&m);
    cfwx_language_model_free(&other);
    cfwx_language_model_free(&g);
    cfwx_language_model_free(&g2);
    puts("PASS dropout gradients batching full evaluation KV equality checkpoint corruption and "
         "exact resume");
    return 0;
}
