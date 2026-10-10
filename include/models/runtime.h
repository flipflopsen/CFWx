/* Training/inference lifecycle beyond individual Transformer blocks. */
#ifndef CFWX_RUNTIME_H
#define CFWX_RUNTIME_H
#include "models/training.h"

/* Caller RNG advances only for training dropout. Backward uses exactly the
 * stored forward multipliers within the same tape, never regenerated masks.
 * Dropout currently covers the two residual branches per block. */
cfwx_status cfwx_language_model_loss_training(const cfwx_language_model *model,
                                              const size_t *tokens, const size_t *targets,
                                              size_t length, double dropout, uint64_t *rng,
                                              double *loss, cfwx_language_model *gradient);

/* Equal/unequal sequence lengths: use lengths[b] or common context if NULL.
 * Arrays hold per-sequence pointers; SIZE_MAX targets are ignored. Gradients
 * are token-weighted across the complete batch, overwritten once per call.
 * This reference implementation accumulates sequences on CPU; not fused GPU
 * batching. Dropout randomness uses the caller-owned stream. */
cfwx_status cfwx_language_model_batch(const cfwx_language_model *model, const size_t *const *tokens,
                                      const size_t *const *targets, const size_t *lengths,
                                      size_t batch, size_t context, double dropout, uint64_t *rng,
                                      double *loss, cfwx_language_model *gradient);
/* Full nonoverlapping held-out evaluation: every next-token target counted
 * once, including the short final chunk. Positions reset at chunk boundaries.
 * No dropout/RNG consumption. Returns token-mean CE and number scored. */
cfwx_status cfwx_language_model_evaluate(const cfwx_language_model *model, const size_t *tokens,
                                         size_t count, size_t context, double *loss,
                                         size_t *scored);

/* Complete experiment state saved with parameters/moments: vocabulary mapping,
 * dataset fingerprint, sampler/dropout RNG, and optimizer hyperparameters.
 * Byte map values are [0,vocabulary) or SIZE_MAX; unknown is a valid token ID.
 * Fingerprint is a reproducibility guard, not cryptographic authentication. */
typedef struct
{
    uint64_t rng, dataset_hash;
    size_t context, batch, unknown, byte_map[256];
    double dropout, rate, beta1, beta2, epsilon, decay, clip;
} cfwx_training_state;
uint64_t cfwx_dataset_hash(const unsigned char *data, size_t count);
/* Versioned binary, little-endian uint64 and IEEE-754 binary64, FNV checksum.
 * Save uses same-directory temporary file and rename; single writer only.
 * Load requires zero-initialized model/optimizer, validates before committing.
 * Limits: width<=4096, depth<=8, vocabulary<=65536, serialized payload<=256MiB.
 * No gradient/tape/cache saved; they are reconstructed after loading. */
cfwx_status cfwx_checkpoint_save(const char *path, cfwx_language_model *model,
                                 const cfwx_adamw *optimizer, const cfwx_training_state *state);
cfwx_status cfwx_checkpoint_load(const char *path, cfwx_language_model *model,
                                 cfwx_adamw *optimizer, cfwx_training_state *state);

/* Real projected per-block key/value storage. Cache borrows an immutable model;
 * updating model weights invalidates cache: reset/reinitialize after training.
 * Token positions increase monotonically, no sliding-window eviction.
 * Reset starts a new sequence without reallocating. One cache per sequence. */
typedef enum
{
    CFWX_CACHE_FIXED,
    CFWX_CACHE_SLIDING
} cfwx_cache_mode;

typedef struct
{
    const cfwx_language_model *model;
    size_t capacity, length;
    size_t position; /* Absolute position; does not reset on eviction. */
    cfwx_cache_mode mode;
    cfwx_matrix keys[CFWX_MAX_BLOCKS], values[CFWX_MAX_BLOCKS];
} cfwx_kv_cache;
cfwx_status cfwx_kv_cache_init(cfwx_kv_cache *cache, const cfwx_language_model *model,
                               size_t capacity);
/* Configurable mode; legacy init defaults to fixed. Sliding evicts oldest K/V
 * only when full. On failed full append committed entries are restored. */
cfwx_status cfwx_kv_cache_init_mode(cfwx_kv_cache *cache, const cfwx_language_model *model,
                                    size_t capacity, cfwx_cache_mode mode);
void cfwx_kv_cache_reset(cfwx_kv_cache *cache);
void cfwx_kv_cache_free(cfwx_kv_cache *cache);
/* Append one token, compute logits [1,vocabulary] for the next-token distribution.
 * Cache length advances only on success; failed slot can be overwritten safely. */
cfwx_status cfwx_language_model_decode(cfwx_kv_cache *cache, size_t token, cfwx_matrix *logits);
/* Stable temperature sampling. temperature=0 means greedy and consumes no RNG.
 * top_k=0 keeps all classes; other values must not exceed vocabulary size. */
cfwx_status cfwx_sample_logits(const double *logits, size_t vocabulary, double temperature,
                               size_t top_k, uint64_t *rng, size_t *token);
#endif
