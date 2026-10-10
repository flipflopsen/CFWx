/**
 * @file include/models/runtime.h
 * @brief runtime interfaces and implementation.
 * @ingroup runtime
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Training/inference lifecycle beyond individual Transformer blocks. */
#ifndef CFWX_RUNTIME_H
#define CFWX_RUNTIME_H
#include "models/training.h"

/** Caller RNG advances only for training dropout. Backward uses exactly the
 * stored forward multipliers within the same tape, never regenerated masks.
 * Dropout currently covers the two residual branches per block. */
/**
 * @anchor api_include_models_runtime_h_cfwx_language_model_loss_training
 * @brief Evaluate/differentiate a language model with replayed residual dropout masks.
 * @ingroup runtime
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Caller RNG advances only for training dropout. Backward uses exactly the stored forward multipliers within the same tape, never regenerated masks. Dropout currently covers the two residual branches per block.
 * @param model Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param tokens Borrowed token IDs, laid out as sequence/batch arrays according to this API.
 * @param targets Borrowed next-token targets; SIZE_MAX is ignored only where the loss API explicitly supports it.
 * @param length Number of sequence positions/tokens processed by this call.
 * @param dropout Training dropout probability in [0,1); exact replay uses the forward mask.
 * @param rng Borrowed mutable RNG state; consult the operation contract for consumption/replay rules.
 * @param loss Output scalar objective; consult scalar-element versus token-mean reduction.
 * @param gradient Architecture/shape-matched preallocated gradient owner; NULL only where explicitly supported.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_language_model_loss_training(const cfwx_language_model *model,
                                              const size_t *tokens, const size_t *targets,
                                              size_t length, double dropout, uint64_t *rng,
                                              double *loss, cfwx_language_model *gradient);

/** Equal/unequal sequence lengths: use lengths[b] or common context if NULL.
 * Arrays hold per-sequence pointers; SIZE_MAX targets are ignored. Gradients
 * are token-weighted across the complete batch, overwritten once per call.
 * This reference implementation accumulates sequences on CPU; not fused GPU
 * batching. Dropout randomness uses the caller-owned stream. */
/**
 * @anchor api_include_models_runtime_h_cfwx_language_model_batch
 * @brief Accumulate independent-sequence gradients weighted by valid token count.
 * @ingroup runtime
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Equal/unequal sequence lengths: use lengths[b] or common context if NULL. Arrays hold per-sequence pointers; SIZE_MAX targets are ignored. Gradients are token-weighted across the complete batch, overwritten once per call. This reference implementation accumulates sequences on CPU; not fused GPU batching. Dropout randomness uses the caller-owned stream.
 * @param model Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param tokens Borrowed token IDs, laid out as sequence/batch arrays according to this API.
 * @param targets Borrowed next-token targets; SIZE_MAX is ignored only where the loss API explicitly supports it.
 * @param lengths Optional per-sequence lengths; NULL selects the common context length.
 * @param batch Positive number of independent sequences; reduction semantics are specified by the API.
 * @param context Common context/chunk length in tokens.
 * @param dropout Training dropout probability in [0,1); exact replay uses the forward mask.
 * @param rng Borrowed mutable RNG state; consult the operation contract for consumption/replay rules.
 * @param loss Output scalar objective; consult scalar-element versus token-mean reduction.
 * @param gradient Architecture/shape-matched preallocated gradient owner; NULL only where explicitly supported.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_language_model_batch(const cfwx_language_model *model, const size_t *const *tokens,
                                      const size_t *const *targets, const size_t *lengths,
                                      size_t batch, size_t context, double dropout, uint64_t *rng,
                                      double *loss, cfwx_language_model *gradient);
/** Full nonoverlapping held-out evaluation: every next-token target counted
 * once, including the short final chunk. Positions reset at chunk boundaries.
 * No dropout/RNG consumption. Returns token-mean CE and number scored. */
/**
 * @anchor api_include_models_runtime_h_cfwx_language_model_evaluate
 * @brief Evaluate every adjacent held-out target once using chunked contexts.
 * @ingroup runtime
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Full nonoverlapping held-out evaluation: every next-token target counted once, including the short final chunk. Positions reset at chunk boundaries. No dropout/RNG consumption. Returns token-mean CE and number scored.
 * @param model Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param tokens Borrowed token IDs, laid out as sequence/batch arrays according to this API.
 * @param count Number of entries/elements; must match the associated buffer or registry contract.
 * @param context Common context/chunk length in tokens.
 * @param loss Output scalar objective; consult scalar-element versus token-mean reduction.
 * @param scored Output count of nonignored/held-out next-token targets included in reduction.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_language_model_evaluate(const cfwx_language_model *model, const size_t *tokens,
                                         size_t count, size_t context, double *loss,
                                         size_t *scored);

/** Complete experiment state saved with parameters/moments: vocabulary mapping,
 * dataset fingerprint, sampler/dropout RNG, and optimizer hyperparameters.
 * Byte map values are [0,vocabulary) or SIZE_MAX; unknown is a valid token ID.
 * Fingerprint is a reproducibility guard, not cryptographic authentication. */
/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef struct
{
    uint64_t rng, dataset_hash; /**< State/dimension/policy value; see the containing type contract. */
    size_t context, batch, unknown, byte_map[256]; /**< State/dimension/policy value; see the containing type contract. */
    double dropout, rate, beta1, beta2, epsilon, decay, clip; /**< State/dimension/policy value; see the containing type contract. */
} cfwx_training_state;
/**
 * @anchor api_include_models_runtime_h_cfwx_dataset_hash
 * @brief Perform the dataset hash operation.
 * @ingroup runtime
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param data Borrowed source bytes/numerical storage; valid for the stated count.
 * @param count Number of entries/elements; must match the associated buffer or registry contract.
 * @return Computed value or status described by the adjacent contract.
 */
uint64_t cfwx_dataset_hash(const unsigned char *data, size_t count);
/** Versioned binary, little-endian uint64 and IEEE-754 binary64, FNV checksum.
 * Save uses same-directory temporary file and rename; single writer only.
 * Load requires zero-initialized model/optimizer, validates before committing.
 * Limits: width<=4096, depth<=8, vocabulary<=65536, serialized payload<=256MiB.
 * No gradient/tape/cache saved; they are reconstructed after loading. */
/**
 * @anchor api_include_models_runtime_h_cfwx_checkpoint_save
 * @brief Publish a versioned CPU checkpoint with payload and parent-directory synchronization.
 * @ingroup runtime
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Versioned binary, little-endian uint64 and IEEE-754 binary64, FNV checksum. Save uses same-directory temporary file and rename; single writer only. Load requires zero-initialized model/optimizer, validates before committing. Limits: width<=4096, depth<=8, vocabulary<=65536, serialized payload<=256MiB. No gradient/tape/cache saved; they are reconstructed after loading.
 * @param path File path used by the operation; checkpoint publication requires a writable parent directory.
 * @param model Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param optimizer AdamW state owner with shape-matched moments and one shared update counter.
 * @param state Experiment state including vocabulary, RNG, dataset fingerprint and hyperparameters.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_checkpoint_save(const char *path, cfwx_language_model *model,
                                 const cfwx_adamw *optimizer, const cfwx_training_state *state);
/**
 * @anchor api_include_models_runtime_h_cfwx_checkpoint_load
 * @brief Validate and transactionally construct CPU model, optimizer and experiment state.
 * @ingroup runtime
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param path File path used by the operation; checkpoint publication requires a writable parent directory.
 * @param model Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param optimizer AdamW state owner with shape-matched moments and one shared update counter.
 * @param state Experiment state including vocabulary, RNG, dataset fingerprint and hyperparameters.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_checkpoint_load(const char *path, cfwx_language_model *model,
                                 cfwx_adamw *optimizer, cfwx_training_state *state);

/** Real projected per-block key/value storage. Cache borrows an immutable model;
 * updating model weights invalidates cache: reset/reinitialize after training.
 * Token positions increase monotonically; fixed/sliding eviction is configurable.
 * Reset starts a new sequence without reallocating. One cache per sequence. */
/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef enum
{
    CFWX_CACHE_FIXED, /**< cfwx cache fixed enum choice. */
    CFWX_CACHE_SLIDING /**< cfwx cache sliding enum choice. */
} cfwx_cache_mode;

/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef struct
{
    const cfwx_language_model *model; /**< Borrowed pointer or owned storage as specified by the containing type. */
    size_t capacity, length; /**< State/dimension/policy value; see the containing type contract. */
    size_t position; /* Absolute position; does not reset on eviction. */ /**< Borrowed pointer or owned storage as specified by the containing type. */
    cfwx_cache_mode mode; /**< State/dimension/policy value; see the containing type contract. */
    cfwx_matrix keys[CFWX_MAX_BLOCKS], values[CFWX_MAX_BLOCKS]; /**< State/dimension/policy value; see the containing type contract. */
} cfwx_kv_cache;
/**
 * @anchor api_include_models_runtime_h_cfwx_kv_cache_init
 * @brief Initialize a fixed-capacity cache using the legacy default policy.
 * @ingroup runtime
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param cache Caller-owned destination; must be zero-initialized/empty before construction. Successful construction transfers owned storage into it.
 * @param model Caller-owned destination; must be zero-initialized/empty before construction. Successful construction transfers owned storage into it.
 * @param capacity Positive allocation/cache capacity or destination buffer capacity, in the units stated by the API.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_kv_cache_init(cfwx_kv_cache *cache, const cfwx_language_model *model,
                               size_t capacity);
/** Configurable mode; legacy init defaults to fixed. Sliding evicts oldest K/V
 * only when full. On failed full append committed entries are restored. */
/**
 * @anchor api_include_models_runtime_h_cfwx_kv_cache_init_mode
 * @brief Initialize an explicitly fixed or sliding projected K/V cache.
 * @ingroup runtime
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Configurable mode; legacy init defaults to fixed. Sliding evicts oldest K/V only when full. On failed full append committed entries are restored.
 * @param cache cache argument; interpretation follows this operation's implementation and contract.
 * @param model Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param capacity Positive allocation/cache capacity or destination buffer capacity, in the units stated by the API.
 * @param mode Fixed or sliding cache eviction policy.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_kv_cache_init_mode(cfwx_kv_cache *cache, const cfwx_language_model *model,
                                    size_t capacity, cfwx_cache_mode mode);
/**
 * @anchor api_include_models_runtime_h_cfwx_kv_cache_reset
 * @brief Perform the kv cache reset operation.
 * @ingroup runtime
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param cache cache argument; interpretation follows this operation's implementation and contract.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void cfwx_kv_cache_reset(cfwx_kv_cache *cache);
/**
 * @anchor api_include_models_runtime_h_cfwx_kv_cache_free
 * @brief Release owned kv cache storage and reset its owner.
 * @ingroup runtime
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param cache Owner whose allocated storage is released and state reset; NULL is accepted where the implementation documents/checks it.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void cfwx_kv_cache_free(cfwx_kv_cache *cache);
/** Append one token, compute logits [1,vocabulary] for the next-token distribution.
 * Cache length advances only on success; failed slot can be overwritten safely. */
/**
 * @anchor api_include_models_runtime_h_cfwx_language_model_decode
 * @brief Append one token and evaluate next-token logits using projected K/V history.
 * @ingroup runtime
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Append one token, compute logits [1,vocabulary] for the next-token distribution. Cache length advances only on success; failed slot can be overwritten safely.
 * @param cache cache argument; interpretation follows this operation's implementation and contract.
 * @param token Input token ID below model vocabulary; appended to the cache on success.
 * @param logits Preallocated logit output matrix: [length,vocabulary] for full forward or [1,vocabulary] for decode.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_language_model_decode(cfwx_kv_cache *cache, size_t token, cfwx_matrix *logits);
/** Stable temperature sampling. temperature=0 means greedy and consumes no RNG.
 * top_k=0 keeps all classes; other values must not exceed vocabulary size. */
/**
 * @anchor api_include_models_runtime_h_cfwx_sample_logits
 * @brief Sample finite logits with greedy/temperature and optional top-k filtering.
 * @ingroup runtime
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Stable temperature sampling. temperature=0 means greedy and consumes no RNG. top_k=0 keeps all classes; other values must not exceed vocabulary size.
 * @param logits Logit matrix/vector; dimensionality and input/output role are specified by the operation contract.
 * @param vocabulary Positive vocabulary size or count of classes.
 * @param temperature Finite nonnegative sampling temperature; zero requests greedy decoding.
 * @param top_k Number of highest-logit classes retained; zero means all classes.
 * @param rng Borrowed mutable RNG state; consult the operation contract for consumption/replay rules.
 * @param token Output sampled categorical token ID.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_sample_logits(const double *logits, size_t vocabulary, double temperature,
                               size_t top_k, uint64_t *rng, size_t *token);
#endif
