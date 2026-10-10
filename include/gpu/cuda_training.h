/* Optional CUDA training ABI. No CUDA headers required by C callers. */
#ifndef CFWX_CUDA_TRAINING_H
#define CFWX_CUDA_TRAINING_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct cfwx_cuda_model cfwx_cuda_model;
    typedef struct
    {
        size_t vocabulary, width, heads, hidden, depth, context, batch;
        uint64_t seed;
        double dropout;
        /* Hard allocation limits. Graph construction fails rather than exceeding
         * these budgets. Device budget includes tape and parameter optimizer arrays. */
        size_t max_nodes, device_budget_bytes;
    } cfwx_cuda_config;
    /* Returns 0 on success. Error text remains valid until next API call on model.
     * Uses current CUDA device without any device-property/hardware probe.
     * No CPU numerical fallback. Parameters initialized on host then uploaded.
     * Double precision reference implementation, not Tensor Core/FlashAttention.
     * Stream and all buffers owned by model; free(NULL) is safe.
     * Requires positive dimensions, width%heads==0, dropout in [0,1), depth<=8. */
    int cfwx_cuda_create(const cfwx_cuda_config *config, cfwx_cuda_model **out, char *error,
                         size_t error_capacity);
    /* Opt-in launch-fused sequence parallelism; serial interpreter is default.
     * FP64 atomics make gradient reduction order nondeterministic. Not fused GEMM.
     * Set before a run; model calls remain single-threaded. */
    int cfwx_cuda_set_fused_batching(cfwx_cuda_model *model, int enabled);
    void cfwx_cuda_free(cfwx_cuda_model *model);
    const char *cfwx_cuda_error(const cfwx_cuda_model *model);
    size_t cfwx_cuda_parameter_count(const cfwx_cuda_model *model);
    /* Input/target arrays are flattened [batch,context], IDs in [0,vocabulary).
     * Training: complete causal LM forward/backward, residual dropout, token-mean
     * CE, global clipping and AdamW, all numerically executed on device. Calls
     * synchronize only at API boundary. Batch contributes to one joint graph/loss.
     * Evaluation: dropout disabled; no gradient/update; optimizer RNG unchanged.
     * Rate=0 in training computes gradients without updating (for gradient tests).
     * Last loss is PRE-update. Optimizer update step advances only for rate>0.
     * Numerical failure poisons the model until destroyed; do not save/reuse it. */
    int cfwx_cuda_run(cfwx_cuda_model *model, const size_t *tokens, const size_t *targets,
                      int training, double rate, double clip, double decay, double *loss);
    /* Copies flattened parameter/gradient arrays for manual parity/finite differences.
     * copy length must equal parameter_count. No optimizer/checkpoint conversion to
     * the CPU model format: this is a distinct backend with its own storage. */
    int cfwx_cuda_get_parameters(cfwx_cuda_model *model, double *out, size_t count);
    int cfwx_cuda_set_parameters(cfwx_cuda_model *model, const double *values, size_t count);
    int cfwx_cuda_get_gradients(cfwx_cuda_model *model, double *out, size_t count);
#ifdef __cplusplus
}
#endif
#endif
