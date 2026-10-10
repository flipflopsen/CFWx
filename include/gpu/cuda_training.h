/**
 * @file include/gpu/cuda_training.h
 * @brief cuda training interfaces and implementation.
 * @ingroup gpu
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Optional CUDA training ABI. No CUDA headers required by C callers. */
#ifndef CFWX_CUDA_TRAINING_H
#define CFWX_CUDA_TRAINING_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C"
{
#endif

/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
    typedef struct cfwx_cuda_model cfwx_cuda_model;
/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
    typedef struct
    {
        size_t vocabulary, width, heads, hidden, depth, context, batch; /**< State/dimension/policy value; see the containing type contract. */
        uint64_t seed; /**< State/dimension/policy value; see the containing type contract. */
        double dropout; /**< State/dimension/policy value; see the containing type contract. */
        /* Hard allocation limits. Graph construction fails rather than exceeding
         * these budgets. Device budget includes tape and parameter optimizer arrays. */
        size_t max_nodes, device_budget_bytes; /**< State/dimension/policy value; see the containing type contract. */
    } cfwx_cuda_config;
    /* Returns 0 on success. Error text remains valid until next API call on model.
     * Uses current CUDA device without any device-property/hardware probe.
     * No CPU numerical fallback. Parameters initialized on host then uploaded.
     * Double precision reference implementation, not Tensor Core/FlashAttention.
     * Stream and all buffers owned by model; free(NULL) is safe.
     * Requires positive dimensions, width%heads==0, dropout in [0,1), depth<=8. */
/**
 * @anchor api_include_gpu_cuda_training_h_cfwx_cuda_create
 * @brief Create the opt-in device reference model and owned CUDA resources.
 * @ingroup gpu
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Returns 0 on success. Error text remains valid until next API call on model. Uses current CUDA device without any device-property/hardware probe. No CPU numerical fallback. Parameters initialized on host then uploaded. Double precision reference implementation, not Tensor Core/FlashAttention. Stream and all buffers owned by model; free(NULL) is safe. Requires positive dimensions, width%heads==0, dropout in [0,1), depth<=8.
 * @param config Borrowed CUDA model configuration including dimensions and allocation budgets.
 * @param out Output pointer to a newly created opaque device model; initialized to NULL on entry.
 * @param error Caller-allocated diagnostic string buffer.
 * @param error_capacity Diagnostic buffer capacity in bytes, including the terminator.
 * @return Zero on success, nonzero on invalid arguments, allocation, CUDA API or numerical failure.
 * @warning CUDA source remains manually validated only. Doxygen parses this
 * symbol without running nvcc, CUDA kernels or hardware probes.
 */
    int cfwx_cuda_create(const cfwx_cuda_config *config, cfwx_cuda_model **out, char *error,
                         size_t error_capacity);
    /* Opt-in launch-fused sequence parallelism; serial interpreter is default.
     * FP64 atomics make gradient reduction order nondeterministic. Not fused GEMM.
     * Set before a run; model calls remain single-threaded. */
/**
 * @anchor api_include_gpu_cuda_training_h_cfwx_cuda_set_fused_batching
 * @brief Select launch-fused independent-sequence execution rather than serial interpretation.
 * @ingroup gpu
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Opt-in launch-fused sequence parallelism; serial interpreter is default. FP64 atomics make gradient reduction order nondeterministic. Not fused GEMM. Set before a run; model calls remain single-threaded.
 * @param model Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param enabled Nonzero selects the opt-in fused sequence launch path; zero selects serial execution.
 * @return Zero on success, nonzero on invalid arguments, allocation, CUDA API or numerical failure.
 * @warning CUDA source remains manually validated only. Doxygen parses this
 * symbol without running nvcc, CUDA kernels or hardware probes.
 */
    int cfwx_cuda_set_fused_batching(cfwx_cuda_model *model, int enabled);
/**
 * @anchor api_include_gpu_cuda_training_h_cfwx_cuda_free
 * @brief Release owned cuda storage and reset its owner.
 * @ingroup gpu
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param model Owner whose allocated storage is released and state reset; NULL is accepted where the implementation documents/checks it.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 * @warning CUDA source remains manually validated only. Doxygen parses this
 * symbol without running nvcc, CUDA kernels or hardware probes.
 */
    void cfwx_cuda_free(cfwx_cuda_model *model);
/**
 * @anchor api_include_gpu_cuda_training_h_cfwx_cuda_error
 * @brief Perform the cuda error operation.
 * @ingroup gpu
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param model Model/operation state; input versus owner role follows the declaration and operation contract.
 * @return Borrowed last-error text; valid until the next model API call or destruction.
 * @warning CUDA source remains manually validated only. Doxygen parses this
 * symbol without running nvcc, CUDA kernels or hardware probes.
 */
    const char *cfwx_cuda_error(const cfwx_cuda_model *model);
/**
 * @anchor api_include_gpu_cuda_training_h_cfwx_cuda_parameter_count
 * @brief Perform the cuda parameter count operation.
 * @ingroup gpu
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param model Model/operation state; input versus owner role follows the declaration and operation contract.
 * @return Flattened scalar parameter count; zero for a NULL model.
 * @warning CUDA source remains manually validated only. Doxygen parses this
 * symbol without running nvcc, CUDA kernels or hardware probes.
 */
    size_t cfwx_cuda_parameter_count(const cfwx_cuda_model *model);
    /* Input/target arrays are flattened [batch,context], IDs in [0,vocabulary).
     * Training: complete causal LM forward/backward, residual dropout, token-mean
     * CE, global clipping and AdamW, all numerically executed on device. Calls
     * synchronize only at API boundary. Batch contributes to one joint graph/loss.
     * Evaluation: dropout disabled; no gradient/update; optimizer RNG unchanged.
     * Rate=0 in training computes gradients without updating (for gradient tests).
     * Last loss is PRE-update. Optimizer update step advances only for rate>0.
     * Numerical failure poisons the model until destroyed; do not save/reuse it. */
/**
 * @anchor api_include_gpu_cuda_training_h_cfwx_cuda_run
 * @brief Execute the device batch objective and optional analytic update.
 * @ingroup gpu
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Input/target arrays are flattened [batch,context], IDs in [0,vocabulary). Training: complete causal LM forward/backward, residual dropout, token-mean CE, global clipping and AdamW, all numerically executed on device. Calls synchronize only at API boundary. Batch contributes to one joint graph/loss. Evaluation: dropout disabled; no gradient/update; optimizer RNG unchanged. Rate=0 in training computes gradients without updating (for gradient tests). Last loss is PRE-update. Optimizer update step advances only for rate>0. Numerical failure poisons the model until destroyed; do not save/reuse it.
 * @param model Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param tokens Borrowed token IDs, laid out as sequence/batch arrays according to this API.
 * @param targets Borrowed next-token targets; SIZE_MAX is ignored only where the loss API explicitly supports it.
 * @param training Training/evaluation mode; evaluation disables dropout and parameter updates.
 * @param rate Optimizer learning rate; zero is allowed only by APIs documenting gradient-only execution.
 * @param clip Positive finite global gradient-clipping threshold.
 * @param decay Finite nonnegative decoupled weight-decay coefficient.
 * @param loss Output scalar objective; consult scalar-element versus token-mean reduction.
 * @return Zero on success, nonzero on invalid arguments, allocation, CUDA API or numerical failure.
 * @warning CUDA source remains manually validated only. Doxygen parses this
 * symbol without running nvcc, CUDA kernels or hardware probes.
 */
    int cfwx_cuda_run(cfwx_cuda_model *model, const size_t *tokens, const size_t *targets,
                      int training, double rate, double clip, double decay, double *loss);
    /* Copies flattened parameter/gradient arrays for manual parity/finite differences.
     * copy length must equal parameter_count. No optimizer/checkpoint conversion to
     * the CPU model format: this is a distinct backend with its own storage. */
/**
 * @anchor api_include_gpu_cuda_training_h_cfwx_cuda_get_parameters
 * @brief Perform the cuda get parameters operation.
 * @ingroup gpu
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Copies flattened parameter/gradient arrays for manual parity/finite differences. copy length must equal parameter_count. No optimizer/checkpoint conversion to the CPU model format: this is a distinct backend with its own storage.
 * @param model Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param out Host destination double array; count must equal the flattened CUDA parameter count.
 * @param count Number of entries/elements; must match the associated buffer or registry contract.
 * @return Zero on success, nonzero on invalid arguments, allocation, CUDA API or numerical failure.
 * @warning CUDA source remains manually validated only. Doxygen parses this
 * symbol without running nvcc, CUDA kernels or hardware probes.
 */
    int cfwx_cuda_get_parameters(cfwx_cuda_model *model, double *out, size_t count);
/**
 * @anchor api_include_gpu_cuda_training_h_cfwx_cuda_set_parameters
 * @brief Perform the cuda set parameters operation.
 * @ingroup gpu
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param model Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param values Borrowed numerical values to import or process; length must match the supplied count.
 * @param count Number of entries/elements; must match the associated buffer or registry contract.
 * @return Zero on success, nonzero on invalid arguments, allocation, CUDA API or numerical failure.
 * @warning CUDA source remains manually validated only. Doxygen parses this
 * symbol without running nvcc, CUDA kernels or hardware probes.
 */
    int cfwx_cuda_set_parameters(cfwx_cuda_model *model, const double *values, size_t count);
/**
 * @anchor api_include_gpu_cuda_training_h_cfwx_cuda_get_gradients
 * @brief Perform the cuda get gradients operation.
 * @ingroup gpu
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param model Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param out Host destination double array; count must equal the flattened CUDA parameter count.
 * @param count Number of entries/elements; must match the associated buffer or registry contract.
 * @return Zero on success, nonzero on invalid arguments, allocation, CUDA API or numerical failure.
 * @warning CUDA source remains manually validated only. Doxygen parses this
 * symbol without running nvcc, CUDA kernels or hardware probes.
 */
    int cfwx_cuda_get_gradients(cfwx_cuda_model *model, double *out, size_t count);
#ifdef __cplusplus
}
#endif
#endif
