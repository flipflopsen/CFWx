/**
 * @file include/models/training.h
 * @brief training interfaces and implementation.
 * @ingroup training
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Analytic reverse-mode training for the existing modular Transformer models. */
#ifndef CFWX_TRAINING_H
#define CFWX_TRAINING_H
#include "models/transformer.h"

/** Gradient owners have exactly the same shapes as their corresponding model.
 * Initialize them with the existing init functions and free with existing free
 * functions. Backward overwrites gradients, does not update model parameters.
 * No aliasing between models, gradients, inputs, upstream and outputs.
 * Forward intermediates are recomputed using current parameters: do not update
 * parameters or change inputs between forward and backward.
 * A bounded local reverse-mode tape analytically differentiates each primitive;
 * no finite differences are used in training. Allocation failures clean up.
 * On failure outputs/gradients may be partial; never apply them. */
/**
 * @anchor api_include_models_training_h_cfwx_attention_backward
 * @brief Compute analytic Q/K/V and optional attention-bias gradients.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Gradient owners have exactly the same shapes as their corresponding model. Initialize them with the existing init functions and free with existing free functions. Backward overwrites gradients, does not update model parameters. No aliasing between models, gradients, inputs, upstream and outputs. Forward intermediates are recomputed using current parameters: do not update parameters or change inputs between forward and backward. A bounded local reverse-mode tape analytically differentiates each primitive; no finite differences are used in training. Allocation failures clean up. On failure outputs/gradients may be partial; never apply them.
 * @param q Query operand [queries,head_depth] for attention, or corresponding model query state.
 * @param k Key operand [keys,head_depth] for attention.
 * @param v Value operand [keys,value_depth] for attention.
 * @param mask Optional attention permission mask or dropout multiplier matrix; see operation-specific contract.
 * @param bias bias argument; interpretation follows this operation's implementation and contract.
 * @param causal Whether to additionally permit only key positions at or before each query.
 * @param dy Borrowed upstream adjoint; same output shape as the corresponding forward operation.
 * @param dq Preallocated analytic gradient output; shape matches the corresponding primal.
 * @param dk Preallocated analytic gradient output; shape matches the corresponding primal.
 * @param dv Preallocated analytic gradient output; shape matches the corresponding primal.
 * @param dbias Preallocated analytic gradient output; shape matches the corresponding primal.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_attention_backward(const cfwx_matrix *q, const cfwx_matrix *k,
                                    const cfwx_matrix *v, const bool *mask, const cfwx_matrix *bias,
                                    bool causal, const cfwx_matrix *dy, cfwx_matrix *dq,
                                    cfwx_matrix *dk, cfwx_matrix *dv, cfwx_matrix *dbias);
/**
 * @anchor api_include_models_training_h_cfwx_layer_norm_backward
 * @brief Differentiate population LayerNorm input, scale and shift.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param gamma Feature scale array; one value per feature. Borrowed input storage.
 * @param epsilon Positive finite numerical-stability constant, not a trainable parameter.
 * @param dy Borrowed upstream adjoint; same output shape as the corresponding forward operation.
 * @param dx Preallocated analytic gradient output; shape matches the corresponding primal.
 * @param dgamma Caller-allocated feature-scale gradient output array.
 * @param dbeta Caller-allocated feature-shift gradient output array.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_layer_norm_backward(const cfwx_matrix *x, const double *gamma, double epsilon,
                                     const cfwx_matrix *dy, cfwx_matrix *dx, double *dgamma,
                                     double *dbeta);
/**
 * @anchor api_include_models_training_h_cfwx_rms_norm_backward
 * @brief Differentiate RMSNorm input and feature scale.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param gamma Feature scale array; one value per feature. Borrowed input storage.
 * @param epsilon Positive finite numerical-stability constant, not a trainable parameter.
 * @param dy Borrowed upstream adjoint; same output shape as the corresponding forward operation.
 * @param dx Preallocated analytic gradient output; shape matches the corresponding primal.
 * @param dgamma Caller-allocated feature-scale gradient output array.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_rms_norm_backward(const cfwx_matrix *x, const double *gamma, double epsilon,
                                   const cfwx_matrix *dy, cfwx_matrix *dx, double *dgamma);
/**
 * @anchor api_include_models_training_h_cfwx_rope_backward
 * @brief Compute analytic adjoints for rope.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param dy Borrowed upstream adjoint; same output shape as the corresponding forward operation.
 * @param offset Position offset or checked flattened tensor offset, as specified by this operation.
 * @param base base argument; interpretation follows this operation's implementation and contract.
 * @param dx Preallocated analytic gradient output; shape matches the corresponding primal.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_rope_backward(const cfwx_matrix *dy, size_t offset, double base, cfwx_matrix *dx);
/**
 * @anchor api_include_models_training_h_cfwx_multihead_backward
 * @brief Compute analytic adjoints for multihead.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param layer Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param query Numerical operand or operation state; see the documented mathematical shape and role.
 * @param context Common context/chunk length in tokens.
 * @param mask Optional attention permission mask or dropout multiplier matrix; see operation-specific contract.
 * @param causal Whether to additionally permit only key positions at or before each query.
 * @param dy Borrowed upstream adjoint; same output shape as the corresponding forward operation.
 * @param dquery Preallocated analytic gradient output; shape matches the corresponding primal.
 * @param dcontext Preallocated analytic gradient output; shape matches the corresponding primal.
 * @param gradient Architecture/shape-matched preallocated gradient owner; NULL only where explicitly supported.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_multihead_backward(const cfwx_multihead *layer, const cfwx_matrix *query,
                                    const cfwx_matrix *context, const bool *mask, bool causal,
                                    const cfwx_matrix *dy, cfwx_matrix *dquery,
                                    cfwx_matrix *dcontext, cfwx_multihead *gradient);
/**
 * @anchor api_include_models_training_h_cfwx_feedforward_backward
 * @brief Compute analytic adjoints for feedforward.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param layer Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param dy Borrowed upstream adjoint; same output shape as the corresponding forward operation.
 * @param dx Preallocated analytic gradient output; shape matches the corresponding primal.
 * @param gradient Architecture/shape-matched preallocated gradient owner; NULL only where explicitly supported.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_feedforward_backward(const cfwx_feedforward *layer, const cfwx_matrix *x,
                                      const cfwx_matrix *dy, cfwx_matrix *dx,
                                      cfwx_feedforward *gradient);
/**
 * @anchor api_include_models_training_h_cfwx_transformer_block_backward
 * @brief Compute analytic adjoints for transformer block.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param block Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param mask Optional attention permission mask or dropout multiplier matrix; see operation-specific contract.
 * @param causal Whether to additionally permit only key positions at or before each query.
 * @param dy Borrowed upstream adjoint; same output shape as the corresponding forward operation.
 * @param dx Preallocated analytic gradient output; shape matches the corresponding primal.
 * @param gradient Architecture/shape-matched preallocated gradient owner; NULL only where explicitly supported.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_transformer_block_backward(const cfwx_transformer_block *block,
                                            const cfwx_matrix *x, const bool *mask, bool causal,
                                            const cfwx_matrix *dy, cfwx_matrix *dx,
                                            cfwx_transformer_block *gradient);

/** Parameter registry supports deterministic enumeration, clipping and updates.
 * Parameter storage is borrowed. Build matching registries for model/gradient.
 * Trainable leaves include all projection weights/biases and norm gamma/beta. */
/** @def CFWX_MAX_PARAMETERS
 * @brief cfwx max parameters macro; see expansion and module contract.
 */
#define CFWX_MAX_PARAMETERS 512
/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef struct
{
    size_t count; /**< State/dimension/policy value; see the containing type contract. */
    double *data[CFWX_MAX_PARAMETERS]; /**< Borrowed pointer or owned storage as specified by the containing type. */
    size_t size[CFWX_MAX_PARAMETERS]; /**< State/dimension/policy value; see the containing type contract. */
} cfwx_parameters;
/**
 * @anchor api_include_models_training_h_cfwx_parameters_dense
 * @brief Perform the parameters dense operation.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param p Parameter registry, dropout probability, or operand as declared; see the operation-specific formula.
 * @param layer Model/operation state; input versus owner role follows the declaration and operation contract.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_parameters_dense(cfwx_parameters *p, cfwx_dense *layer);
/**
 * @anchor api_include_models_training_h_cfwx_parameters_block
 * @brief Perform the parameters block operation.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param p Parameter registry, dropout probability, or operand as declared; see the operation-specific formula.
 * @param block Model/operation state; input versus owner role follows the declaration and operation contract.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_parameters_block(cfwx_parameters *p, cfwx_transformer_block *block);
/**
 * @anchor api_include_models_training_h_cfwx_clip_gradients
 * @brief Perform the clip gradients operation.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param gradient Architecture/shape-matched preallocated gradient owner; NULL only where explicitly supported.
 * @param max_norm Positive finite maximum global gradient norm after accumulation.
 * @param norm Output for the measured pre-clipping global gradient norm.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_clip_gradients(cfwx_parameters *gradient, double max_norm, double *norm);

/** AdamW owns first/second moments, initialized against a registry. step is
 * shared across ALL tensors and incremented once per optimizer update.
 * Standard decoupled decay is applied to every registered tensor. Use decay=0
 * or separate optimizer groups to exclude biases/norm parameters. */
/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef struct
{
    size_t count, step; /**< State/dimension/policy value; see the containing type contract. */
    size_t size[CFWX_MAX_PARAMETERS]; /**< State/dimension/policy value; see the containing type contract. */
    double *first[CFWX_MAX_PARAMETERS], *second[CFWX_MAX_PARAMETERS]; /**< Borrowed pointer or owned storage as specified by the containing type. */
} cfwx_adamw;
/**
 * @anchor api_include_models_training_h_cfwx_adamw_init
 * @brief Construct and initialize adamw state.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param optimizer Caller-owned destination; must be zero-initialized/empty before construction. Successful construction transfers owned storage into it.
 * @param p Parameter registry, dropout probability, or operand as declared; see the operation-specific formula.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_adamw_init(cfwx_adamw *optimizer, const cfwx_parameters *p);
/**
 * @anchor api_include_models_training_h_cfwx_adamw_free
 * @brief Release owned adamw storage and reset its owner.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param optimizer Owner whose allocated storage is released and state reset; NULL is accepted where the implementation documents/checks it.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void cfwx_adamw_free(cfwx_adamw *optimizer);
/**
 * @anchor api_include_models_training_h_cfwx_adamw_step
 * @brief Apply one global AdamW update with bias-corrected moments.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param optimizer AdamW state owner with shape-matched moments and one shared update counter.
 * @param p Parameter registry, dropout probability, or operand as declared; see the operation-specific formula.
 * @param gradient Architecture/shape-matched preallocated gradient owner; NULL only where explicitly supported.
 * @param rate Optimizer learning rate; zero is allowed only by APIs documenting gradient-only execution.
 * @param beta1 Adam first-moment decay in [0,1).
 * @param beta2 Adam second-moment decay in [0,1).
 * @param epsilon Positive finite numerical-stability constant, not a trainable parameter.
 * @param decay Finite nonnegative decoupled weight-decay coefficient.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_adamw_step(cfwx_adamw *optimizer, cfwx_parameters *p,
                            const cfwx_parameters *gradient, double rate, double beta1,
                            double beta2, double epsilon, double decay);

/** Small trainable causal LM: embeddings + fixed sinusoidal positions + stack
 * of pre-LN blocks + final LayerNorm + untied vocabulary projection.
 * Dropout is disabled in this deterministic reference training path.
 * Max 8 blocks, one sequence per call, caller-owned tokenization/dataset. */
/** @def CFWX_MAX_BLOCKS
 * @brief cfwx max blocks macro; see expansion and module contract.
 */
#define CFWX_MAX_BLOCKS 8
/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef struct
{
    size_t vocabulary, width, depth; /**< State/dimension/policy value; see the containing type contract. */
    cfwx_matrix embeddings; /**< State/dimension/policy value; see the containing type contract. */
    cfwx_transformer_block blocks[CFWX_MAX_BLOCKS]; /**< State/dimension/policy value; see the containing type contract. */
    cfwx_vector gamma, beta; /**< State/dimension/policy value; see the containing type contract. */
    cfwx_dense head; /**< State/dimension/policy value; see the containing type contract. */
} cfwx_language_model;
/**
 * @anchor api_include_models_training_h_cfwx_language_model_init
 * @brief Construct and initialize language model state.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param model Caller-owned destination; must be zero-initialized/empty before construction. Successful construction transfers owned storage into it.
 * @param vocabulary Positive vocabulary size or count of classes.
 * @param width Positive model-feature width divisible by the configured head count.
 * @param heads Positive attention head count dividing model width.
 * @param hidden Positive feed-forward hidden-feature width.
 * @param depth Positive layer count, bounded by CFWX_MAX_BLOCKS in the language model.
 * @param seed Caller-supplied pseudorandom initialization seed or mutable RNG state; noncryptographic.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_language_model_init(cfwx_language_model *model, size_t vocabulary, size_t width,
                                     size_t heads, size_t hidden, size_t depth, uint64_t seed);
/**
 * @anchor api_include_models_training_h_cfwx_language_model_free
 * @brief Release owned language model storage and reset its owner.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param model Owner whose allocated storage is released and state reset; NULL is accepted where the implementation documents/checks it.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void cfwx_language_model_free(cfwx_language_model *model);
/**
 * @anchor api_include_models_training_h_cfwx_parameters_language_model
 * @brief Perform the parameters language model operation.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param p Parameter registry, dropout probability, or operand as declared; see the operation-specific formula.
 * @param model Model/operation state; input versus owner role follows the declaration and operation contract.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_parameters_language_model(cfwx_parameters *p, cfwx_language_model *model);
/**
 * @anchor api_include_models_training_h_cfwx_language_model_forward
 * @brief Perform the language model forward operation.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param model Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param tokens Borrowed token IDs, laid out as sequence/batch arrays according to this API.
 * @param length Number of sequence positions/tokens processed by this call.
 * @param logits Preallocated logit output matrix: [length,vocabulary] for full forward or [1,vocabulary] for decode.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_language_model_forward(const cfwx_language_model *model, const size_t *tokens,
                                        size_t length, cfwx_matrix *logits);
/** Mean unsmoothed next-token CE. With gradient!=NULL compute full analytic
 * gradients; otherwise evaluate only. targets has length entries. */
/**
 * @anchor api_include_models_training_h_cfwx_language_model_loss
 * @brief Evaluate token loss and optionally differentiate all CPU language-model parameters.
 * @ingroup training
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Mean unsmoothed next-token CE. With gradient!=NULL compute full analytic gradients; otherwise evaluate only. targets has length entries.
 * @param model Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param tokens Borrowed token IDs, laid out as sequence/batch arrays according to this API.
 * @param targets Borrowed next-token targets; SIZE_MAX is ignored only where the loss API explicitly supports it.
 * @param length Number of sequence positions/tokens processed by this call.
 * @param loss Output scalar objective; consult scalar-element versus token-mean reduction.
 * @param gradient Architecture/shape-matched preallocated gradient owner; NULL only where explicitly supported.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_language_model_loss(const cfwx_language_model *model, const size_t *tokens,
                                     const size_t *targets, size_t length, double *loss,
                                     cfwx_language_model *gradient);
#endif
