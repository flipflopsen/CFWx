/**
 * @file include/models/transformer.h
 * @brief transformer interfaces and implementation.
 * @ingroup transformer
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Composable double-precision Transformer reference building blocks. */
#ifndef CFWX_TRANSFORMER_H
#define CFWX_TRANSFORMER_H

#include "models/ann.h"
#include "models/attention.h"
#include "maths/activation_functions.h"
#include "maths/loss_functions.h"

/** All owners start as {0}; no shallow copying. Matrix arguments must be valid
 * library-created objects. Outputs are preallocated. Unless noted, buffers are
 * disjoint, including partial overlaps. Failure may partially modify outputs.
 * Sequence matrices are [tokens,features]; batches are processed by the caller.
 * Analytic composite backward APIs are declared in models/training.h.
 * Existing dense backward remains available separately. */
/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef enum
{
    CFWX_ACT_IDENTITY, /**< cfwx act identity enum choice. */
    CFWX_ACT_RELU, /**< cfwx act relu enum choice. */
    CFWX_ACT_GELU, /**< cfwx act gelu enum choice. */
    CFWX_ACT_SILU, /**< cfwx act silu enum choice. */
    CFWX_ACT_TANH, /**< cfwx act tanh enum choice. */
    CFWX_ACT_SIGMOID /**< cfwx act sigmoid enum choice. */
} cfwx_activation;

/**
 * @anchor api_include_models_transformer_h_cfwx_activate
 * @brief Perform the activate operation.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param kind kind argument; interpretation follows this operation's implementation and contract.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_activate(const cfwx_matrix *x, cfwx_activation kind, cfwx_matrix *out);
/** Uses preactivation x and upstream dy. No implicit loss reduction. */
/**
 * @anchor api_include_models_transformer_h_cfwx_activate_backward
 * @brief Compute analytic adjoints for activate.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Uses preactivation x and upstream dy. No implicit loss reduction.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param dy Borrowed upstream adjoint; same output shape as the corresponding forward operation.
 * @param kind kind argument; interpretation follows this operation's implementation and contract.
 * @param dx Preallocated analytic gradient output; shape matches the corresponding primal.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_activate_backward(const cfwx_matrix *x, const cfwx_matrix *dy,
                                   cfwx_activation kind, cfwx_matrix *dx);
/** Exact out==a or out==b allowed. */
/**
 * @anchor api_include_models_transformer_h_cfwx_residual_add
 * @brief Perform the residual add operation.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Exact out==a or out==b allowed.
 * @param a First borrowed numerical operand or internal node/state as declared.
 * @param b Second borrowed numerical operand or internal node/state as declared.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_residual_add(const cfwx_matrix *a, const cfwx_matrix *b, cfwx_matrix *out);
/** Inverted dropout: training survivors scaled by 1/(1-p). Caller-owned RNG;
 * optional mask stores multiplier for backward. Evaluation consumes no RNG.
 * out may alias x exactly; mask must be disjoint. p in [0,1). */
/**
 * @anchor api_include_models_transformer_h_cfwx_dropout
 * @brief Perform the dropout operation.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Inverted dropout: training survivors scaled by 1/(1-p). Caller-owned RNG; optional mask stores multiplier for backward. Evaluation consumes no RNG. out may alias x exactly; mask must be disjoint. p in [0,1).
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param p Parameter registry, dropout probability, or operand as declared; see the operation-specific formula.
 * @param training Training/evaluation mode; evaluation disables dropout and parameter updates.
 * @param seed Caller-supplied pseudorandom initialization seed or mutable RNG state; noncryptographic.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @param mask Optional attention permission mask or dropout multiplier matrix; see operation-specific contract.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_dropout(const cfwx_matrix *x, double p, bool training, uint64_t *seed,
                         cfwx_matrix *out, cfwx_matrix *mask);
/**
 * @anchor api_include_models_transformer_h_cfwx_dropout_backward
 * @brief Compute analytic adjoints for dropout.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param dy Borrowed upstream adjoint; same output shape as the corresponding forward operation.
 * @param mask Optional attention permission mask or dropout multiplier matrix; see operation-specific contract.
 * @param dx Preallocated analytic gradient output; shape matches the corresponding primal.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_dropout_backward(const cfwx_matrix *dy, const cfwx_matrix *mask, cfwx_matrix *dx);
/** Bias-free linear operator. Weights [input,output], x [tokens,input]. */
/**
 * @anchor api_include_models_transformer_h_cfwx_linear
 * @brief Perform the linear operation.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Bias-free linear operator. Weights [input,output], x [tokens,input].
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param weights Borrowed row-major projection weights [input,output].
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_linear(const cfwx_matrix *x, const cfwx_matrix *weights, cfwx_matrix *out);

/** Embeddings reuse a dense matrix as an owning [vocabulary,features] table. */
/**
 * @anchor api_include_models_transformer_h_cfwx_embedding
 * @brief Perform the embedding operation.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Embeddings reuse a dense matrix as an owning [vocabulary,features] table.
 * @param table table argument; interpretation follows this operation's implementation and contract.
 * @param ids Borrowed token/index array with one entry per sequence position; validate against vocabulary/table bounds.
 * @param count Number of entries/elements; must match the associated buffer or registry contract.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_embedding(const cfwx_matrix *table, const size_t *ids, size_t count,
                           cfwx_matrix *out);
/** Overwrite table gradient, accumulating repeated token IDs. */
/**
 * @anchor api_include_models_transformer_h_cfwx_embedding_backward
 * @brief Compute analytic adjoints for embedding.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Overwrite table gradient, accumulating repeated token IDs.
 * @param ids Borrowed token/index array with one entry per sequence position; validate against vocabulary/table bounds.
 * @param count Number of entries/elements; must match the associated buffer or registry contract.
 * @param dy Borrowed upstream adjoint; same output shape as the corresponding forward operation.
 * @param table_gradient table_gradient argument; interpretation follows this operation's implementation and contract.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_embedding_backward(const size_t *ids, size_t count, const cfwx_matrix *dy,
                                    cfwx_matrix *table_gradient);
/** Standard paired sin/cos frequencies; odd width permitted. */
/**
 * @anchor api_include_models_transformer_h_cfwx_sinusoidal_positions
 * @brief Perform the sinusoidal positions operation.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Standard paired sin/cos frequencies; odd width permitted.
 * @param offset Position offset or checked flattened tensor offset, as specified by this operation.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_sinusoidal_positions(size_t offset, cfwx_matrix *out);
/** RoPE uses adjacent feature pairs, even width; in-place supported. */
/**
 * @anchor api_include_models_transformer_h_cfwx_rope
 * @brief Perform the rope operation.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * RoPE uses adjacent feature pairs, even width; in-place supported.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param offset Position offset or checked flattened tensor offset, as specified by this operation.
 * @param base base argument; interpretation follows this operation's implementation and contract.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_rope(const cfwx_matrix *x, size_t offset, double base, cfwx_matrix *out);
/** RMS normalization is per row, gamma[features], epsilon>0. In-place allowed. */
/**
 * @anchor api_include_models_transformer_h_cfwx_rms_norm
 * @brief Perform the rms norm operation.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * RMS normalization is per row, gamma[features], epsilon>0. In-place allowed.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param gamma Feature scale array; one value per feature. Borrowed input storage.
 * @param epsilon Positive finite numerical-stability constant, not a trainable parameter.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_rms_norm(const cfwx_matrix *x, const double *gamma, double epsilon,
                          cfwx_matrix *out);

/** mask is optional [queries,keys]: true means allowed. Causal additionally
 * allows key<=query and requires equal lengths. Fully masked rows fail with
 * CFWX_SHAPE, never NaN/uniform probabilities. bias is finite additive logits.
 * Probabilities optional [queries,keys]. Outputs disjoint from inputs. */
/**
 * @anchor api_include_models_transformer_h_cfwx_masked_attention
 * @brief Perform the masked attention operation.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * mask is optional [queries,keys]: true means allowed. Causal additionally allows key<=query and requires equal lengths. Fully masked rows fail with CFWX_SHAPE, never NaN/uniform probabilities. bias is finite additive logits. Probabilities optional [queries,keys]. Outputs disjoint from inputs.
 * @param q Query operand [queries,head_depth] for attention, or corresponding model query state.
 * @param k Key operand [keys,head_depth] for attention.
 * @param v Value operand [keys,value_depth] for attention.
 * @param mask Optional attention permission mask or dropout multiplier matrix; see operation-specific contract.
 * @param bias bias argument; interpretation follows this operation's implementation and contract.
 * @param causal Whether to additionally permit only key positions at or before each query.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @param probabilities Optional preallocated attention probability matrix [queries,keys].
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_masked_attention(const cfwx_matrix *q, const cfwx_matrix *k, const cfwx_matrix *v,
                                  const bool *mask, const cfwx_matrix *bias, bool causal,
                                  cfwx_matrix *out, cfwx_matrix *probabilities);

/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef struct
{
    size_t width, heads; /**< State/dimension/policy value; see the containing type contract. */
    cfwx_dense query, key, value, output; /**< State/dimension/policy value; see the containing type contract. */
} cfwx_multihead;
/**
 * @anchor api_include_models_transformer_h_cfwx_multihead_init
 * @brief Construct and initialize multihead state.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param layer Caller-owned destination; must be zero-initialized/empty before construction. Successful construction transfers owned storage into it.
 * @param width Positive model-feature width divisible by the configured head count.
 * @param heads Positive attention head count dividing model width.
 * @param seed Caller-supplied pseudorandom initialization seed or mutable RNG state; noncryptographic.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_multihead_init(cfwx_multihead *layer, size_t width, size_t heads, uint64_t seed);
/**
 * @anchor api_include_models_transformer_h_cfwx_multihead_free
 * @brief Release owned multihead storage and reset its owner.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param layer Owner whose allocated storage is released and state reset; NULL is accepted where the implementation documents/checks it.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void cfwx_multihead_free(cfwx_multihead *layer);
/** Self or cross attention: query input [queries,width], context [keys,width]. */
/**
 * @anchor api_include_models_transformer_h_cfwx_multihead_forward
 * @brief Project Q/K/V, evaluate independent heads, concatenate and project outputs.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Self or cross attention: query input [queries,width], context [keys,width].
 * @param layer Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param query Numerical operand or operation state; see the documented mathematical shape and role.
 * @param context Common context/chunk length in tokens.
 * @param mask Optional attention permission mask or dropout multiplier matrix; see operation-specific contract.
 * @param causal Whether to additionally permit only key positions at or before each query.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_multihead_forward(const cfwx_multihead *layer, const cfwx_matrix *query,
                                   const cfwx_matrix *context, const bool *mask, bool causal,
                                   cfwx_matrix *out);

/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef struct
{
    cfwx_dense up, down, gate; /**< State/dimension/policy value; see the containing type contract. */
    cfwx_activation activation; /**< State/dimension/policy value; see the containing type contract. */
    bool gated; /**< State/dimension/policy value; see the containing type contract. */
} cfwx_feedforward;
/** gated=true computes down(activation(gate(x))*up(x)); SiLU gives SwiGLU. */
/**
 * @anchor api_include_models_transformer_h_cfwx_feedforward_init
 * @brief Construct and initialize feedforward state.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * gated=true computes down(activation(gate(x))*up(x)); SiLU gives SwiGLU.
 * @param layer Caller-owned destination; must be zero-initialized/empty before construction. Successful construction transfers owned storage into it.
 * @param width Positive model-feature width divisible by the configured head count.
 * @param hidden Positive feed-forward hidden-feature width.
 * @param activation activation argument; interpretation follows this operation's implementation and contract.
 * @param gated gated argument; interpretation follows this operation's implementation and contract.
 * @param seed Caller-supplied pseudorandom initialization seed or mutable RNG state; noncryptographic.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_feedforward_init(cfwx_feedforward *layer, size_t width, size_t hidden,
                                  cfwx_activation activation, bool gated, uint64_t seed);
/**
 * @anchor api_include_models_transformer_h_cfwx_feedforward_free
 * @brief Release owned feedforward storage and reset its owner.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param layer Owner whose allocated storage is released and state reset; NULL is accepted where the implementation documents/checks it.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void cfwx_feedforward_free(cfwx_feedforward *layer);
/**
 * @anchor api_include_models_transformer_h_cfwx_feedforward_forward
 * @brief Evaluate the configured ungated or gated feed-forward block.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param layer Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_feedforward_forward(const cfwx_feedforward *layer, const cfwx_matrix *x,
                                     cfwx_matrix *out);

/** @brief Record/enumeration used by this module; member contracts describe storage and policy. */
typedef struct
{
    cfwx_multihead attention; /**< State/dimension/policy value; see the containing type contract. */
    cfwx_feedforward feedforward; /**< State/dimension/policy value; see the containing type contract. */
    cfwx_vector gamma1, beta1, gamma2, beta2; /**< State/dimension/policy value; see the containing type contract. */
    double epsilon; /**< State/dimension/policy value; see the containing type contract. */
} cfwx_transformer_block;
/** Pre-LayerNorm block: x+MHA(LN(x)), then residual+FFN(LN(residual)).
 * Deterministic evaluation; compose dropout externally when training.
 * This is an encoder/causal self-attention block, not a full decoder with
 * cross-attention. Training APIs are in models/training.h. */
/**
 * @anchor api_include_models_transformer_h_cfwx_transformer_block_init
 * @brief Construct and initialize transformer block state.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Pre-LayerNorm block: x+MHA(LN(x)), then residual+FFN(LN(residual)). Deterministic evaluation; compose dropout externally when training. This is an encoder/causal self-attention block, not a full decoder with cross-attention. Training APIs are in models/training.h.
 * @param block Caller-owned destination; must be zero-initialized/empty before construction. Successful construction transfers owned storage into it.
 * @param width Positive model-feature width divisible by the configured head count.
 * @param heads Positive attention head count dividing model width.
 * @param hidden Positive feed-forward hidden-feature width.
 * @param seed Caller-supplied pseudorandom initialization seed or mutable RNG state; noncryptographic.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_transformer_block_init(cfwx_transformer_block *block, size_t width, size_t heads,
                                        size_t hidden, uint64_t seed);
/**
 * @anchor api_include_models_transformer_h_cfwx_transformer_block_free
 * @brief Release owned transformer block storage and reset its owner.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param block Owner whose allocated storage is released and state reset; NULL is accepted where the implementation documents/checks it.
 * @note This operation returns no value; inspect its documented mutation/lifetime effects.
 */
void cfwx_transformer_block_free(cfwx_transformer_block *block);
/**
 * @anchor api_include_models_transformer_h_cfwx_transformer_block_forward
 * @brief Evaluate both pre-LayerNorm residual branches of a Transformer block.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param block Model/operation state; input versus owner role follows the declaration and operation contract.
 * @param x Borrowed primal/input numeric operand; shape is stated in the operation contract.
 * @param mask Optional attention permission mask or dropout multiplier matrix; see operation-specific contract.
 * @param causal Whether to additionally permit only key positions at or before each query.
 * @param out Caller-supplied output storage; allocate its required shape before calling. Constructor APIs instead receive an owner destination.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_transformer_block_forward(const cfwx_transformer_block *block,
                                           const cfwx_matrix *x, const bool *mask, bool causal,
                                           cfwx_matrix *out);
/** Row-wise categorical CE, mean over nonignored tokens. SIZE_MAX is a useful
 * ignore label. Smoothing in [0,1); uniform smoothing includes target class.
 * Optional gradient [tokens,vocabulary] includes mean reduction. */
/**
 * @anchor api_include_models_transformer_h_cfwx_token_cross_entropy
 * @brief Compute token-mean categorical loss with ignore labels and smoothing.
 * @ingroup transformer
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @par Operation contract
 * Row-wise categorical CE, mean over nonignored tokens. SIZE_MAX is a useful ignore label. Smoothing in [0,1); uniform smoothing includes target class. Optional gradient [tokens,vocabulary] includes mean reduction.
 * @param logits Logit matrix/vector; dimensionality and input/output role are specified by the operation contract.
 * @param labels Categorical target IDs, one per row; ignore label is interpreted where supported.
 * @param ignore_label Target ID excluded from token-loss reduction; SIZE_MAX is the conventional sentinel.
 * @param smoothing Uniform label-smoothing fraction in [0,1); target class is included in uniform mass.
 * @param loss Output scalar objective; consult scalar-element versus token-mean reduction.
 * @param gradient Architecture/shape-matched preallocated gradient owner; NULL only where explicitly supported.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
cfwx_status cfwx_token_cross_entropy(const cfwx_matrix *logits, const size_t *labels,
                                     size_t ignore_label, double smoothing, double *loss,
                                     cfwx_matrix *gradient);
#endif
