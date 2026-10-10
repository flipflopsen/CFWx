/* Composable double-precision Transformer reference building blocks. */
#ifndef CFWX_TRANSFORMER_H
#define CFWX_TRANSFORMER_H

#include "models/ann.h"
#include "models/attention.h"
#include "maths/activation_functions.h"
#include "maths/loss_functions.h"

/* All owners start as {0}; no shallow copying. Matrix arguments must be valid
 * library-created objects. Outputs are preallocated. Unless noted, buffers are
 * disjoint, including partial overlaps. Failure may partially modify outputs.
 * Sequence matrices are [tokens,features]; batches are processed by the caller.
 * Analytic composite backward APIs are declared in models/training.h.
 * Existing dense backward remains available separately. */
typedef enum
{
    CFWX_ACT_IDENTITY,
    CFWX_ACT_RELU,
    CFWX_ACT_GELU,
    CFWX_ACT_SILU,
    CFWX_ACT_TANH,
    CFWX_ACT_SIGMOID
} cfwx_activation;

cfwx_status cfwx_activate(const cfwx_matrix *x, cfwx_activation kind, cfwx_matrix *out);
/* Uses preactivation x and upstream dy. No implicit loss reduction. */
cfwx_status cfwx_activate_backward(const cfwx_matrix *x, const cfwx_matrix *dy,
                                   cfwx_activation kind, cfwx_matrix *dx);
/* Exact out==a or out==b allowed. */
cfwx_status cfwx_residual_add(const cfwx_matrix *a, const cfwx_matrix *b, cfwx_matrix *out);
/* Inverted dropout: training survivors scaled by 1/(1-p). Caller-owned RNG;
 * optional mask stores multiplier for backward. Evaluation consumes no RNG.
 * out may alias x exactly; mask must be disjoint. p in [0,1). */
cfwx_status cfwx_dropout(const cfwx_matrix *x, double p, bool training, uint64_t *seed,
                         cfwx_matrix *out, cfwx_matrix *mask);
cfwx_status cfwx_dropout_backward(const cfwx_matrix *dy, const cfwx_matrix *mask, cfwx_matrix *dx);
/* Bias-free linear operator. Weights [input,output], x [tokens,input]. */
cfwx_status cfwx_linear(const cfwx_matrix *x, const cfwx_matrix *weights, cfwx_matrix *out);

/* Embeddings reuse a dense matrix as an owning [vocabulary,features] table. */
cfwx_status cfwx_embedding(const cfwx_matrix *table, const size_t *ids, size_t count,
                           cfwx_matrix *out);
/* Overwrite table gradient, accumulating repeated token IDs. */
cfwx_status cfwx_embedding_backward(const size_t *ids, size_t count, const cfwx_matrix *dy,
                                    cfwx_matrix *table_gradient);
/* Standard paired sin/cos frequencies; odd width permitted. */
cfwx_status cfwx_sinusoidal_positions(size_t offset, cfwx_matrix *out);
/* RoPE uses adjacent feature pairs, even width; in-place supported. */
cfwx_status cfwx_rope(const cfwx_matrix *x, size_t offset, double base, cfwx_matrix *out);
/* RMS normalization is per row, gamma[features], epsilon>0. In-place allowed. */
cfwx_status cfwx_rms_norm(const cfwx_matrix *x, const double *gamma, double epsilon,
                          cfwx_matrix *out);

/* mask is optional [queries,keys]: true means allowed. Causal additionally
 * allows key<=query and requires equal lengths. Fully masked rows fail with
 * CFWX_SHAPE, never NaN/uniform probabilities. bias is finite additive logits.
 * Probabilities optional [queries,keys]. Outputs disjoint from inputs. */
cfwx_status cfwx_masked_attention(const cfwx_matrix *q, const cfwx_matrix *k, const cfwx_matrix *v,
                                  const bool *mask, const cfwx_matrix *bias, bool causal,
                                  cfwx_matrix *out, cfwx_matrix *probabilities);

typedef struct
{
    size_t width, heads;
    cfwx_dense query, key, value, output;
} cfwx_multihead;
cfwx_status cfwx_multihead_init(cfwx_multihead *layer, size_t width, size_t heads, uint64_t seed);
void cfwx_multihead_free(cfwx_multihead *layer);
/* Self or cross attention: query input [queries,width], context [keys,width]. */
cfwx_status cfwx_multihead_forward(const cfwx_multihead *layer, const cfwx_matrix *query,
                                   const cfwx_matrix *context, const bool *mask, bool causal,
                                   cfwx_matrix *out);

typedef struct
{
    cfwx_dense up, down, gate;
    cfwx_activation activation;
    bool gated;
} cfwx_feedforward;
/* gated=true computes down(activation(gate(x))*up(x)); SiLU gives SwiGLU. */
cfwx_status cfwx_feedforward_init(cfwx_feedforward *layer, size_t width, size_t hidden,
                                  cfwx_activation activation, bool gated, uint64_t seed);
void cfwx_feedforward_free(cfwx_feedforward *layer);
cfwx_status cfwx_feedforward_forward(const cfwx_feedforward *layer, const cfwx_matrix *x,
                                     cfwx_matrix *out);

typedef struct
{
    cfwx_multihead attention;
    cfwx_feedforward feedforward;
    cfwx_vector gamma1, beta1, gamma2, beta2;
    double epsilon;
} cfwx_transformer_block;
/* Pre-LayerNorm block: x+MHA(LN(x)), then residual+FFN(LN(residual)).
 * Deterministic evaluation; compose dropout externally when training.
 * This is an encoder/causal self-attention block, not a full decoder with
 * cross-attention. Training APIs are in models/training.h. */
cfwx_status cfwx_transformer_block_init(cfwx_transformer_block *block, size_t width, size_t heads,
                                        size_t hidden, uint64_t seed);
void cfwx_transformer_block_free(cfwx_transformer_block *block);
cfwx_status cfwx_transformer_block_forward(const cfwx_transformer_block *block,
                                           const cfwx_matrix *x, const bool *mask, bool causal,
                                           cfwx_matrix *out);
/* Row-wise categorical CE, mean over nonignored tokens. SIZE_MAX is a useful
 * ignore label. Smoothing in [0,1); uniform smoothing includes target class.
 * Optional gradient [tokens,vocabulary] includes mean reduction. */
cfwx_status cfwx_token_cross_entropy(const cfwx_matrix *logits, const size_t *labels,
                                     size_t ignore_label, double smoothing, double *loss,
                                     cfwx_matrix *gradient);
#endif
