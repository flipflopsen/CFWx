/* Analytic reverse-mode training for the existing modular Transformer models. */
#ifndef CFWX_TRAINING_H
#define CFWX_TRAINING_H
#include "models/transformer.h"

/* Gradient owners have exactly the same shapes as their corresponding model.
 * Initialize them with the existing init functions and free with existing free
 * functions. Backward overwrites gradients, does not update model parameters.
 * No aliasing between models, gradients, inputs, upstream and outputs.
 * Forward intermediates are recomputed using current parameters: do not update
 * parameters or change inputs between forward and backward.
 * A bounded local reverse-mode tape analytically differentiates each primitive;
 * no finite differences are used in training. Allocation failures clean up.
 * On failure outputs/gradients may be partial; never apply them. */
cfwx_status cfwx_attention_backward(const cfwx_matrix *q, const cfwx_matrix *k,
    const cfwx_matrix *v, const bool *mask, const cfwx_matrix *bias, bool causal,
    const cfwx_matrix *dy, cfwx_matrix *dq, cfwx_matrix *dk, cfwx_matrix *dv,
    cfwx_matrix *dbias);
cfwx_status cfwx_layer_norm_backward(const cfwx_matrix *x, const double *gamma,
    double epsilon, const cfwx_matrix *dy, cfwx_matrix *dx, double *dgamma, double *dbeta);
cfwx_status cfwx_rms_norm_backward(const cfwx_matrix *x, const double *gamma,
    double epsilon, const cfwx_matrix *dy, cfwx_matrix *dx, double *dgamma);
cfwx_status cfwx_rope_backward(const cfwx_matrix *dy, size_t offset, double base, cfwx_matrix *dx);
cfwx_status cfwx_multihead_backward(const cfwx_multihead *layer,
    const cfwx_matrix *query, const cfwx_matrix *context, const bool *mask,
    bool causal, const cfwx_matrix *dy, cfwx_matrix *dquery,
    cfwx_matrix *dcontext, cfwx_multihead *gradient);
cfwx_status cfwx_feedforward_backward(const cfwx_feedforward *layer,
    const cfwx_matrix *x, const cfwx_matrix *dy, cfwx_matrix *dx, cfwx_feedforward *gradient);
cfwx_status cfwx_transformer_block_backward(const cfwx_transformer_block *block,
    const cfwx_matrix *x, const bool *mask, bool causal, const cfwx_matrix *dy,
    cfwx_matrix *dx, cfwx_transformer_block *gradient);

/* Parameter registry supports deterministic enumeration, clipping and updates.
 * Parameter storage is borrowed. Build matching registries for model/gradient.
 * Trainable leaves include all projection weights/biases and norm gamma/beta. */
#define CFWX_MAX_PARAMETERS 512
typedef struct
{
    size_t count;
    double *data[CFWX_MAX_PARAMETERS];
    size_t size[CFWX_MAX_PARAMETERS];
} cfwx_parameters;
cfwx_status cfwx_parameters_dense(cfwx_parameters *p, cfwx_dense *layer);
cfwx_status cfwx_parameters_block(cfwx_parameters *p, cfwx_transformer_block *block);
cfwx_status cfwx_clip_gradients(cfwx_parameters *gradient, double max_norm, double *norm);

/* AdamW owns first/second moments, initialized against a registry. step is
 * shared across ALL tensors and incremented once per optimizer update.
 * Standard decoupled decay is applied to every registered tensor. Use decay=0
 * or separate optimizer groups to exclude biases/norm parameters. */
typedef struct
{
    size_t count, step;
    size_t size[CFWX_MAX_PARAMETERS];
    double *first[CFWX_MAX_PARAMETERS], *second[CFWX_MAX_PARAMETERS];
} cfwx_adamw;
cfwx_status cfwx_adamw_init(cfwx_adamw *optimizer, const cfwx_parameters *p);
void cfwx_adamw_free(cfwx_adamw *optimizer);
cfwx_status cfwx_adamw_step(cfwx_adamw *optimizer, cfwx_parameters *p,
    const cfwx_parameters *gradient, double rate, double beta1, double beta2,
    double epsilon, double decay);

/* Small trainable causal LM: embeddings + fixed sinusoidal positions + stack
 * of pre-LN blocks + final LayerNorm + untied vocabulary projection.
 * Dropout is disabled in this deterministic reference training path.
 * Max 8 blocks, one sequence per call, caller-owned tokenization/dataset. */
#define CFWX_MAX_BLOCKS 8
typedef struct
{
    size_t vocabulary, width, depth;
    cfwx_matrix embeddings;
    cfwx_transformer_block blocks[CFWX_MAX_BLOCKS];
    cfwx_vector gamma, beta;
    cfwx_dense head;
} cfwx_language_model;
cfwx_status cfwx_language_model_init(cfwx_language_model *model, size_t vocabulary,
    size_t width, size_t heads, size_t hidden, size_t depth, uint64_t seed);
void cfwx_language_model_free(cfwx_language_model *model);
cfwx_status cfwx_parameters_language_model(cfwx_parameters *p, cfwx_language_model *model);
cfwx_status cfwx_language_model_forward(const cfwx_language_model *model,
    const size_t *tokens, size_t length, cfwx_matrix *logits);
/* Mean unsmoothed next-token CE. With gradient!=NULL compute full analytic
 * gradients; otherwise evaluate only. targets has length entries. */
cfwx_status cfwx_language_model_loss(const cfwx_language_model *model,
    const size_t *tokens, const size_t *targets, size_t length, double *loss,
    cfwx_language_model *gradient);
#endif
