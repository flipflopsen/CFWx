/* Modular Transformer reference kernels; numerical status propagation and
 * explicit temporary ownership keep experiments independent of the backend. */
#include "models/transformer.h"
#include <string.h>
#include <float.h>

static bool valid(const cfwx_matrix *x)
{
    size_t n;
    return x && x->data && cfwx_count(x->rows, x->cols, &n);
}

static bool same(const cfwx_matrix *a, const cfwx_matrix *b)
{
    return valid(a) && valid(b) && a->rows == b->rows && a->cols == b->cols;
}

static bool activation_valid(cfwx_activation kind)
{
    return kind >= CFWX_ACT_IDENTITY && kind <= CFWX_ACT_SIGMOID;
}

static double activation_value(double x, cfwx_activation kind)
{
    switch (kind)
    {
        case CFWX_ACT_RELU: return cfwx_relu(x);
        case CFWX_ACT_GELU: return cfwx_gelu(x);
        case CFWX_ACT_SILU: return x * cfwx_sigmoid(x);
        case CFWX_ACT_TANH: return tanh(x);
        case CFWX_ACT_SIGMOID: return cfwx_sigmoid(x);
        default: return x;
    }
}

static double activation_derivative(double x, cfwx_activation kind)
{
    switch (kind)
    {
        case CFWX_ACT_RELU: return cfwx_relu_derivative(x);
        case CFWX_ACT_GELU: return cfwx_gelu_derivative(x);
        case CFWX_ACT_SILU:
        {
            double s = cfwx_sigmoid(x);
            return s + x * s * (1.0 - s);
        }
        case CFWX_ACT_TANH: return cfwx_tanh_derivative(x);
        case CFWX_ACT_SIGMOID: return cfwx_sigmoid_derivative(x);
        default: return 1.0;
    }
}

cfwx_status cfwx_activate(const cfwx_matrix *x, cfwx_activation kind, cfwx_matrix *out)
{
    if (!same(x, out) || !activation_valid(kind)) return CFWX_INVALID;
    for (size_t i = 0; i < x->rows * x->cols; i++)
    {
        if (!isfinite(x->data[i])) return CFWX_NUMERIC;
        double y = activation_value(x->data[i], kind);
        if (!isfinite(y)) return CFWX_NUMERIC;
        out->data[i] = y;
    }
    return CFWX_OK;
}

cfwx_status cfwx_activate_backward(const cfwx_matrix *x, const cfwx_matrix *dy,
                                  cfwx_activation kind, cfwx_matrix *dx)
{
    if (!same(x, dy) || !same(x, dx) || !activation_valid(kind)) return CFWX_INVALID;
    for (size_t i = 0; i < x->rows * x->cols; i++)
    {
        if (!isfinite(x->data[i]) || !isfinite(dy->data[i])) return CFWX_NUMERIC;
        double value = dy->data[i] * activation_derivative(x->data[i], kind);
        if (!isfinite(value)) return CFWX_NUMERIC;
        dx->data[i] = value;
    }
    return CFWX_OK;
}

cfwx_status cfwx_residual_add(const cfwx_matrix *a, const cfwx_matrix *b, cfwx_matrix *out)
{
    if (!same(a, b) || !same(a, out)) return CFWX_SHAPE;
    for (size_t i = 0; i < a->rows * a->cols; i++)
    {
        double value = a->data[i] + b->data[i];
        if (!isfinite(value)) return CFWX_NUMERIC;
        out->data[i] = value;
    }
    return CFWX_OK;
}

/* SplitMix64: deterministic local pseudorandomness, not cryptographic. */
static double uniform(uint64_t *seed)
{
    *seed += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t z = *seed;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return (double)((z ^ (z >> 31)) >> 11) * 0x1.0p-53;
}

cfwx_status cfwx_dropout(const cfwx_matrix *x, double p, bool training,
                        uint64_t *seed, cfwx_matrix *out, cfwx_matrix *mask)
{
    if (!same(x, out) || !isfinite(p) || p < 0.0 || p >= 1.0 ||
        (training && !seed) || (mask && (!same(x, mask) || mask->data == x->data || mask->data == out->data)))
        return CFWX_INVALID;
    for (size_t i = 0; i < x->rows * x->cols; i++)
    {
        if (!isfinite(x->data[i])) return CFWX_NUMERIC;
        double multiplier = training && p > 0.0 ? (uniform(seed) < p ? 0.0 : 1.0 / (1.0 - p)) : 1.0;
        double value = x->data[i] * multiplier;
        if (!isfinite(value)) return CFWX_NUMERIC;
        out->data[i] = value;
        if (mask) mask->data[i] = multiplier;
    }
    return CFWX_OK;
}

cfwx_status cfwx_dropout_backward(const cfwx_matrix *dy, const cfwx_matrix *mask, cfwx_matrix *dx)
{
    if (!same(dy, mask) || !same(dy, dx)) return CFWX_SHAPE;
    for (size_t i = 0; i < dy->rows * dy->cols; i++)
    {
        double value = dy->data[i] * mask->data[i];
        if (!isfinite(value)) return CFWX_NUMERIC;
        dx->data[i] = value;
    }
    return CFWX_OK;
}

cfwx_status cfwx_linear(const cfwx_matrix *x, const cfwx_matrix *weights, cfwx_matrix *out)
{
    return cfwx_matrix_multiply(x, weights, out);
}

cfwx_status cfwx_embedding(const cfwx_matrix *table, const size_t *ids, size_t count, cfwx_matrix *out)
{
    if (!valid(table) || !valid(out) || !ids || count == 0 || out->data == table->data) return CFWX_INVALID;
    if (out->rows != count || out->cols != table->cols) return CFWX_SHAPE;
    for (size_t i = 0; i < count; i++) if (ids[i] >= table->rows) return CFWX_SHAPE;
    for (size_t i = 0; i < count; i++)
        for (size_t j = 0; j < table->cols; j++)
        {
            double value = table->data[ids[i] * table->cols + j];
            if (!isfinite(value)) return CFWX_NUMERIC;
            out->data[i * out->cols + j] = value;
        }
    return CFWX_OK;
}

cfwx_status cfwx_embedding_backward(const size_t *ids, size_t count,
                                   const cfwx_matrix *dy, cfwx_matrix *gradient)
{
    if (!ids || !valid(dy) || !valid(gradient) || gradient->data == dy->data) return CFWX_INVALID;
    if (!count || dy->rows != count || dy->cols != gradient->cols) return CFWX_SHAPE;
    for (size_t i = 0; i < count; i++) if (ids[i] >= gradient->rows) return CFWX_SHAPE;
    memset(gradient->data, 0, gradient->rows * gradient->cols * sizeof(double));
    for (size_t i = 0; i < count; i++)
        for (size_t j = 0; j < dy->cols; j++)
        {
            size_t index = ids[i] * dy->cols + j;
            gradient->data[index] += dy->data[i * dy->cols + j];
            if (!isfinite(gradient->data[index])) return CFWX_NUMERIC;
        }
    return CFWX_OK;
}

cfwx_status cfwx_sinusoidal_positions(size_t offset, cfwx_matrix *out)
{
    if (!valid(out) || offset > SIZE_MAX - (out->rows - 1)) return CFWX_INVALID;
    for (size_t i = 0; i < out->rows; i++)
        for (size_t j = 0; j < out->cols; j++)
        {
            double angle = (double)(offset + i) / pow(10000.0, (double)(j - j % 2) / (double)out->cols);
            out->data[i * out->cols + j] = j % 2 ? cos(angle) : sin(angle);
        }
    return CFWX_OK;
}

cfwx_status cfwx_rope(const cfwx_matrix *x, size_t offset, double base, cfwx_matrix *out)
{
    if (!same(x, out) || x->cols % 2 || !isfinite(base) || base <= 1.0 || offset > SIZE_MAX - (x->rows - 1)) return CFWX_INVALID;
    for (size_t i = 0; i < x->rows; i++)
        for (size_t j = 0; j < x->cols; j += 2)
        {
            double angle = (double)(offset + i) / pow(base, (double)j / (double)x->cols);
            double a = x->data[i * x->cols + j], b = x->data[i * x->cols + j + 1];
            double c = cos(angle), s = sin(angle);
            double u = a * c - b * s, v = a * s + b * c;
            if (!isfinite(u) || !isfinite(v)) return CFWX_NUMERIC;
            out->data[i * out->cols + j] = u;
            out->data[i * out->cols + j + 1] = v;
        }
    return CFWX_OK;
}

cfwx_status cfwx_rms_norm(const cfwx_matrix *x, const double *gamma, double epsilon, cfwx_matrix *out)
{
    if (!same(x, out) || !gamma || !isfinite(epsilon) || epsilon <= 0.0 || gamma == out->data) return CFWX_INVALID;
    for (size_t i = 0; i < x->rows; i++)
    {
        /* hypot-based norm avoids overflow in the unscaled sum of squares. */
        double norm = 0.0;
        for (size_t j = 0; j < x->cols; j++)
        {
            if (!isfinite(x->data[i * x->cols + j]) || !isfinite(gamma[j])) return CFWX_NUMERIC;
            norm = hypot(norm, x->data[i * x->cols + j] / sqrt((double)x->cols));
        }
        double denominator = hypot(norm, sqrt(epsilon));
        if (!isfinite(denominator)) return CFWX_NUMERIC;
        for (size_t j = 0; j < x->cols; j++)
        {
            double value = x->data[i * x->cols + j] / denominator * gamma[j];
            if (!isfinite(value)) return CFWX_NUMERIC;
            out->data[i * out->cols + j] = value;
        }
    }
    return CFWX_OK;
}

cfwx_status cfwx_masked_attention(const cfwx_matrix *q, const cfwx_matrix *k,
                                 const cfwx_matrix *v, const bool *mask,
                                 const cfwx_matrix *bias, bool causal,
                                 cfwx_matrix *out, cfwx_matrix *probabilities)
{
    if (!valid(q) || !valid(k) || !valid(v) || !valid(out)) return CFWX_INVALID;
    if (q->cols != k->cols || k->rows != v->rows || out->rows != q->rows || out->cols != v->cols ||
        (causal && q->rows != k->rows) || (bias && (!valid(bias) || bias->rows != q->rows || bias->cols != k->rows)) ||
        (probabilities && (!valid(probabilities) || probabilities->rows != q->rows || probabilities->cols != k->rows))) return CFWX_SHAPE;
    if (out->data == q->data || out->data == k->data || out->data == v->data ||
        (bias && out->data == bias->data) || (probabilities && (probabilities->data == out->data ||
        probabilities->data == q->data || probabilities->data == k->data || probabilities->data == v->data ||
        (bias && probabilities->data == bias->data)))) return CFWX_INVALID;
    double *scores = calloc(k->rows, sizeof(double));
    if (!scores) return CFWX_ALLOC;
    cfwx_status status = CFWX_OK;
    for (size_t i = 0; i < q->rows; i++)
    {
        size_t allowed = 0;
        double max = -DBL_MAX;
        for (size_t j = 0; j < k->rows; j++)
        {
            bool keep = (!mask || mask[i * k->rows + j]) && (!causal || j <= i);
            scores[j] = -INFINITY;
            if (!keep) continue;
            double score = 0.0;
            for (size_t d = 0; d < q->cols; d++) score += q->data[i * q->cols + d] * k->data[j * k->cols + d];
            score /= sqrt((double)q->cols);
            if (bias) score += bias->data[i * k->rows + j];
            if (!isfinite(score)) { status = CFWX_NUMERIC; goto cleanup; }
            scores[j] = score;
            if (score > max) max = score;
            allowed++;
        }
        if (!allowed) { status = CFWX_SHAPE; goto cleanup; }
        double sum = 0.0;
        for (size_t j = 0; j < k->rows; j++) { scores[j] = exp(scores[j] - max); sum += scores[j]; }
        for (size_t j = 0; j < k->rows; j++)
        {
            scores[j] /= sum;
            if (probabilities) probabilities->data[i * k->rows + j] = scores[j];
        }
        for (size_t d = 0; d < v->cols; d++)
        {
            double value = 0.0;
            for (size_t j = 0; j < k->rows; j++)
                if (scores[j] != 0.0) value += scores[j] * v->data[j * v->cols + d];
            if (!isfinite(value)) { status = CFWX_NUMERIC; goto cleanup; }
            out->data[i * out->cols + d] = value;
        }
    }
cleanup:
    free(scores);
    return status;
}

void cfwx_multihead_free(cfwx_multihead *layer)
{
    if (!layer) return;
    cfwx_dense_free(&layer->query); cfwx_dense_free(&layer->key);
    cfwx_dense_free(&layer->value); cfwx_dense_free(&layer->output);
    *layer = (cfwx_multihead){0};
}

cfwx_status cfwx_multihead_init(cfwx_multihead *layer, size_t width, size_t heads, uint64_t seed)
{
    if (!layer || layer->width || layer->heads || layer->query.weights.data || layer->key.weights.data ||
        layer->value.weights.data || layer->output.weights.data || !width || !heads || width % heads) return CFWX_INVALID;
    cfwx_multihead temp = {0};
    cfwx_dense *parts[] = {&temp.query, &temp.key, &temp.value, &temp.output};
    for (size_t i = 0; i < 4; i++)
    {
        cfwx_status s = cfwx_dense_init(parts[i], width, width, seed + i);
        if (s != CFWX_OK) { cfwx_multihead_free(&temp); return s; }
    }
    temp.width = width; temp.heads = heads; *layer = temp;
    return CFWX_OK;
}

cfwx_status cfwx_multihead_forward(const cfwx_multihead *layer, const cfwx_matrix *query,
                                  const cfwx_matrix *context, const bool *mask, bool causal, cfwx_matrix *out)
{
    if (!layer || !same(query, out) || !valid(context) || !layer->heads || !layer->width ||
        layer->width % layer->heads || query->cols != layer->width || context->cols != layer->width) return CFWX_INVALID;
    if (out->data == query->data || out->data == context->data) return CFWX_INVALID;
    cfwx_matrix q = {0}, k = {0}, v = {0}, joined = {0}, hq = {0}, hk = {0}, hv = {0}, ho = {0};
    cfwx_matrix *matrices[] = {&q, &k, &v, &joined, &hq, &hk, &hv, &ho};
    size_t depth = layer->width / layer->heads;
    size_t rows[] = {query->rows, context->rows, context->rows, query->rows, query->rows, context->rows, context->rows, query->rows};
    cfwx_status s = CFWX_OK;
    for (size_t i = 0; i < 8; i++)
    {
        s = cfwx_matrix_init(matrices[i], rows[i], i < 4 ? layer->width : depth);
        if (s != CFWX_OK) goto cleanup;
    }
    s = cfwx_dense_forward(&layer->query, query, &q); if (s != CFWX_OK) goto cleanup;
    s = cfwx_dense_forward(&layer->key, context, &k); if (s != CFWX_OK) goto cleanup;
    s = cfwx_dense_forward(&layer->value, context, &v); if (s != CFWX_OK) goto cleanup;
    for (size_t head = 0; head < layer->heads; head++)
    {
        /* Pack contiguous head matrices; owning matrix API has no strided views. */
        for (size_t i = 0; i < query->rows; i++)
            memcpy(hq.data + i * depth, q.data + i * layer->width + head * depth, depth * sizeof(double));
        for (size_t i = 0; i < context->rows; i++)
        {
            memcpy(hk.data + i * depth, k.data + i * layer->width + head * depth, depth * sizeof(double));
            memcpy(hv.data + i * depth, v.data + i * layer->width + head * depth, depth * sizeof(double));
        }
        s = cfwx_masked_attention(&hq, &hk, &hv, mask, NULL, causal, &ho, NULL);
        if (s != CFWX_OK) goto cleanup;
        for (size_t i = 0; i < query->rows; i++)
            memcpy(joined.data + i * layer->width + head * depth, ho.data + i * depth, depth * sizeof(double));
    }
    s = cfwx_dense_forward(&layer->output, &joined, out);
cleanup:
    for (size_t i = 0; i < 8; i++) cfwx_matrix_free(matrices[i]);
    return s;
}

void cfwx_feedforward_free(cfwx_feedforward *layer)
{
    if (!layer) return;
    cfwx_dense_free(&layer->up); cfwx_dense_free(&layer->down); cfwx_dense_free(&layer->gate);
    *layer = (cfwx_feedforward){0};
}

cfwx_status cfwx_feedforward_init(cfwx_feedforward *layer, size_t width, size_t hidden,
                                 cfwx_activation activation, bool gated, uint64_t seed)
{
    if (!layer || layer->up.weights.data || layer->down.weights.data || layer->gate.weights.data || !activation_valid(activation)) return CFWX_INVALID;
    cfwx_feedforward temp = {0};
    cfwx_status s = cfwx_dense_init(&temp.up, width, hidden, seed);
    if (s == CFWX_OK) s = cfwx_dense_init(&temp.down, hidden, width, seed + 1);
    if (s == CFWX_OK && gated) s = cfwx_dense_init(&temp.gate, width, hidden, seed + 2);
    if (s != CFWX_OK) { cfwx_feedforward_free(&temp); return s; }
    temp.activation = activation; temp.gated = gated; *layer = temp;
    return CFWX_OK;
}

cfwx_status cfwx_feedforward_forward(const cfwx_feedforward *layer, const cfwx_matrix *x, cfwx_matrix *out)
{
    if (!layer || !same(x, out) || out->data == x->data) return CFWX_INVALID;
    cfwx_matrix hidden = {0}, activated = {0}, gate = {0};
    cfwx_status s = cfwx_matrix_init(&hidden, x->rows, layer->up.weights.cols);
    if (s == CFWX_OK) s = cfwx_matrix_init(&activated, x->rows, hidden.cols);
    if (s == CFWX_OK) s = cfwx_dense_forward(&layer->up, x, &hidden);
    if (s != CFWX_OK) goto cleanup;
    if (layer->gated)
    {
        s = cfwx_matrix_init(&gate, x->rows, hidden.cols);
        if (s == CFWX_OK) s = cfwx_dense_forward(&layer->gate, x, &gate);
        if (s == CFWX_OK) s = cfwx_activate(&gate, layer->activation, &activated);
        if (s != CFWX_OK) goto cleanup;
        for (size_t i = 0; i < x->rows * hidden.cols; i++) activated.data[i] *= hidden.data[i];
    }
    else s = cfwx_activate(&hidden, layer->activation, &activated);
    if (s == CFWX_OK) s = cfwx_dense_forward(&layer->down, &activated, out);
cleanup:
    cfwx_matrix_free(&hidden); cfwx_matrix_free(&activated); cfwx_matrix_free(&gate);
    return s;
}

void cfwx_transformer_block_free(cfwx_transformer_block *block)
{
    if (!block) return;
    cfwx_multihead_free(&block->attention); cfwx_feedforward_free(&block->feedforward);
    cfwx_vector_free(&block->gamma1); cfwx_vector_free(&block->beta1);
    cfwx_vector_free(&block->gamma2); cfwx_vector_free(&block->beta2);
    *block = (cfwx_transformer_block){0};
}

cfwx_status cfwx_transformer_block_init(cfwx_transformer_block *block, size_t width,
                                       size_t heads, size_t hidden, uint64_t seed)
{
    if (!block || block->attention.width || block->feedforward.up.weights.data || block->gamma1.data || block->beta1.data || block->gamma2.data || block->beta2.data) return CFWX_INVALID;
    cfwx_transformer_block temp = {0};
    cfwx_status s = cfwx_multihead_init(&temp.attention, width, heads, seed);
    if (s == CFWX_OK) s = cfwx_feedforward_init(&temp.feedforward, width, hidden, CFWX_ACT_GELU, false, seed + 4);
    cfwx_vector *vectors[] = {&temp.gamma1, &temp.beta1, &temp.gamma2, &temp.beta2};
    for (size_t i = 0; i < 4 && s == CFWX_OK; i++) s = cfwx_vector_init(vectors[i], width);
    if (s != CFWX_OK) { cfwx_transformer_block_free(&temp); return s; }
    for (size_t j = 0; j < width; j++) temp.gamma1.data[j] = temp.gamma2.data[j] = 1.0;
    temp.epsilon = 1e-5; *block = temp;
    return CFWX_OK;
}

cfwx_status cfwx_transformer_block_forward(const cfwx_transformer_block *block,
                                          const cfwx_matrix *x, const bool *mask,
                                          bool causal, cfwx_matrix *out)
{
    if (!block || !same(x, out) || out->data == x->data || block->gamma1.size != x->cols ||
        block->beta1.size != x->cols || block->gamma2.size != x->cols || block->beta2.size != x->cols) return CFWX_INVALID;
    cfwx_matrix normalized = {0}, branch = {0}, residual = {0};
    cfwx_status s = cfwx_matrix_init(&normalized, x->rows, x->cols);
    if (s == CFWX_OK) s = cfwx_matrix_init(&branch, x->rows, x->cols);
    if (s == CFWX_OK) s = cfwx_matrix_init(&residual, x->rows, x->cols);
    if (s == CFWX_OK) s = cfwx_layer_norm(x, block->gamma1.data, block->beta1.data, block->epsilon, &normalized);
    if (s == CFWX_OK) s = cfwx_multihead_forward(&block->attention, &normalized, &normalized, mask, causal, &branch);
    if (s == CFWX_OK) s = cfwx_residual_add(x, &branch, &residual);
    if (s == CFWX_OK) s = cfwx_layer_norm(&residual, block->gamma2.data, block->beta2.data, block->epsilon, &normalized);
    if (s == CFWX_OK) s = cfwx_feedforward_forward(&block->feedforward, &normalized, &branch);
    if (s == CFWX_OK) s = cfwx_residual_add(&residual, &branch, out);
    cfwx_matrix_free(&normalized); cfwx_matrix_free(&branch); cfwx_matrix_free(&residual);
    return s;
}

cfwx_status cfwx_token_cross_entropy(const cfwx_matrix *logits, const size_t *labels,
                                    size_t ignore_label, double smoothing,
                                    double *loss, cfwx_matrix *gradient)
{
    if (!valid(logits) || !labels || !loss || !isfinite(smoothing) || smoothing < 0.0 || smoothing >= 1.0 ||
        (gradient && (!same(logits, gradient) || gradient->data == logits->data))) return CFWX_INVALID;
    size_t count = 0;
    for (size_t i = 0; i < logits->rows; i++)
    {
        if (labels[i] == ignore_label) continue;
        if (labels[i] >= logits->cols) return CFWX_SHAPE;
        count++;
    }
    if (!count) return CFWX_INVALID;
    if (gradient) memset(gradient->data, 0, logits->rows * logits->cols * sizeof(double));
    double total = 0.0;
    for (size_t i = 0; i < logits->rows; i++)
    {
        if (labels[i] == ignore_label) continue;
        const double *x = logits->data + i * logits->cols;
        double max = x[0], sum = 0.0;
        for (size_t j = 0; j < logits->cols; j++)
        {
            if (!isfinite(x[j])) return CFWX_NUMERIC;
            if (x[j] > max) max = x[j];
        }
        for (size_t j = 0; j < logits->cols; j++) sum += exp(x[j] - max);
        double logsum = log(sum), uniform_loss = 0.0;
        for (size_t j = 0; j < logits->cols; j++)
        {
            uniform_loss += ((max - x[j]) + logsum) / (double)logits->cols;
            if (gradient)
            {
                double target = smoothing / (double)logits->cols + (j == labels[i] ? 1.0 - smoothing : 0.0);
                gradient->data[i * logits->cols + j] = (exp(x[j] - max) / sum - target) / (double)count;
            }
        }
        total += ((1.0 - smoothing) * ((max - x[labels[i]]) + logsum) + smoothing * uniform_loss) / (double)count;
    }
    if (!isfinite(total)) return CFWX_NUMERIC;
    *loss = total;
    return CFWX_OK;
}
