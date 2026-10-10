/* SM89-compatible CUDA reference reverse-mode engine.
 * Host constructs symbolic topology, device executes every numerical operation.
 * Sequential tape traversal is intentionally a correctness baseline, not a
 * high-throughput backend. AdamW is parallel. No hardware-property probing. */
#include "gpu/cuda_training.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <climits>
#include <limits>
#include <vector>
#include <string>
#include <stdexcept>
#include <new>

enum Op
{
    CONSTANT,
    PARAMETER,
    ADD,
    SUB,
    MUL,
    DIV,
    EXP,
    LOG,
    ROOT,
    GELU,
    DROP,
    LOOKUP,
    MAXIMUM
};
struct Node
{
    int op, a, b;
    double constant;
};
struct Dense
{
    size_t in, out, weight, bias;
};
struct Block
{
    Dense q, k, v, o, up, down;
    size_t g1, b1, g2, b2;
};
struct cfwx_cuda_model
{
    cfwx_cuda_config config{};
    cudaStream_t stream = nullptr;
    double *parameters = nullptr, *gradients = nullptr, *first = nullptr, *second = nullptr;
    Node *nodes = nullptr;
    double *value = nullptr, *adjoint = nullptr;
    double *dropmask = nullptr;
    int *status = nullptr;
    size_t *tokens = nullptr;
    size_t count = 0, capacity = 0, step = 0;
    Dense head{};
    size_t embeddings = 0, final_gamma = 0, final_beta = 0;
    std::vector<Block> blocks;
    std::vector<double> initial;
    std::string error;
    bool poisoned = false;
    uint64_t rng = 0;
};

__device__ static uint64_t next_random(uint64_t &s)
{
    s += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t z = s;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

/* Ordered DAG evaluation. All nodes reside on GPU; no host scalar evaluation. */
__global__ static void forward(const Node *nodes, size_t n, double *v, const double *p,
                               const size_t *tokens, size_t width, size_t vocabulary, bool training,
                               uint64_t rng, double *dropmask, int *status)
{
    if (blockIdx.x || threadIdx.x)
        return;
    *status = 0;
    for (size_t i = 0; i < n; i++)
    {
        Node x = nodes[i];
        bool parents = x.op != CONSTANT && x.op != PARAMETER && x.op != LOOKUP;
        double a = parents && x.a >= 0 ? v[x.a] : 0.0, b = parents && x.b >= 0 ? v[x.b] : 0.0,
               y = 0.0;
        switch (x.op)
        {
        case CONSTANT:
            y = x.constant;
            break;
        case PARAMETER:
            y = p[x.a];
            break;
        case LOOKUP:
            if (tokens[x.a] >= vocabulary)
            {
                *status = 1;
                return;
            }
            y = p[tokens[x.a] * width + (size_t)x.b];
            break;
        case ADD:
            y = a + b;
            break;
        case SUB:
            y = a - b;
            break;
        case MUL:
            y = a * b;
            break;
        case DIV:
            y = a / b;
            break;
        case EXP:
            y = exp(a);
            break;
        case LOG:
            y = log(a);
            break;
        case ROOT:
            y = sqrt(a);
            break;
        case GELU:
            y = 0.5 * a * (1.0 + erf(a * 0.7071067811865475244));
            break;
        case MAXIMUM:
            y = fmax(a, b);
            break;
        case DROP:
            dropmask[i] = training && x.constant > 0.0
                              ? ((double)(next_random(rng) >> 11) * 0x1.0p-53 < x.constant
                                     ? 0.0
                                     : 1.0 / (1.0 - x.constant))
                              : 1.0;
            y = a * dropmask[i];
            break;
        }
        if (!isfinite(y))
        {
            *status = 2;
            return;
        }
        v[i] = y;
    }
}

__global__ static void backward(const Node *nodes, size_t n, const double *v, double *g, double *dp,
                                const size_t *tokens, size_t width, size_t parameter_count,
                                bool training, const double *dropmask, int *status)
{
    if (blockIdx.x || threadIdx.x || *status)
        return;
    for (size_t i = 0; i < n; i++)
        g[i] = 0.0;
    for (size_t i = 0; i < parameter_count; i++)
        dp[i] = 0.0;
    g[n - 1] = 1.0;
    for (size_t i = n; i-- > 0;)
    {
        Node x = nodes[i];
        double u = g[i];
        bool parents = x.op != CONSTANT && x.op != PARAMETER && x.op != LOOKUP;
        double a = parents && x.a >= 0 ? v[x.a] : 0.0, b = parents && x.b >= 0 ? v[x.b] : 0.0;
        switch (x.op)
        {
        case PARAMETER:
            dp[x.a] += u;
            break;
        case LOOKUP:
            dp[tokens[x.a] * width + (size_t)x.b] += u;
            break;
        case ADD:
            g[x.a] += u;
            g[x.b] += u;
            break;
        case SUB:
            g[x.a] += u;
            g[x.b] -= u;
            break;
        case MUL:
            g[x.a] += u * b;
            g[x.b] += u * a;
            break;
        case DIV:
            g[x.a] += u / b;
            g[x.b] -= u * a / (b * b);
            break;
        case EXP:
            g[x.a] += u * v[i];
            break;
        case LOG:
            g[x.a] += u / a;
            break;
        case ROOT:
            g[x.a] += u * 0.5 / v[i];
            break;
        case GELU:
            g[x.a] += u * (0.5 * (1 + erf(a * 0.7071067811865475244)) +
                           a * exp(-0.5 * a * a) * 0.39894228040143267794);
            break;
        case DROP:
            g[x.a] += u * dropmask[i];
            break;
        /* Detached max shift has zero adjoint. Shift contributions cancel
         * exactly in softmax/log-sum-exp, including maximum ties. */
        case MAXIMUM:
            break;
        default:
            break;
        }
        if (!isfinite(u))
        {
            *status = 3;
            return;
        }
    }
    for (size_t i = 0; i < parameter_count; i++)
        if (!isfinite(dp[i]))
        {
            *status = 3;
            return;
        }
}

__global__ static void clip_gradients(double *g, size_t n, double maximum, int *status)
{
    if (blockIdx.x || threadIdx.x || *status)
        return;
    double norm = 0.0;
    for (size_t i = 0; i < n; i++)
        norm = hypot(norm, g[i]);
    if (!isfinite(norm))
    {
        *status = 3;
        return;
    }
    double scale = norm > maximum ? maximum / norm : 1.0;
    for (size_t i = 0; i < n; i++)
        g[i] *= scale;
}
__global__ static void adam(double *p, const double *g, double *m, double *v, size_t n, size_t step,
                            double rate, double decay, int *status)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || *status)
        return;
    double first = 0.9 * m[i] + 0.1 * g[i], second = 0.999 * v[i] + 0.001 * g[i] * g[i];
    double update = first / (1.0 - pow(0.9, (double)step)) /
                    (sqrt(second / (1.0 - pow(0.999, (double)step))) + 1e-8);
    double value = p[i] - rate * (update + decay * p[i]);
    if (!isfinite(value) || !isfinite(first) || !isfinite(second))
    {
        atomicExch(status, 4);
        return;
    }
    p[i] = value;
    m[i] = first;
    v[i] = second;
}

static void cuda_check(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
static size_t product(size_t a, size_t b)
{
    if (!a || !b || a > SIZE_MAX / b)
        throw std::runtime_error("dimension overflow");
    return a * b;
}
static size_t parameter(cfwx_cuda_model &m, size_t n, double value = 0.0)
{
    if (n > m.config.device_budget_bytes / 32 ||
        m.initial.size() > m.config.device_budget_bytes / 32 - n)
        throw std::runtime_error("parameter budget exceeded");
    size_t offset = m.initial.size();
    m.initial.resize(offset + n, value);
    return offset;
}
static uint64_t host_rng(uint64_t &s)
{
    s += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t z = s;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static Dense init_dense(cfwx_cuda_model &m, size_t in, size_t out, uint64_t &seed)
{
    Dense d{in, out, parameter(m, product(in, out)), parameter(m, out)};
    double limit = sqrt(6.0 / (double)(in + out));
    for (size_t i = 0; i < in * out; i++)
        m.initial[d.weight + i] = (2.0 * (double)(host_rng(seed) >> 11) * 0x1.0p-53 - 1.0) * limit;
    return d;
}

struct Graph
{
    cfwx_cuda_model &model;
    std::vector<Node> nodes;
    int add(int op, int a = -1, int b = -1, double value = 0.0)
    {
        if (nodes.size() >= model.config.max_nodes || nodes.size() >= INT_MAX)
            throw std::runtime_error("graph node budget exceeded");
        nodes.push_back(Node{op, a, b, value});
        return (int)nodes.size() - 1;
    }
    int constant(double x)
    {
        return add(CONSTANT, -1, -1, x);
    }
    int p(size_t index)
    {
        if (index > INT_MAX)
            throw std::runtime_error("parameter index too large");
        return add(PARAMETER, (int)index);
    }
    int sum(const std::vector<int> &x)
    {
        int s = constant(0.0);
        for (int v : x)
            s = add(ADD, s, v);
        return s;
    }
    std::vector<int> dense(const std::vector<int> &x, Dense d)
    {
        std::vector<int> y(d.out);
        for (size_t j = 0; j < d.out; j++)
        {
            int s = p(d.bias + j);
            for (size_t i = 0; i < d.in; i++)
                s = add(ADD, s, add(MUL, x[i], p(d.weight + i * d.out + j)));
            y[j] = s;
        }
        return y;
    }
    std::vector<int> norm(const std::vector<int> &x, size_t gamma, size_t beta)
    {
        size_t w = x.size();
        int mean = add(DIV, sum(x), constant((double)w));
        std::vector<int> centered(w), squares(w), y(w);
        for (size_t j = 0; j < w; j++)
        {
            centered[j] = add(SUB, x[j], mean);
            squares[j] = add(MUL, centered[j], centered[j]);
        }
        int inv =
            add(DIV, constant(1.0),
                add(ROOT, add(ADD, add(DIV, sum(squares), constant((double)w)), constant(1e-5))));
        for (size_t j = 0; j < w; j++)
            y[j] = add(ADD, add(MUL, add(MUL, centered[j], inv), p(gamma + j)), p(beta + j));
        return y;
    }
    std::vector<int> residual(const std::vector<int> &x, const std::vector<int> &branch)
    {
        std::vector<int> y(x.size());
        for (size_t j = 0; j < x.size(); j++)
            y[j] = add(ADD, x[j], add(DROP, branch[j], -1, model.config.dropout));
        return y;
    }
    void build(const size_t *targets)
    {
        size_t width = model.config.width, T = model.config.context, heads = model.config.heads,
               depth = width / heads;
        std::vector<int> losses;
        /* A single device tape contains every sequence and one token-mean loss.
         * No CPU per-sequence forward/backward accumulation or numeric fallback. */
        for (size_t batch = 0; batch < model.config.batch; batch++)
        {
            std::vector<std::vector<int>> x(T, std::vector<int>(width));
            for (size_t t = 0; t < T; t++)
                for (size_t j = 0; j < width; j++)
                {
                    double angle = (double)t / pow(10000.0, (double)(j - j % 2) / (double)width);
                    x[t][j] = add(ADD, add(LOOKUP, (int)(batch * T + t), (int)j),
                                  constant(j % 2 ? cos(angle) : sin(angle)));
                }
            for (const Block &block : model.blocks)
            {
                std::vector<std::vector<int>> q(T), k(T), v(T), next(T);
                for (size_t t = 0; t < T; t++)
                {
                    auto ln = norm(x[t], block.g1, block.b1);
                    q[t] = dense(ln, block.q);
                    k[t] = dense(ln, block.k);
                    v[t] = dense(ln, block.v);
                }
                for (size_t t = 0; t < T; t++)
                {
                    std::vector<int> joined(width);
                    for (size_t h = 0; h < heads; h++)
                    {
                        std::vector<int> exponentials(t + 1), scores(t + 1);
                        for (size_t key = 0; key <= t; key++)
                        {
                            std::vector<int> terms(depth);
                            for (size_t d = 0; d < depth; d++)
                                terms[d] = add(MUL, q[t][h * depth + d], k[key][h * depth + d]);
                            scores[key] = add(DIV, sum(terms), constant(sqrt((double)depth)));
                        }
                        /* Max shift is a detached numerical stabilizer computed
                         * on GPU using a dedicated MAX operation added below. */
                        int shift = scores[0];
                        for (size_t key = 1; key <= t; key++)
                            shift = add(MAXIMUM, shift, scores[key]);
                        for (size_t key = 0; key <= t; key++)
                            exponentials[key] = add(EXP, add(SUB, scores[key], shift));
                        int denominator = sum(exponentials);
                        for (size_t d = 0; d < depth; d++)
                        {
                            std::vector<int> terms(t + 1);
                            for (size_t key = 0; key <= t; key++)
                                terms[key] = add(MUL, add(DIV, exponentials[key], denominator),
                                                 v[key][h * depth + d]);
                            joined[h * depth + d] = sum(terms);
                        }
                    }
                    auto res = residual(x[t], dense(joined, block.o));
                    auto hidden = dense(norm(res, block.g2, block.b2), block.up);
                    for (int &n : hidden)
                        n = add(GELU, n);
                    next[t] = residual(res, dense(hidden, block.down));
                }
                x.swap(next);
            }
            for (size_t t = 0; t < T; t++)
            {
                auto logits = dense(norm(x[t], model.final_gamma, model.final_beta), model.head);
                int max = logits[0];
                for (size_t j = 1; j < logits.size(); j++)
                    max = add(MAXIMUM, max, logits[j]);
                std::vector<int> e(logits.size());
                for (size_t j = 0; j < logits.size(); j++)
                    e[j] = add(EXP, add(SUB, logits[j], max));
                int loss =
                    add(ADD, add(SUB, max, logits[targets[batch * T + t]]), add(LOG, sum(e)));
                losses.push_back(loss);
            }
        }
        add(DIV, sum(losses), constant((double)losses.size()));
    }
};

extern "C" const char *cfwx_cuda_error(const cfwx_cuda_model *m)
{
    return m ? m->error.c_str() : "NULL CUDA model";
}
extern "C" size_t cfwx_cuda_parameter_count(const cfwx_cuda_model *m)
{
    return m ? m->count : 0;
}
extern "C" void cfwx_cuda_free(cfwx_cuda_model *m)
{
    if (!m)
        return;
    if (m->stream)
        cudaStreamSynchronize(m->stream);
    cudaFree(m->parameters);
    cudaFree(m->gradients);
    cudaFree(m->first);
    cudaFree(m->second);
    cudaFree(m->nodes);
    cudaFree(m->value);
    cudaFree(m->adjoint);
    cudaFree(m->dropmask);
    cudaFree(m->status);
    cudaFree(m->tokens);
    if (m->stream)
        cudaStreamDestroy(m->stream);
    delete m;
}
extern "C" int cfwx_cuda_create(const cfwx_cuda_config *c, cfwx_cuda_model **out, char *error,
                                size_t capacity)
{
    if (!out)
        return 1;
    *out = nullptr;
    cfwx_cuda_model *m = nullptr;
    try
    {
        if (!c || !c->vocabulary || !c->width || !c->heads || c->width % c->heads || !c->hidden ||
            !c->depth || c->depth > 8 || !c->context || !c->batch || !std::isfinite(c->dropout) ||
            c->dropout < 0 || c->dropout >= 1 || !c->max_nodes || c->max_nodes > INT_MAX ||
            c->device_budget_bytes < 1024)
            throw std::runtime_error("invalid CUDA configuration");
        if (product(c->context, c->batch) > INT_MAX)
            throw std::runtime_error("batch token count exceeds indexing range");
        m = new cfwx_cuda_model;
        m->config = *c;
        m->rng = c->seed;
        uint64_t seed = c->seed;
        size_t embeddings = product(c->vocabulary, c->width);
        m->embeddings = parameter(*m, embeddings);
        double limit = sqrt(6.0 / (double)(c->vocabulary + c->width));
        for (size_t i = 0; i < embeddings; i++)
            m->initial[i] = (2.0 * (double)(host_rng(seed) >> 11) * 0x1.0p-53 - 1.0) * limit;
        for (size_t i = 0; i < c->depth; i++)
        {
            Block b{};
            b.q = init_dense(*m, c->width, c->width, seed);
            b.k = init_dense(*m, c->width, c->width, seed);
            b.v = init_dense(*m, c->width, c->width, seed);
            b.o = init_dense(*m, c->width, c->width, seed);
            b.up = init_dense(*m, c->width, c->hidden, seed);
            b.down = init_dense(*m, c->hidden, c->width, seed);
            b.g1 = parameter(*m, c->width, 1.0);
            b.b1 = parameter(*m, c->width);
            b.g2 = parameter(*m, c->width, 1.0);
            b.b2 = parameter(*m, c->width);
            m->blocks.push_back(b);
        }
        m->final_gamma = parameter(*m, c->width, 1.0);
        m->final_beta = parameter(*m, c->width);
        m->head = init_dense(*m, c->width, c->vocabulary, seed);
        m->count = m->initial.size();
        if (m->count > INT_MAX)
            throw std::runtime_error("parameter count exceeds indexing range");
        cuda_check(cudaStreamCreateWithFlags(&m->stream, cudaStreamNonBlocking));
        cuda_check(cudaMalloc((void **)&m->parameters, m->count * sizeof(double)));
        cuda_check(cudaMalloc((void **)&m->gradients, m->count * sizeof(double)));
        cuda_check(cudaMalloc((void **)&m->first, m->count * sizeof(double)));
        cuda_check(cudaMalloc((void **)&m->second, m->count * sizeof(double)));
        cuda_check(cudaMalloc((void **)&m->tokens, c->batch * c->context * sizeof(size_t)));
        cuda_check(cudaMalloc((void **)&m->status, sizeof(int)));
        cuda_check(cudaMemcpyAsync(m->parameters, m->initial.data(), m->count * sizeof(double),
                                   cudaMemcpyHostToDevice, m->stream));
        cuda_check(cudaMemsetAsync(m->first, 0, m->count * sizeof(double), m->stream));
        cuda_check(cudaMemsetAsync(m->second, 0, m->count * sizeof(double), m->stream));
        cuda_check(cudaMemsetAsync(m->gradients, 0, m->count * sizeof(double), m->stream));
        cuda_check(cudaStreamSynchronize(m->stream));
        m->initial.clear();
        *out = m;
        return 0;
    }
    catch (const std::exception &e)
    {
        if (error && capacity)
            std::snprintf(error, capacity, "%s", e.what());
        cfwx_cuda_free(m);
        return 1;
    }
}
extern "C" int cfwx_cuda_run(cfwx_cuda_model *m, const size_t *ids, const size_t *targets,
                             int training, double rate, double maximum, double decay, double *loss)
{
    if (!m)
        return 1;
    try
    {
        if (m->poisoned || !ids || !targets || !loss || !std::isfinite(rate) || rate < 0 ||
            !std::isfinite(maximum) || maximum <= 0 || !std::isfinite(decay) || decay < 0)
            throw std::runtime_error("invalid run arguments or poisoned model");
        size_t tokens = m->config.context * m->config.batch;
        for (size_t i = 0; i < tokens; i++)
            if (ids[i] >= m->config.vocabulary || targets[i] >= m->config.vocabulary)
                throw std::runtime_error("token ID out of range");
        Graph graph{*m, {}};
        graph.build(targets);
        size_t n = graph.nodes.size();
        size_t bytes = product(n, sizeof(Node) + 3 * sizeof(double));
        size_t fixed = m->count * 32 + tokens * sizeof(size_t) + sizeof(int);
        if (fixed > m->config.device_budget_bytes || bytes > m->config.device_budget_bytes - fixed)
            throw std::runtime_error("device tape budget exceeded");
        if (n > m->capacity)
        {
            cuda_check(cudaStreamSynchronize(m->stream));
            cudaFree(m->nodes);
            cudaFree(m->value);
            cudaFree(m->adjoint);
            cudaFree(m->dropmask);
            m->nodes = nullptr;
            m->value = nullptr;
            m->adjoint = nullptr;
            m->dropmask = nullptr;
            m->capacity = 0;
            cuda_check(cudaMalloc((void **)&m->nodes, n * sizeof(Node)));
            cuda_check(cudaMalloc((void **)&m->value, n * sizeof(double)));
            cuda_check(cudaMalloc((void **)&m->adjoint, n * sizeof(double)));
            cuda_check(cudaMalloc((void **)&m->dropmask, n * sizeof(double)));
            m->capacity = n;
        }
        cuda_check(cudaMemcpyAsync(m->nodes, graph.nodes.data(), n * sizeof(Node),
                                   cudaMemcpyHostToDevice, m->stream));
        cuda_check(cudaMemcpyAsync(m->tokens, ids, tokens * sizeof(size_t), cudaMemcpyHostToDevice,
                                   m->stream));
        forward<<<1, 1, 0, m->stream>>>(m->nodes, n, m->value, m->parameters, m->tokens,
                                        m->config.width, m->config.vocabulary, training != 0,
                                        m->rng, m->dropmask, m->status);
        cuda_check(cudaGetLastError());
        if (training)
        {
            backward<<<1, 1, 0, m->stream>>>(m->nodes, n, m->value, m->adjoint, m->gradients,
                                             m->tokens, m->config.width, m->count, true,
                                             m->dropmask, m->status);
            cuda_check(cudaGetLastError());
            if (rate > 0)
            {
                clip_gradients<<<1, 1, 0, m->stream>>>(m->gradients, m->count, maximum, m->status);
                adam<<<(unsigned)((m->count + 255) / 256), 256, 0, m->stream>>>(
                    m->parameters, m->gradients, m->first, m->second, m->count, m->step + 1, rate,
                    decay, m->status);
                cuda_check(cudaGetLastError());
            }
        }
        int status = 0;
        cuda_check(cudaMemcpyAsync(loss, m->value + n - 1, sizeof(double), cudaMemcpyDeviceToHost,
                                   m->stream));
        cuda_check(
            cudaMemcpyAsync(&status, m->status, sizeof(int), cudaMemcpyDeviceToHost, m->stream));
        cuda_check(cudaStreamSynchronize(m->stream));
        if (status)
        {
            m->poisoned = true;
            throw std::runtime_error("CUDA numerical status " + std::to_string(status));
        }
        if (training)
        {
            size_t draws = 0;
            for (const Node &node : graph.nodes)
                if (node.op == DROP && node.constant > 0.0)
                    draws++;
            m->rng += UINT64_C(0x9e3779b97f4a7c15) * (uint64_t)draws;
            if (rate > 0)
                m->step++;
        }
        m->error.clear();
        return 0;
    }
    catch (const std::exception &e)
    {
        m->error = e.what();
        return 1;
    }
}
static int copy(cfwx_cuda_model *m, double *host, const double *device, size_t count)
{
    if (!m || !host || count != m->count || m->poisoned)
        return 1;
    try
    {
        cuda_check(cudaMemcpy(host, device, count * sizeof(double), cudaMemcpyDeviceToHost));
        return 0;
    }
    catch (const std::exception &e)
    {
        m->error = e.what();
        return 1;
    }
}
extern "C" int cfwx_cuda_get_parameters(cfwx_cuda_model *m, double *out, size_t n)
{
    return copy(m, out, m ? m->parameters : nullptr, n);
}
extern "C" int cfwx_cuda_get_gradients(cfwx_cuda_model *m, double *out, size_t n)
{
    return copy(m, out, m ? m->gradients : nullptr, n);
}
extern "C" int cfwx_cuda_set_parameters(cfwx_cuda_model *m, const double *values, size_t n)
{
    if (!m || !values || n != m->count || m->poisoned)
        return 1;
    for (size_t i = 0; i < n; i++)
        if (!std::isfinite(values[i]))
            return 1;
    try
    {
        cuda_check(cudaMemcpy(m->parameters, values, n * sizeof(double), cudaMemcpyHostToDevice));
        return 0;
    }
    catch (const std::exception &e)
    {
        m->error = e.what();
        return 1;
    }
}
