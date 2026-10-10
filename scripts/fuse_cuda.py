"""Static CUDA source transformation only. Does NOT compile/probe/execute GPU code."""
from pathlib import Path
p=Path('src/gpu/cuda_training.cu');s=p.read_text()
if 'struct Segment' not in s:
 s=s.replace('struct Dense\n', 'struct Segment { size_t begin, end; uint64_t rng_offset; };\n\nstruct Dense\n',1)
 s=s.replace('    cudaStream_t stream = nullptr;', '    cudaStream_t stream = nullptr;\n    bool fused = false;\n    Segment *segments = nullptr;')
 s=s.replace('    std::vector<Node> nodes;', '    std::vector<Node> nodes;\n    std::vector<Segment> segments;\n    uint64_t rng_offset = 0;')
 marker='''        for (size_t batch = 0; batch < model.config.batch; batch++)
        {'''
 if marker not in s:raise RuntimeError('batch graph pattern missing')
 s=s.replace(marker,marker+'\n            size_t begin = nodes.size();\n            size_t loss_begin = losses.size();')
 marker='''        add(DIV, sum(losses), constant((double)losses.size()));'''
 index=s.index(marker)
 # Insert sequence mean node immediately before closing batch loop.
 closing=s.rfind('        }',0,index)
 s=s[:closing]+'''            std::vector<int> sample_losses(losses.begin() + loss_begin, losses.end());
            add(DIV, sum(sample_losses), constant((double)sample_losses.size()));
            segments.push_back(Segment{begin, nodes.size(), rng_offset});
            for (size_t i = begin; i < nodes.size(); i++)
                if (nodes[i].op == DROP && nodes[i].constant > 0.0) rng_offset++;
'''+s[closing:]
 # Generate batched kernels from the existing operation implementation, not a second math copy.
 start=s.index('__global__ static void forward(');end=s.index('__global__ static void backward(',start)
 f=s[start:end]
 f=f.replace('static void forward(', 'static void fused_forward(').replace('size_t n, double *v,', 'size_t n, const Segment *segments, size_t batch, double *v,')
 f=f.replace('''    if (blockIdx.x || threadIdx.x)
        return;
    *status = 0;''','''    size_t sample = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch) return;
    Segment segment = segments[sample];
    rng += UINT64_C(0x9e3779b97f4a7c15) * segment.rng_offset;''')
 f=f.replace('for (size_t i = 0; i < n; i++)','for (size_t i = segment.begin; i < segment.end; i++)')
 f=f.replace('*status = 1;', 'atomicExch(status, 1);').replace('*status = 2;', 'atomicExch(status, 2);')
 startb=end;endb=s.index('__global__ static void clip_gradients(',startb)
 b=s[startb:endb]
 b=b.replace('static void backward(', 'static void fused_backward(').replace('size_t n, const double *v,', 'size_t n, const Segment *segments, size_t batch, const double *v,')
 old='''    if (blockIdx.x || threadIdx.x || *status)
        return;
    for (size_t i = 0; i < n; i++)
        g[i] = 0.0;
    for (size_t i = 0; i < parameter_count; i++)
        dp[i] = 0.0;
    g[n - 1] = 1.0;
    for (size_t i = n; i-- > 0;)'''
 new='''    size_t sample = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch || *status) return;
    Segment segment = segments[sample];
    for (size_t i = segment.begin; i < segment.end; i++) g[i] = 0.0;
    g[segment.end - 1] = 1.0 / (double)batch;
    for (size_t i = segment.end; i-- > segment.begin;)'''
 if old not in b:raise RuntimeError('backward kernel pattern missing')
 b=b.replace(old,new)
 b=b.replace('dp[x.a] += u;', 'atomicAdd(&dp[x.a], u);').replace('dp[tokens[x.a] * width + (size_t)x.b] += u;', 'atomicAdd(&dp[tokens[x.a] * width + (size_t)x.b], u);')
 b=b.replace('*status = 3;', 'atomicExch(status, 3);')
 check=b.find('    for (size_t i = 0; i < parameter_count; i++)')
 if check<0:raise RuntimeError('gradient validation pattern missing')
 b=b[:check]+'}\n\n'
 extra='''/* Loss reduction is separate from sequence graphs: reverse does not traverse
 * the final serial reduction, avoiding duplicate gradient contributions. */
__global__ static void fused_loss(double *values, size_t n, const Segment *segments, size_t batch)
{
    if (blockIdx.x || threadIdx.x) return;
    double sum = 0.0;
    for (size_t i = 0; i < batch; i++) sum += values[segments[i].end - 1];
    values[n - 1] = sum / (double)batch;
}

'''
 s=s[:endb]+f+b+extra+s[endb:]
 s=s.replace('    cudaFree(m->parameters);','    cudaFree(m->segments);\n    cudaFree(m->parameters);',1)
 s=s.replace('        cuda_check(cudaMalloc((void **)&m->tokens,', '        cuda_check(cudaMalloc((void **)&m->segments, c->batch * sizeof(Segment)));\n        cuda_check(cudaMalloc((void **)&m->tokens,',1)
 start=s.index('        forward<<<1, 1, 0, m->stream>>>');end=s.index('        int status = 0;',start)
 s=s[:start]+'''        cuda_check(cudaMemsetAsync(m->status, 0, sizeof(int), m->stream));
        if (m->fused)
        {
            cuda_check(cudaMemcpyAsync(m->segments, graph.segments.data(),
                graph.segments.size() * sizeof(Segment), cudaMemcpyHostToDevice, m->stream));
            fused_forward<<<(unsigned)((m->config.batch + 31) / 32), 32, 0, m->stream>>>(
                m->nodes, n, m->segments, m->config.batch, m->value, m->parameters, m->tokens,
                m->config.width, m->config.vocabulary, training != 0, m->rng, m->dropmask, m->status);
            fused_loss<<<1, 1, 0, m->stream>>>(m->value, n, m->segments, m->config.batch);
        }
        else forward<<<1, 1, 0, m->stream>>>(m->nodes, n, m->value, m->parameters, m->tokens,
            m->config.width, m->config.vocabulary, training != 0, m->rng, m->dropmask, m->status);
        cuda_check(cudaGetLastError());
        if (training)
        {
            if (m->fused)
            {
                cuda_check(cudaMemsetAsync(m->gradients, 0, m->count * sizeof(double), m->stream));
                fused_backward<<<(unsigned)((m->config.batch + 31) / 32), 32, 0, m->stream>>>(
                    m->nodes, n, m->segments, m->config.batch, m->value, m->adjoint,
                    m->gradients, m->tokens, m->config.width, m->count, true, m->dropmask, m->status);
            }
            else backward<<<1, 1, 0, m->stream>>>(m->nodes, n, m->value, m->adjoint, m->gradients,
                m->tokens, m->config.width, m->count, true, m->dropmask, m->status);
            cuda_check(cudaGetLastError());
            if (rate > 0)
            {
                clip_gradients<<<1, 1, 0, m->stream>>>(m->gradients, m->count, maximum, m->status);
                adam<<<(unsigned)((m->count + 255) / 256), 256, 0, m->stream>>>(m->parameters,
                    m->gradients, m->first, m->second, m->count, m->step + 1, rate, decay, m->status);
                cuda_check(cudaGetLastError());
            }
        }
'''+s[end:]
 s+='''
extern "C" int cfwx_cuda_set_fused_batching(cfwx_cuda_model *model, int enabled)
{
    if (!model || model->poisoned) return 1;
    model->fused = enabled != 0;
    return 0;
}
'''
 p.write_text(s)
p=Path('include/gpu/cuda_training.h');s=p.read_text()
if 'cfwx_cuda_set_fused_batching' not in s:
 s=s.replace('void cfwx_cuda_free(', '''/* Opt-in launch-fused sequence parallelism; serial interpreter is default.
 * FP64 atomics make gradient reduction order nondeterministic. Not fused GEMM.
 * Set before a run; model calls remain single-threaded. */
int cfwx_cuda_set_fused_batching(cfwx_cuda_model *model, int enabled);
void cfwx_cuda_free(''')
 p.write_text(s)
import subprocess
subprocess.run(['clang-format','-i','src/gpu/cuda_training.cu','include/gpu/cuda_training.h'],check=True)
