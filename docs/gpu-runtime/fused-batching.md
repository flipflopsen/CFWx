# Opt-in fused mini-batch execution

First increment: one CUDA launch evaluates all independent sequence DAG segments in parallel, and one reverse launch computes all sequence adjoints. Each lane owns a disjoint node range. Shared parameter gradients are reduced with FP64 atomicAdd. A separate small reduction kernel combines per-sequence mean losses. Equal fixed lengths permit weight 1/batch; unequal/ignored-token support requires explicit valid-token weights.

This is fused mini-batch **execution** relative to one serial interpreter for the entire batch. It is not fused GEMM, tiled softmax or FlashAttention. The API selection must be opt-in; retain serial baseline for comparison. Atomic accumulation can change rounding/order and is not bit-deterministic across launches.

Sequence DAGs cannot reference nodes in other segments. Parameter references are not activation-parent references. Segment boundaries and per-sequence RNG counters must be validated before launch. Clear shared gradients once before reverse; each segment seeds its mean-loss adjoint with 1/batch. Do not also backpropagate the final reduction graph, which would double gradients.

Launch policy: sequence lanes in one warp/block is safe only when no divergent __syncthreads is used. Status is shared and set with atomic operations. On failure, reject/poison rather than publish partial optimizer updates. Reuse the serial mask stream by assigning each segment its prefix RNG draw count.

Next increment is an operator-level batched tensor graph with GEMM and fused elementwise epilogues. Remove scalar topology rebuild/upload overhead and cache shapes. Tests run manually: serial/fused loss+all gradients, batch=1, batch>warp width, fixed dropout stream, numerical failures, budget limits, Compute Sanitizer. Do not infer speedup without profiling.
