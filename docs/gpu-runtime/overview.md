# GPU runtime design and implementation map

This directory distinguishes proposed optimized GPU designs from implemented reference features. `uBLAS` is interpreted as NVIDIA **cuBLAS**, not Boost uBLAS (a CPU C++ library).

Assumed target: RTX 4090 SM89, 24 GiB VRAM, Ryzen 9 7900X and 64 GiB RAM; user-reported KMD 617.14 / CUDA UMD 13.4. These are not verified Toolkit/host-compiler compatibility claims. No CUDA build, tests, training or GPU probes are performed by the assistant.

| Topic | Design document | Status |
|---|---|---|
| CPU/CUDA model interchange | [checkpoint-interoperability.md](checkpoint-interoperability.md) | Proposed semantic mapping, not raw memcpy |
| CUDA optimizer persistence | [cuda-resume.md](cuda-resume.md) | Proposed; CPU durable checkpoint implemented |
| GPU KV cache | [gpu-kv-cache.md](gpu-kv-cache.md) | Proposed; CPU fixed/sliding configurable |
| cuBLAS | [cublas.md](cublas.md) | Proposed operator backend |
| Tensor Cores | [tensor-cores.md](tensor-cores.md) | Proposed mixed precision, not FP64 interpreter |
| Fused mini-batch execution | [fused-batching.md](fused-batching.md) | Opt-in sequence-parallel fused CUDA interpreter; manual verification required |
| FlashAttention | [flash-attention.md](flash-attention.md) | Proposed tiled attention, not implemented |
| Durable named checkpoints | [durability.md](durability.md) | POSIX fsync publication and stable name flag |

Read the mathematical contracts and resource limits before implementing optimization. A device-resident scalar DAG is not a cuBLAS/FlashAttention backend. Parallel mini-batch interpreter fusion is not dense-operator fusion or a throughput guarantee.

Manual acceptance gates: compile with the user's Toolkit, run Compute Sanitizer, compare analytic gradients to finite differences, compare serial/fused losses/gradients, exercise checkpoint interruption/corruption, compare cache modes against their matching attention semantics, then profile representative workloads. Never claim performance from theoretical hardware specifications.
