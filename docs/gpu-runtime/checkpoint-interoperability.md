# CPU checkpoint interoperability: semantic, not positional

CPU and CUDA parameter registries currently differ in ordering and representation. Define a backend-neutral manifest: schema version, architecture, tensor name, logical shape, layout, dtype and byte length. Stable names include `embedding.weight`, `blocks.i.attention.query.weight`, `blocks.i.norm1.gamma`, `head.bias`. Build adapters from names to CPU arrays and CUDA offsets. Never copy the flattened CUDA array into the CPU registry by index.

Use canonical row-major logical `[in,out]` weights, `[out]` bias and `[vocabulary,width]` embedding. Transpose/layout-convert on import to a future column-major or packed device representation. Padding/quantization scales need explicit metadata; store master values, not hardware-packed artifacts. Download asynchronously into pinned host staging on the owning stream, then synchronize an event before writing. Upload only after checksum, dtype/shape and architecture validation; commit a newly constructed device model only on success.

Preserve positional policy, norm epsilon, FFN activation/gating, masks, tied-parameter aliases, dropout semantics and vocabulary/UNK. The current CPU v1 format assumes homogeneous GELU blocks and epsilon 1e-5; interoperability needs a new version or explicit limited adapter, not silent reinterpretation. Version migration must reject unsupported architecture.

Acceptance: identical logical tensor round-trip, CPU/CUDA forward and backward tolerance tests, malformed dimensions/duplicates/aliases, truncation/checksum tests, and endian/dtype conversion tests. Exact training continuation across different kernels is generally not bit-identical; claim semantic equivalence with stated tolerance instead.
