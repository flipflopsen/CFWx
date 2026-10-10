# Composable Transformer reference blocks

Public entry point: `include/models/transformer.h`. These are portable CPU double-precision educational kernels, not an optimized or fully trainable framework.

## Building blocks
- Activation dispatch: identity, ReLU, exact GELU, SiLU, tanh and sigmoid, with preactivation backward.
- Existing stable softmax, MSE, BCE-with-logits and categorical CE; token-wise mean CE adds ignore labels, uniform label smoothing and gradients.
- Bias-free linear projection, existing owning affine dense layers with backward and SGD.
- Token embedding lookup and scatter-add backward for repeated IDs.
- Sinusoidal positions and adjacent-pair RoPE (even width).
- Existing LayerNorm plus RMSNorm; residual addition; deterministic seeded inverted dropout and mask-based backward.
- Masked scaled dot-product heads, additive finite bias and optional returned probabilities.
- Learned Q/K/V/output projections with multi-head packing/concatenation for self- or cross-attention.
- Configurable FFN `down(activation(up(x)))` and gated FFN `down(activation(gate(x))*up(x))`; SiLU gives SwiGLU.
- Pre-LayerNorm block: attention residual, then feed-forward residual. Set causal=true for autoregressive self-attention. Stack blocks by passing each output into the next block.

## Shapes and ownership
Matrices are `[tokens,features]`; callers loop over batches. Linear/dense weights are `[input,output]`. Embeddings `[vocabulary,width]`. Multi-head width must divide evenly by head count. Head scaling uses `sqrt(width/heads)`, not sqrt(model width). Mask `[queries,keys]` uses true=allowed; causal mode additionally requires key<=query and equal lengths. A completely masked query returns CFWX_SHAPE. Padding query rows should be omitted or handled outside the attention call rather than expecting a fabricated uniform distribution.

Zero-initialize owners, call matching free, never shallow-copy them. Allocate outputs first. Exact in-place operation is allowed for residuals, RoPE, RMSNorm and dropout; all other buffers must be disjoint, including partial overlap. Activation operations support exact in-place as well. Optional dropout mask must not alias input/output. Kernels return cfwx_status; errors may partially modify output. Composite constructors commit only on success and free all temporary allocations on failure. Forward calls use temporary owning matrices; this is clear reference code, not a low-allocation execution engine.

RNG state is caller-owned SplitMix64, noncryptographic. Evaluation dropout consumes no RNG. Loss reduction is mean over nonignored rows; all-ignored input is invalid. Label smoothing distributes smoothing/vocabulary mass to every class including the target. The last LM prediction is scored against an explicit next-token target supplied by the runner.

## Examples
`transformer_language_runner(logger)` connects token lookup -> sinusoidal positional residual -> causal pre-LN Transformer block -> dense LM head -> smoothed token CE.

`transformer_cross_attention_runner(logger)` connects projected cross-attention with context padding mask -> query residual -> RMSNorm -> SwiGLU. These examples use random weights and emit a warning where predictions could otherwise be mistaken for a benchmark. Every runner borrows a named logger and propagates statuses after cleanup.

```sh
cmake -S . -B build -DCFWX_USE_VCPKG=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/Debug/cfwx_transformer_examples
```

Default vcpkg integration, cmocka and custom targets are retained. The example executable is created when BUILD_TESTING=ON; the runner functions are also linked into the main application for later selection. Main startup remains unchanged.

## Validation and boundaries
Tests include activation finite differences, repeated embedding IDs, smoothed CE finite differences/ignored rows, dropout masks, RoPE norm preservation, RMSNorm reference, fully masked rows, identity-projection multi-head comparison, zero-branch block identity, and causal invariance when a future token changes.

Backward exists for activations, embeddings, losses, dropout and the original dense layer. Attention, LayerNorm/RMSNorm, RoPE, FFN composites and Transformer blocks do NOT yet expose complete backward passes. Therefore the causal LM example is forward/loss evaluation, not end-to-end training. No KV cache, optimizer beyond SGD, tokenizer, serialization, GPU, fused kernels, batched 3-D attention, or full encoder-decoder cross-attention block. RoPE is an explicit Q/K operation, not silently applied inside projected multi-head attention. Dropout is an explicit block, not automatically enabled inside Transformer evaluation. These boundaries keep the APIs modular and results interpretable.
