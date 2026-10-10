# Training a small Transformer in CFWx

## Implemented training path
`models/training.h` exposes analytic attention, LayerNorm/RMSNorm, RoPE, projected multi-head attention, gated/ungated FFN and residual-block backward passes. A local reverse-mode tape reconstructs intermediates and accumulates fan-out gradients. Training never uses numerical finite differences; those are a separate correctness test.

The language model connects:

```text
byte token IDs -> learned embeddings + fixed sinusoidal positions
              -> N pre-LayerNorm causal Transformer blocks
              -> final LayerNorm -> untied vocabulary projection -> mean next-token CE
                                      |
analytic reverse pass <---------------+
              -> global gradient clipping -> AdamW update
```

Every embedding, projection weight/bias and normalization gamma/beta is registered and updated. Sequence matrices are `[tokens,features]`. Mini-batches are not fused: the demonstration performs one sequence/update. The reference runner disables dropout. Standalone inverted dropout/backward remains available; if integrating it into the tape, save its exact forward mask rather than regenerating randomness during backward.

## Build and run
Default CMake retains vcpkg at `/opt/vcpkg`. Use the actual checkout's CMake, not the original attached template with placeholder paths. System-toolchain mode:

```sh
cmake -S . -B build -DCFWX_USE_VCPKG=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/Release/cfwx_train
```

The default runner uses a deterministic repetitive `abc` corpus, 200 updates, width 8, 2 heads, FFN width 16 and one block. This checks that full training genuinely reduces loss; it is not evidence of natural-language generalization. The original main application's argument parser/startup is not replaced; `cfwx_train` is a standalone driver with validated arguments. It is built even with `BUILD_TESTING=OFF`.

## Use your own small dataset
Choose a small UTF-8/plain-text corpus you have permission to use. The runner reads raw bytes, not Unicode code points or words. Multi-byte characters are multiple tokens. The file must contain at least 48 bytes and at most 1 MiB. A few KB of repeated/simple text is a useful first experiment.

```sh
mkdir -p data
printf 'the cat sits. the dog runs. the cat runs. the dog sits.\n%.0s' $(seq 1 100) > data/tiny.txt
./build/Release/cfwx_train data/tiny.txt 1000
```

Arguments are `[dataset-file [steps]]`; steps must be 1..100000. Missing file, too-short/oversized data, allocation failures and numerical failures return nonzero. The API additionally accepts a NULL dataset for its built-in regression.

### Dataset construction and leakage
The first 80% of bytes is training; the final 20% is held out. Vocabulary mapping is created from the training partition only and reserves one unknown token. Unseen validation bytes map to UNK. Vocabulary size therefore ranges from 2 through 257; initialization includes the extra UNK class even if all validation bytes are known.

Context length is 8. A training sample takes `tokens = corpus[start:start+8]` and `targets = corpus[start+1:start+9]`, with `start=(step*8)%(train_length-8)`. Targets stay in the training partition. Positions reset for each window. The final report evaluates a fixed first training window and a fixed first validation window, not the entire partitions. For a credible experiment, extend evaluation to every held-out window, weight by valid token count, report UNK coverage, and split by document before windowing. Repeated text across a contiguous split can still make validation trivial. Do not report this toy metric as a benchmark.

### Training update
For each sample:
1. Call `cfwx_language_model_loss(model,tokens,targets,length,&loss,&gradient)`.
2. The function recomputes the graph, uses mean token CE, and exports every gradient leaf.
3. Clip the global gradient norm to 1.0 with `cfwx_clip_gradients`.
4. Call `cfwx_adamw_step` once over the matching parameter/gradient registries.

Defaults: learning rate 0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.0. Bias correction uses one global step per full optimizer update, not per tensor. Model weights and optimizer moments are not changed by backward. Never update parameters between a forward and the corresponding backward. Gradient owners are initialized with the same architecture as model owners; their initial random values are irrelevant because backward overwrites them.

Loss already contains mean reduction: do not divide gradients by context length again. To accumulate several sequences, copy/sum each overwritten gradient into separate accumulation storage, divide by the number of equal-length sequences, then clip/update once. Unequal-length or ignored-label batches require token-weighted reduction. Clip after accumulation, not each micro-batch.

AdamW applies decoupled decay to every registry tensor. The runner uses zero decay. For nonzero decay, create separate groups/optimizer states to exclude bias and norm parameters if desired. Repeated/tied parameter pointers are not supported by the registry; the supplied LM has untied embeddings/head weights. Registries borrow storage and must not outlive models.

## Configure architecture
Edit the init call in `src/experiments/training_runner.c`: vocabulary comes from the dataset, model width must divide evenly by head count, hidden size is positive, and depth is 1..8. Apply identical architecture arguments to the gradient model. The tape capacity is 2048 nodes; very large head counts can exceed it and return allocation failure. Every tape node owns both value and adjoint, so this reference implementation is not memory-efficient. Attention is quadratic in context length.

For longer training, lower the learning rate first if losses diverge; consider warmup and a decay schedule. Neither scheduling nor shuffling is automatic in this runner. Split realistic data before tokenization/windowing and record seed, tokenizer mapping, architecture and hyperparameters. The code has no checkpoint persistence: the model lives only for the runner's process lifetime and is freed on exit. Add explicit parameter and optimizer/vocabulary serialization before claiming restart/resume support.

## Backward APIs and semantics
- Attention exports dQ, dK, dV and optional additive dbias. Masked score derivatives are zero; fully masked rows fail rather than manufacture a distribution.
- Softmax vector-Jacobian product is `p*(dp-dot(p,dp))`.
- LayerNorm removes mean and radial components; RMSNorm removes only the radial component. Both accumulate learned scale gradients across rows; LayerNorm also exports shift gradients.
- Multi-head backward exports separate query and context input derivatives. For self-attention invoked through that API, sum dquery+dcontext externally. Block backward internally shares the normalized input and accumulates all three Q/K/V branches automatically.
- Feed-forward backward supports plain and gated activation paths, including SwiGLU.
- Residual block backward preserves both identity edges and differentiation through both pre-normalizations.
- RoPE backward applies the transpose/inverse pair rotation; positions/base are fixed hyperparameters.

Gradient outputs must be shape-matched preallocated owners, disjoint from model/input/upstream storage. Errors may leave partial gradients: never update after a non-OK status. Numeric checks reject non-finite intermediate adjoints. The deterministic training path uses recomputation, not a persistent cached tape; input data and parameters must be unchanged.

## Validation and limitations
`transformer_gradients` checks attention Q/K/V/bias, LayerNorm/RMSNorm input/scale/shift, RoPE, gated FFN input/parameters and every parameter of a tiny complete LM against centered finite differences. `transformer_training` asserts substantial loss reduction on the built-in corpus. Additional tests cover cross-attention inputs, complete-block input gradients and AdamW behavior. Tests remain active in Release.

```sh
cmake -S . -B build-asan -DCFWX_USE_VCPKG=OFF -DCFWX_SANITIZERS=ON
cmake --build build-asan --parallel
ctest --test-dir build-asan --output-on-failure
```

No GPU/fused kernels, KV cache, serialization/resume, BPE tokenizer, distributed training, mixed precision, or full encoder-decoder stack is supplied. Byte training with a causal stack is genuinely end-to-end trainable but educational. The logger backend remains single-threaded; experiment contexts borrow caller-provided names and sinks.

## Runtime lifecycle update
The previous no-checkpoint, single-window, dropout-disabled and no-KV-cache
limitations are superseded by [RUNTIME.md](RUNTIME.md) and `models/runtime.h`.
Use `cfwx_run` for dropout-enabled mini-batch training, complete held-out evaluation,
checkpoint/resume and incremental cached generation. Earlier descriptions refer to
the original `cfwx_train` runner. CPU micro-batch accumulation is implemented;
fused batching and GPU execution remain explicitly unsupported rather than mislabeled.
