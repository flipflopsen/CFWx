# Training runtime, reproducibility and inference

## Available lifecycle features
`models/runtime.h` adds training dropout, token-weighted mini-batches, full held-out evaluation, portable checkpoint/resume, real per-layer projected K/V caching and temperature/top-k generation. These extend the earlier reference trainer; use `cfwx_run` for this lifecycle. `cfwx_train` remains the original deterministic overfit demonstration.

```sh
cmake -S . -B build -DCFWX_USE_VCPKG=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
mkdir -p data
printf 'the cat sits. the dog runs. the cat runs. the dog sits.\n%.0s' $(seq 1 100) > data/tiny.txt
./build/Release/cfwx_run train data/tiny.txt model.ckpt 300 cpu
./build/Release/cfwx_run resume data/tiny.txt model.ckpt 100 cpu
./build/Release/cfwx_run generate model.ckpt 'the cat ' 80
```

Training/resume steps are **additional** updates, not an absolute stop count. Training defaults: width 8, two heads, hidden 16, one block, context 8, batch 4, residual-branch dropout 0.1, AdamW LR 0.01, zero decay, clip norm 1. Defaults are in the new driver and persisted in the checkpoint. `cpu` is explicit: unsupported backend names fail. Every experiment logs through a caller-created named context. Main startup and original build integration are preserved.

## Dropout and mini-batches
Each block's attention and FFN residual branches use inverted dropout in training. Actual masks/multipliers live on tape nodes and the reverse pass reuses them. Evaluation and generation disable dropout and never consume the training RNG. Attention-probability dropout and embedding dropout are not implicitly enabled.

Batch API accepts pointers to independent sequence arrays and optional per-sequence lengths. Targets of SIZE_MAX are ignored; loss and gradients are weighted by each sequence's **nonignored target count**, not by number of sequences. All-ignored batches fail. It accumulates CPU micro-sequences, clips after accumulation, then updates Adam once. This is real mini-batch semantics but **not fused batched GEMM or fused GPU attention**. There are no cross-example attention edges. The temporary gradient architecture currently assumes homogeneous GELU ungated blocks as constructed by the language-model initializer.

Driver samples starts using a persisted local RNG. The same RNG stream supplies dropout, so resume reproduces sample/mask sequences. Vocabulary is built only from the first 80% partition with an additional UNK token. Dataset is raw bytes, 48 bytes..1 MiB. Dataset fingerprint and mapping are persisted. Resume rejects a changed dataset; fingerprint uses FNV-1a, a reproducibility guard rather than a cryptographic guarantee.

## Full evaluation
Every adjacent next-token target in the validation partition is counted exactly once. Nonoverlapping chunks include the final shorter chunk and losses are weighted by their token counts. Positional indices reset per chunk. The boundary target crossing training/validation partitions is deliberately excluded. Printed `scored` equals validation bytes minus one. This is no longer a selected single-window metric. Repeated text across partitions remains a trivial task; use document-level splits for credible generalization tests.

## Checkpoint format and guarantees
Version 1 stores magic/version, architecture, vocabulary mapping, unknown token, dataset fingerprint, RNG state, context/batch/dropout, optimizer hyperparameters, parameter arrays, Adam first/second moments and global update step. The format uses little-endian uint64 and IEEE-754 binary64 with an FNV checksum. Load bounds architecture/file sizes, validates sizes, finite values and nonnegative second moments, verifies checksum/trailing data, and commits only after successful construction. Unsupported modified architectures (gated FFNs, heterogeneous blocks or nondefault epsilon) are rejected rather than saved incorrectly.

Save writes a same-directory `.tmp` and renames it only on successful output/flush/close. This protects replacement from incomplete writes, **not fsync-based power-loss durability**. Single writer required; concurrent saves to the same path are unsupported. Checkpoint checksum is corruption detection, not malicious-file authentication. Checkpoints are ignored by Git. Gradient buffers and tape/cache are reconstructed after loading. Exact next-update equality is tested on the same platform/backend; different floating-point kernels/hardware can change rounding.

## KV caching and sampling
The cache stores learned projected keys/values for each Transformer layer. Appending a token computes its embedding at its absolute position, new Q/K/V, attention over cached history, residual/FFN and vocabulary logits. Previous keys/values are not recomputed. Tests compare every cached logit row against full-prefix forward for a multi-layer model. Cache length advances only on success; failed slots may be overwritten. Capacity exhaustion is an explicit error; there is no sliding-window eviction. Model updates invalidate existing caches. Reset/new cache is required after training or loading another model.

Generation consumes prompt bytes then autoregressively samples next-token logits with temperature 0.8. The public sampling API supports greedy temperature 0 (no RNG consumption) and deterministic top-k tie handling. No EOS token is assumed by the byte vocabulary; requested token count is the stopping criterion. UNK prints `<UNK>`. The fixed positional representation can evaluate positions beyond training context, but generation quality there is not guaranteed.

## GPU and fusion status — not silently claimed
The implementation is verified CPU double precision. The execution environment provides neither `nvcc` nor `nvidia-smi`; no CUDA hardware validation was possible. GPU execution and fused batched kernels are **not implemented by these CPU lifecycle additions**. A genuine GPU backend must own device buffers, implement projection/normalization/softmax forward and backward, batch strides/masks, optimizer updates, and cross-backend numerical tests. Replacing one CPU matrix multiply with a GPU call would not constitute end-to-end GPU training. This repository rejects unsupported GPU requests instead of mislabeling CPU fallback as GPU execution.

See TRAINING.md for numerical contracts and ARCHITECTURE.md for dependencies. Runtime tests check fixed-mask gradient finite differences, unequal-length token weighting, full evaluation, cache/full-prefix equivalence, corrupted checkpoint rejection and bit-identical next-update resume.
