# CUDA optimizer checkpoint/resume design

Persist FP32/FP64 master weights, first/second Adam moments, one global optimizer step, betas, epsilon, decay/group policy, scheduler state, loss-scaler state, sampler cursor/shuffle state and RNG state/counters. Moment tensors use the same logical names/shapes as master weights. Gradient accumulation state must be saved if checkpointing inside a micro-batch group; otherwise checkpoint only after a completed update and zeroed gradients.

CUDA operations are asynchronous: stop submitting updates, record an event after the last update, wait for it, enqueue parameter/moment downloads on the ordered stream, then synchronize before checksum/fsync. Use bounded pinned buffers; account for staging in host/device budgets. Capture graph-independent state; do not serialize device pointers, CUDA streams/events or compiled graphs. Rebuild workspaces and KV caches after import.

Resume transaction: validate dataset fingerprint/vocabulary and manifest -> construct temporary device model/optimizer -> upload tensors -> restore counters -> wait/error-check -> publish model. Failure must not alter the live model. A poisoned partial-update model cannot produce a valid checkpoint.

For dropout, prefer counter-based Philox indexed by update, example, layer and element. This decouples random masks from launch topology and makes serial/fused partitioning reproducible. State only needs key and counters. This proposal does not add CUDA persistence to the current ABI.

Tests: uninterrupted N updates vs K updates + checkpoint + N-K, empty/nonempty optimizer moments, nonzero decay, scheduler boundary, restored accumulation, injected I/O/CUDA errors. Bit equality is only meaningful with the same backend/kernel/math policy and deterministic reductions.
