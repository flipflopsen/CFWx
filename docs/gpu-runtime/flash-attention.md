# FlashAttention implementation plan

FlashAttention is IO-aware tiled exact attention, not sparse approximation. Avoid materializing the full `[B,H,T,T]` probability matrix. For each query tile maintain running max m, normalizer l and output accumulator O. For score tile S: m'=max(m,rowmax(S)); alpha=exp(m-m'); P=exp(S-m'); l'=alpha*l+rowsum(P); O'=alpha*O+PV. Normalize O/l at the end. Keep statistics FP32 for mixed precision.

Causal/window/padding masks must be applied before row max and exp; fully masked rows have an explicit policy, not NaNs. Store log-sum-exp per row for backward. Backward recomputes score/probability tiles: dV=P^T dO, dP=dO V^T, dS=P*(dP-D), D=sum(dO*O), dQ=dS K/sqrt(d), dK=dS^T Q/sqrt(d). Masked entries remain zero. Dropout probability masks must use a replayable counter RNG; store seed/counters, not a dense mask.

Use a maintained library/CUTLASS implementation with compatible SM89/Toolkit support before hand-rolling. Ampere/Ada tiling and synchronization differ from Hopper TMA/WGMMA; do not assume Hopper-only kernels work on RTX4090. Shape/head-dimension tails, shared memory and occupancy must be explicit.

Validation: compare dense masked reference forward/backward, several head sizes, long contexts, short tails, dropout replay, sliding window, all-masked rows, mixed precision and deterministic reduction modes. Profile actual memory traffic and throughput. This document does not claim FlashAttention is implemented in the current reference backend.
