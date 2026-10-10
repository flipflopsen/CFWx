# GPU KV caching design and sliding semantics

Store projected K/V in `[layers,batch,heads,capacity,head_dim]` or a documented equivalent contiguous layout. Cache state includes retained count, next-write index, absolute token position and per-sequence validity. Append Q/K/V on the GPU, attend over valid retained keys, never reproject prior tokens. A ring buffer avoids memmove; attention maps chronological key indices through `(oldest+i)%capacity`.

Fixed mode rejects exhaustion. Sliding mode retains the latest W tokens including the newly appended token. Absolute positions continue increasing; RoPE rotates newly projected keys using absolute position and must not re-rotate retained keys. Sinusoidal embeddings also use absolute positions. Reset sets count/index/position to zero. Prefix prefill may use a batched kernel but must share the same positional/mask conventions as decode.

Eviction changes the computation: cached earlier-layer representations already encode earlier context. Sliding decoding is not equivalent to recomputing the last W raw tokens with positions reset. Compare against an incremental reference with the same eviction policy. Beyond-window generation should be trained with matching attention windows for meaningful quality.

Transactional append: write an uncommitted slot, run all layers, commit metadata only on success. Full-buffer eviction must preserve overwritten committed entries until success (staging slot or backup); otherwise a failed append must poison/reset cache. Model updates invalidate cache. Separate caches per sequence/beam; beam reorder must move metadata and tensor references consistently.

Tests: capacity 1, exact boundary, many wraps, reset/reuse, failed append, absolute-position overflow, variable batch sequence lengths, prefill/decode equality and fixed full-prefix equivalence. GPU cache implementation remains proposed; CPU implementation supplies configurable sliding behavior.
