# cuBLAS/cuBLASLt operator backend

Replace scalar matrix-product DAG nodes with tensor operators. Keep operator-level reverse mode: dense forward `Y=XW+b`; backward `dX=dY W^T`, `dW=X^T dY`, `db=sum_rows(dY)`. Flatten batch/token axes for shared projection weights; use strided-batched GEMM for independent head QK^T and PV products.

cuBLAS historically assumes column-major storage. Either explicitly transpose the interpretation (`Y^T=W^T X^T`) or use cuBLASLt descriptors with documented row-major strides. Leading dimensions and batch strides are element counts. Verify non-square cases; square identity tests cannot detect transposition mistakes.

Own a handle per execution context, attach it to the stream, set pointer/math mode explicitly, and account for workspace. cuBLASLt can fuse bias/activation epilogues but supported epilogues depend on Toolkit/type/layout; retain fallback operators. Do not silently enable TF32 for a strict FP64 reference comparison.

Manual tests: non-square forward/dX/dW, broadcast weights across batches, padded strides, deterministic gradient reduction, status/error propagation. Profile GEMM shapes and tune algorithms under a fixed workspace budget, not a presumed device speed. RTX4090 favors lower precision, so FP64 cuBLAS is correctness support rather than expected peak throughput.
