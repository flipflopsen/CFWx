# Tensor Cores and mixed precision

The current double scalar interpreter does not use Tensor Cores. An optimized backend should use FP16/BF16 activations/weights, FP32 accumulation/master weights/moments, and FP32 normalization/softmax statistics. TF32 is an optional FP32 GEMM policy, not a numerically identical replacement for FP64.

Choose dimensions/alignment suitable for SM89 tile kernels; pad tails without changing mask/reduction counts. Store logical shapes separately from packed strides. Use cuBLASLt or CUTLASS before writing custom MMA kernels. Enforce workspace and shared-memory/register budgets.

FP16 training requires dynamic loss scaling: scale loss gradients, detect nonfinite device gradients, unscale before clipping, skip all updates atomically on overflow and adjust scaler. BF16 often needs less scaling but still requires finite checks. Save scaler/scheduler state with optimizer. Do not clip scaled gradients or apply decay on a skipped update.

Acceptance compares outputs/gradients/loss curves to FP64 with operation-specific tolerances, not bit equality. Test saturation, tiny variances, logits extremes, all masks, ignored labels and overflow skip. Benchmark after correctness on the user's hardware; no local GPU tests are performed by the assistant.
