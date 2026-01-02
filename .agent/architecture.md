# System Architecture

## Overview
`metalsvd` provides a high-performance SVD implementation for PyTorch on macOS by bypassing the generic MPS fallback and executing custom Metal kernels.

## Components

### 1. The Host Orchestrator (`src/svd_mps.mm`)
- **Objective-C++**: Bridges PyTorch (C++) and Metal (Obj-C).
- **Dynamic Dispatch**: Checks `tensor.scalar_type()` to select the correct kernel variant (`float`, `half`, `bfloat`).
- **Durability**: Uses runtime checks and preprocessor macros (`#if __METAL_VERSION__ >= 310`) to safely degrade on older macOS versions lacking BFloat16 support.

### 2. The Metal Kernels (`src/svd.metal`)
- **One-Sided Jacobi Algorithm**: Operates on a batch of matrices $A$.
- **Parallelism**:
  - `jacobi_rotate_kernel`: Assigns a threadgroup to each matrix column-pair $(i, j)$.
  - **Threadgroup Reduction**: Uses SIMD shuffle instructions (`simd_sum`, `simd_shuffle_down`) to compute dot products $(A_i \cdot A_j)$ in effectively $O(1)$ time relative to column height (for $M \le 1024$).
  - **Shared Memory**: Caches reduction results to minimize VRAM round-trips.
- **Templating**: Uses C++ templates in MSL to instantiate `half` and `bfloat` variants from a single source of truth.

### 3. Python Wrapper (`metalsvd/func.py`)
- **Autograd**: Implements a custom `torch.autograd.Function`. The backward pass is computed analytically using the standard SVD gradient formula in pure PyTorch (efficient since $U, S, V$ are already on GPU).
- **Wide Matrices ($M < N$)**: Automatically handles wide inputs by transposing, calling SVD, and swapping $U/V$.

## Design Decisions
- **Embedded Metal**: We store the Metal source as a string in the C++ binary. This simplifies distribution (no separate `.metallib` file management) at the cost of a small compilation hit on first run.
- **Monkeypatching**: We deliberately provide a mechanism to overwrite `torch.linalg.svd` to allow users to upgrade existing codebases with zero refactoring.

## Developer Notes & Gotchas

### 1. The "Stale Encoder" Segfault
**Symptom**: `segmentation fault` at subclass `[encoder setComputePipelineState:]`.
**Cause**: Calling complex PyTorch operations (like `tensor.to(device)` or `torch.empty(..., device='mps')`) *after* fetching `stream->commandEncoder()` causes the underlying Metal Command Buffer to be committed or modified by PyTorch's internal pool. This invalidates the retrieved `id<MTLComputeCommandEncoder>` pointer.
**Solution**: ALWAYS perform all tensor allocations, copies (`.to()`), and shape calculations *before* calling `stream->commandEncoder()`. Once you fetch the encoder, strictly perform dispatch operations only.

### 2. Metal Versioning on MPS
**Symptom**: Crash or `nil` PSO (Pipeline State Object) when `options.languageVersion = MTLLanguageVersion3_0` is set.
**Cause**: Not all Apple Silicon devices/OS combinations report strict Metal 3.0 compliance in the way `MTLCompileOptions` expects, even if they support the features.
**Solution**: Rely on the default compiler version (leave `languageVersion` unset) unless strictly necessary. For `bfloat16`, check `__METAL_VERSION__ >= 310` inside the shader code rather than forcing the compiler version host-side.

### 3. Fused Kernel Safety
**Symptom**: Race conditions or incorrect reductions in block-Jacobi.
**Constraint**: The Fused Block-Jacobi kernel (`svd_fused_block_kernel`) enforces `ThreadsPerPair = 1` in the host dispatch logic. While technically inefficient (low occupancy), it is orders of magnitude faster than launching individual kernels for each step ($O(1)$ launch vs $O(N)$ launches). Increasing `ThreadsPerPair` requires careful verification of SIMD-group (wavefront) execution widths, which vary between M1 (32) and other architectures.

