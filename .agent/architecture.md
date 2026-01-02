# System Architecture

## Overview
`mpssvd` provides a high-performance SVD implementation for PyTorch on macOS by bypassing the generic MPS fallback and executing custom Metal kernels.

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

### 3. Python Wrapper (`mpssvd/func.py`)
- **Autograd**: Implements a custom `torch.autograd.Function`. The backward pass is computed analytically using the standard SVD gradient formula in pure PyTorch (efficient since $U, S, V$ are already on GPU).
- **Wide Matrices ($M < N$)**: Automatically handles wide inputs by transposing, calling SVD, and swapping $U/V$.

## Design Decisions
- **Embedded Metal**: We store the Metal source as a string in the C++ binary. This simplifies distribution (no separate `.metallib` file management) at the cost of a small compilation hit on first run.
- **Monkeypatching**: We deliberately provide a mechanism to overwrite `torch.linalg.svd` to allow users to upgrade existing codebases with zero refactoring.
