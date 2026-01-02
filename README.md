# MetalSVD: GPU SVD for PyTorch MPS (Apple Silicon, No CPU Fallback)

**Problem:** PyTorch does **not** support `torch.linalg.svd` / `torch.svd` on the MPS backend. On Apple Silicon, these operations either error or fall back to the CPU, causing severe performance regressions.

**Solution:** **MetalSVD** is a fully GPU-resident Metal implementation of Singular Value Decomposition for PyTorch **MPS** tensors. It provides a fast, drop-in replacement that runs entirely on the Apple GPU with **no CPU fallback**.

MetalSVD supports **any matrix shape**, **any size**, and multiple SVD algorithms optimized for Apple Silicon (M1 / M2 / M3).

---

## Why MetalSVD

- Fixes the missing `torch.linalg.svd` implementation on **PyTorch MPS**
- Runs **entirely on the GPU** (Metal), no hidden CPU hops
- Drop-in replacement via monkeypatching
- Optimized for both **batched small matrices** and **large dense matrices**

---

## Quick Start (Drop-in Replacement)

```python
import torch
import metalsvd

metalsvd.patch_torch()  # overrides torch.linalg.svd for MPS tensors

device = torch.device("mps")
A = torch.randn(64, 128, 128, device=device)

# Now runs fully on the Apple GPU
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
```

CPU and CUDA behavior remain unchanged.

---

## Features

- **Full PyTorch Compatibility**
  - Drop-in replacement for `torch.linalg.svd`
  - Works transparently with existing codebases

- **Universal Shape Support**
  - Tall (`M ≥ N`), Wide (`M < N`), and Square matrices
  - Batched and unbatched tensors

- **No CPU Fallbacks**
  - All orthogonalization and rotations run on Metal
  - Unlike PyTorch MPS, no silent CPU execution

- **Multiple SVD Algorithms**
  - Batched One-Sided Jacobi (small matrices)
  - Randomized SVD (large matrices)
  - Lanczos / Golub–Kahan (high precision)

---

## Performance

MetalSVD avoids the PyTorch MPS CPU fallback path and exploits Apple GPU parallelism.

| Scenario | Matrix Size | Speedup | Notes |
|--------|-------------|----------|-------|
| Batched SVD | 64 × 128 × 128 | **4.23×** | Fused Jacobi kernel |
| Square SVD | 1024 × 1024 | **3.53×** | Block-Jacobi |
| Square SVD | 2048 × 2048 | **16.28×** | GPU parallelism |
| Tall SVD | 4096 × 2048 | **40.52×** | Threadgroup reductions |
| Large rSVD | 4096 × 4096 | **8.60×** | Probabilistic |
| Huge rSVD | 8192 × 8192 | **>10×** | FP16 compute bound |

---

## Installation

### Requirements

- macOS 12.0+ (Apple Silicon)
- PyTorch 2.0+

### Supported Data Types

- `torch.float32` — Recommended
- `torch.float16` — Supported on M1 / M2 / M3
- `torch.bfloat16` — Metal 3.1+ (macOS 14+)

### Build from Source

```bash
git clone https://github.com/myfykris/metalsvd.git
cd metalsvd
pip install .
```

---

## Usage

### Standard Batched SVD

```python
import torch, metalsvd

device = torch.device("mps")
A = torch.randn(64, 128, 128, device=device)

U, S, V = metalsvd.svd(A)
```

---

### Randomized SVD (Large Matrices)

Efficient when only the top-`k` singular values are needed.

```python
M, N, K = 10_000, 10_000, 100
A = torch.randn(M, N, device=device)

U, S, V = metalsvd.randomized_svd(A, k=K, n_iter=2)
```

---

### Lanczos SVD (High Precision)

Deterministic and robust for slowly decaying spectra.

```python
U, S, V = metalsvd.lanczos_svd(A, k=10, n_iter=30)
```

---

## How It Works

### One-Sided Jacobi SVD

MetalSVD implements a parallel **one-sided Jacobi** algorithm using custom Metal kernels. Column rotations and dot products are performed using threadgroup reductions to fully saturate GPU bandwidth.

### Randomized SVD

For large matrices:
1. Sketch the input (`Y = AΩ`)
2. Orthogonalize using MetalSVD (acts as a QR substitute)
3. Project and solve a small SVD

### Lanczos Bidiagonalization

Implements **Golub–Kahan–Lanczos** with full re-orthogonalization for numerical stability in `float32`.

---

## Roadmap

- [ ] Complex number support
- [ ] Prebuilt PyPI wheels

---

## Development Notes

This repository includes a `.agent/` directory with architectural details, benchmarks, and task history. New contributors should start with `.agent/architecture.md`.

---

## License

MIT

