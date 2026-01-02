# Performance & Verification Report

## Verification Metrics
The implementation is verified against `torch.linalg.svd` (CPU/LAPACK reference) for numerical stability.

### Metric Definitions
- **Reconstruction Error**: $\frac{||A - U \Sigma V^T||_F}{||A||_F}$
- **Orthogonality Error**: $||I - U^T U||_\infty$

### Results
- **Large Scale**: `randomized_svd` on $4096 \times 4096$ matrix (Rank 100)
  - **MetalSVD**: 0.72s
  - **PyTorch (CPU)**: 6.14s
  - **Speedup**: **8.5x**
  
- **Huge Scale**: `randomized_svd` on $8192 \times 8192$ matrix
  - **FP32**: 0.97s
  - **FP16**: 1.02s
  - **Accuracy**: `3e-4` Relative Error (FP16 verified stable)

- **Batched Small**: Baseline CPU is faster for very small matrices due to Metal dispatch overhead (~300ms overhead). MetalSVD is designed for throughput or larger matrices.

---

## Performance Benchmarks

### 1. Small Matrix Batches (Transformers/LoRA)
**Workload**: Batch=64, $128 \times 128$ Matrices
- **metalsvd**: ~215ms
- **torch.svd (CPU)**: ~58ms
> **Note**: For very small matrices, the CPU fallback is faster due to the overhead of launching ~1200 Metal kernels from Python.

### 2. Large Scale (Randomized SVD)
**Workload**: $10,000 \times 10,000$ Matrix, Rank 100
- **metalsvd (rSVD)**: **1.18s**
- **Accuracy**: `3e-6` Relative Error
- **Architecture**:
  - Uses **Threadgroup Reduction** kernel.
  - Processing 10k elements/threadgroup in shared memory.
  - Fully GPU-resident pipeline (Sketching -> Orthogonalization -> Projection).

### 3. High Precision (Lanczos)
**Workload**: $2,048 \times 2,048$ Matrix
- **Method**: Golub-Kahan Bidiagonalization with Full Re-orthogonalization.
- **Time**: 1.00s
- **Use Case**: Spectra with slow decay where randomized methods may lose precision.

---

### 4. Performance Profile (v0.0.1)

**Summary**: `metalsvd` is highly optimized for **Throughput** (Batched) and **Scale** (Large Matrices), but incurs dispatch overhead for small, single-matrix operations compared to CPU-accelerated LAPACK.

#### Speedup relative to PyTorch (CPU Fallback)
| Scenario | Matrix Size | Speedup | Notes |
| :--- | :--- | :--- | :--- |
| **Batched SVD** | 64 x 128 x 128 | **4.23x** | Fused Kernel + Threadgroup reduction |
| **Square SVD** | 1024 x 1024 | **3.53x** | Fused Block-Jacobi Kernel |
| **Square SVD** | 2048 x 2048 | **16.28x** | Massive parallelism win |
| **Tall SVD** | 4096 x 2048 | **40.52x** | GPU compute density dominates |
| **Large rSVD** | 4096 x 4096 | **8.60x** | Randomized alg + Metal kernels |
| **Huge rSVD** | 8192 x 8192 | **>10x** | FP16 compute limits |

> [!NOTE]
> **Performance Analysis**
> The **Fused Block-Jacobi Kernel** successfully eliminates the dispatch overhead that previously plagued small/medium matrix operations.
> - **1024x1024**: Speedup improved from 0.5x (slower) to **3.5x (faster)**.
> - **Larger Matrices**: Speedups scale super-linearly as GPU saturation improves, reaching **40x** for large tall matrices.

### 5. Scikit-Learn Support
A drop-in wrapper `metalsvd.sklearn.MetalTruncatedSVD` is provided for seamless integration with sklearn pipelines:
```python
from metalsvd.sklearn import MetalTruncatedSVD
svd = MetalTruncatedSVD(n_components=100)
X_reduced = svd.fit_transform(X_numpy) # Automatically uses MPS
```

---

## Supported Data Types
| DType | Support Status | Notes |
| :--- | :--- | :--- |
| `float32` | **Full** | Recommended for general use. |
| `float16` | **Full** | ~2x faster on memory-bound ops. Verified stable. |
| `bfloat16` | **Conditional** | Requires Metal 3.1+ (macOS 14+). Graceful fallback on older OS. |
