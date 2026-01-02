# Project Status and History

## Accomplished (v0.0.2)
- **Core Engine**: Implemented Batched One-Sided Jacobi SVD on Metal.
- **Performance**: Threadgroup optimized kernels (Simd shuffle reduction).
- **Scalability**: Randomized SVD (rSVD) for 10k x 10k matrices.
- **Precision**: Full support for `float32`, `float16`, and `bfloat16`.
- **Robustness**: 
  - Durability guards for older Metal versions.
  - Monkeypatching for seamless `torch.linalg.svd` replacement.
  - Autograd support for training.

## Next Steps / Roadmap
- [ ] **Optimization**: Tune threadgroup sizes dynamically for M3 Max vs M1.
- [ ] **Feature**: Complex number support (Requires $2 \times 2$ unitary updates).
- [ ] **Infra**: Set up CI/CD for binary wheels.
