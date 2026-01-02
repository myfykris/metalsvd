# Release Roadmap

## v0.1.0 Goals (Current Release)
- [x] Functional SVD for `float32`.
- [x] FP16 / BF16 Support.
- [x] Autograd Support.
- [x] Drop-in replacement via `patch_torch()`.

## v0.2.0 Goals (Performance & Features)
### 1. Complex Number Support
- **Why**: Essential for Audio (STFT) and Quantum Simulation.
- **Challenge**: Jacobi rotations become $2 \times 2$ unitary matrices. Requires new kernel logic.

### 2. CI/CD & Binary Wheels
- **Why**: Local compilation (`pip install .`) requires Xcode command line tools. 
- **Plan**: Use GitHub Actions `macos-14` runners to build wheels (`.whl`) and upload to PyPI.

### 3. Dynamic Tuning
- **Why**: `threads_per_group` is currently hardcoded or derived simply.
- **Plan**:Query device registry ID to detect M1/M2/M3 variants and select optimal block sizes.
