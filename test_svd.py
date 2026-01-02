import torch
import mpssvd
import time

def test_svd_correctness():
    if not torch.backends.mps.is_available():
        print("Skipping test: MPS not available")
        return

    device = torch.device('mps')
    # Small batch for correctness
    B, M, N = 2, 64, 64 
    # M must be >= N for One-Sided Jacobi usually? 
    # If M < N, we usually transpose A first, but let's stick to M >= N first.
    
    torch.manual_seed(42)
    A = torch.randn(B, M, N, device=device)
    
    print(f"Testing SVD on input shape {A.shape}...")
    
    # Run Custom SVD
    start = time.time()
    U, S, V = mpssvd.svd(A)
    torch.mps.synchronize()
    end = time.time()
    print(f"Custom SVD took: {(end - start)*1000:.2f} ms")
    
    # Run Reference SVD
    # torch.linalg.svd returns U, S, Vh (adjoint)
    # Our implementation returns V (not Vh) usually? 
    # Let's check our kernel. 
    # We rotate V. V starts as Identity. 
    # So we return V. 
    # Reconstruction: A = U @ S @ V.T
    
    # Check reconstruction
    # U: (B, M, M) or (B, M, N) ?
    # One-sided Jacobi produces U with same shape as A (if M >= N, it produces M x N 'thin' U).
    # Wait, our code:
    # A_T (N x M). 
    # normalize_kernel writes to U_T (N x M).
    # U = U_T.transpose -> (M, N).
    # So U is (B, M, N). Thin SVD.
    # V is (B, N, N).
    # S is (B, N).
    
    S_diag = torch.zeros(B, N, N, device=device)
    for b in range(B):
        S_diag[b] = torch.diag(S[b])
        
    A_recon = U @ S_diag @ V.transpose(-2, -1)
    
    diff = (A - A_recon).abs().max().item()
    print(f"Reconstruction Max Error: {diff:.6f}")
    
    if diff > 1e-4: # Jacobi SVD can be very precise
        print("FAIL: Reconstruction error too high")
    else:
        print("PASS: Reconstruction OK")
        
    # Check Orthogonality of U
    # U.T @ U should be Identity
    UTU = U.transpose(-2, -1) @ U
    I = torch.eye(N, device=device).unsqueeze(0).expand(B, N, N)
    ort_err = (UTU - I).abs().max().item()
    print(f"U Orthogonality Error: {ort_err:.6f}")
    
    # Check Orthogonality of V
    VTV = V.transpose(-2, -1) @ V
    ort_err_v = (VTV - I).abs().max().item()
    print(f"V Orthogonality Error: {ort_err_v:.6f}")

if __name__ == "__main__":
    test_svd_correctness()
