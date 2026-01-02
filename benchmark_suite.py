import torch
import metalsvd
import time
import math

class BenchmarkRunner:
    def __init__(self):
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        print(f"Running Benchmarks on: {self.device}")
        
        # Explicit Library Load Warmup
        # This ensures all Metal kernels are compiled/loaded before any timing starts.
        if self.device.type == 'mps':
            print("Warming up Metal kernels to exclude laoding time...")
            dummy = torch.randn(1, 16, 16, device=self.device)
            metalsvd.svd(dummy) 
            torch.mps.synchronize()
            print("Warmup complete.\n")
            
        self.results = []

    def run_case(self, name, func, *args, warmups=3, iters=5):
        print(f"Benchmarking: {name}...")
        
        # Warmup
        for _ in range(warmups):
            func(*args)
        torch.mps.synchronize()
        
        # Timing
        start = time.time()
        for _ in range(iters):
            func(*args)
            torch.mps.synchronize()
        end = time.time()
        
        avg_time = (end - start) / iters
        self.results.append((name, avg_time))
        print(f"  -> {avg_time*1000:.2f} ms")
        return avg_time

    def compare(self, name, baseline_func, target_func, *args, iters=5):
        print(f"Comparing: {name}")
        t_base = self.run_case(name + " (Baseline)", baseline_func, *args, iters=iters)
        t_target = self.run_case(name + " (MetalSVD)", target_func, *args, iters=iters)
        
        speedup = t_base / t_target
        print(f"  => Speedup: {speedup:.2f}x\n")
        
        if speedup < 0.9: # Allow small variance
             if "Square" in name and "1024" in name:
                  # We expect this to fail without Fused Kernel. 
                  # With Fused Kernel (TPP=1), it should pass or be close.
                  pass 
             raise RuntimeError(f"Performance Regression detected on {name}: {speedup:.2f}x (Target > 1.0x)")
             
        return speedup

    def print_summary(self):
        print("\n" + "="*40)
        print("BENCHMARK SUMMARY")
        print("="*40)
        print(f"{'Benchmark Name':<30} | {'Time (ms)':<10}")
        print("-" * 45)
        for name, t in self.results:
            print(f"{name:<30} | {t*1000:10.2f}")
        print("="*40)

def benchmark_suite():
    runner = BenchmarkRunner()
    device = runner.device
    
    # 1. Batched Small Matrices (Attention Heads / LoRA)
    # 64 x 128 x 128
    B, M, N = 64, 128, 128
    A_small = torch.randn(B, M, N, device=device)
    
    runner.compare(
        "Batched SVD (64x128x128)",
        lambda: torch.linalg.svd(A_small),
        lambda: metalsvd.svd(A_small),
        iters=50
    )
    
    # 2. Medium Square Matrix
    # 1024 x 1024
    # Full Decomposition
    A_med = torch.randn(1024, 1024, device=device)
    # Baseline on CPU might be slow? MPS usually falls back.
    # We compare single run.
    runner.compare(
        "Square SVD (1024x1024)",
        lambda: torch.linalg.svd(A_med),
        lambda: metalsvd.svd(A_med),
        iters=20
    )
    
    # 3. Large Matrix Randomized SVD
    # 4096 x 4096, Rank 100
    M_large = 4096
    A_large = torch.randn(M_large, M_large, device=device)
    
    def rsvd_run():
        metalsvd.randomized_svd(A_large, k=100, n_iter=2)
        
    def torch_svd_run():
        # Torch doesn't have rSVD easily accessible on MPS usually.
        # We verify full SVD time (truncating later)
        torch.linalg.svd(A_large)
        
    runner.compare(
        "Large rSVD vs Full SVD (4096^2)",
        torch_svd_run,
        rsvd_run,
        iters=5 # Huge, fewer iters
    )
    
    # 4. FP16 vs FP32 Performance
    # Using 10k x 10k rSVD for max load
    M_huge = 8192
    A_fp32 = torch.randn(M_huge, M_huge, device=device, dtype=torch.float32)
    A_fp16 = A_fp32.to(torch.float16)
    
    runner.run_case("Huge rSVD FP32 (8192^2)", lambda: metalsvd.randomized_svd(A_fp32, k=100), iters=3)
    runner.run_case("Huge rSVD FP16 (8192^2)", lambda: metalsvd.randomized_svd(A_fp16, k=100), iters=3)
    
    
    # 6. Exhaustive Permutations (User Request)
    print("\n" + "="*40)
    print("EXHAUSTIVE PERMUTATIONS (Size x Shape)")
    print("="*40)
    
    sizes = [32, 64, 128, 256, 512, 1024, 2048] # 4096 in separate test
    shapes = [
        ("Square", lambda N: (1, N, N)), 
        ("Tall", lambda N: (1, 2*N, N)),
        ("Wide", lambda N: (1, N, 2*N)) # Library should transpose internally or error? Library supports N<=M.
        # metalsvd handles Wide by check or transpose?
        # Standard SVD requires M >= N. If Wide, we just transpose, SVD, then swap U/V.
        # metalsvd.svd currently: N must be even. M >= N.
    ]
    
    for size in sizes:
        for shape_name, shape_fn in shapes:
            B, M, N = shape_fn(size)
            
            # Skip if N > M (Wide) if handled poorly, but let's test it.
            # Usually users want M >= N.
            if N > M:
                # Transpose for native compat test
                pass
                
            name = f"{shape_name} ({M}x{N})"
            
            # Generate input
            try:
                A = torch.randn(B, M, N, device=device)
            except:
                print(f"Skipping {name} (OOM or Error)")
                continue

            # Check logic for Wide support in metalsvd wrapper?
            # If naive implementation crashes on Wide, we skip/note it.
            # torch.linalg.svd handles it.
            
            def run_ours():
                metalsvd.svd(A) # Will error if not implemented for Wide?
                                
            def run_torch():
                torch.linalg.svd(A)
            
            # Fewer iters for larger ones
            n_iters = 50 if M <= 256 else (20 if M <= 1024 else 5)
                
            try:
                runner.compare(name, run_torch, run_ours, iters=n_iters)
            except Exception as e:
                print(f"  [FAIL] {name}: {e}\n")

    runner.print_summary()

if __name__ == "__main__":
    benchmark_suite()
