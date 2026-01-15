# CUDA Matrix Multiplication Optimization

This project implements and optimizes a CUDA-based matrix multiplication kernel.
Starting from a naive global-memory implementation, the kernel is optimized using shared-memory tiling and profiled using NVIDIA Nsight Compute to analyze memory access patterns, occupancy, and instruction throughput.

-------------------------------------------------------------------------------

OVERVIEW

Matrix multiplication (C = A × B) is a fundamental GPU workload that is often limited
by memory access efficiency. This project demonstrates how shared-memory tiling can
significantly reduce redundant global memory accesses and improve overall kernel
performance.

Two CUDA kernels are implemented:
- Naive kernel: Each thread loads operands directly from global memory.
- Tiled kernel: Tiles of input matrices are loaded into shared memory and reused
  across threads in a block.

-------------------------------------------------------------------------------

IMPLEMENTATION DETAILS

Language: CUDA C++
Block size: 16 × 16
Tile size: 16
Data type: float
Hardware: NVIDIA GeForce RTX 3050 (WDDM)

Correctness is verified by comparing the outputs of the naive and tiled kernels.

-------------------------------------------------------------------------------

PERFORMANCE RESULTS

Matrix Size    Naive GFLOP/s    Tiled GFLOP/s    Speedup
-----------    --------------   --------------   --------
1024³          577.7            770.4            1.33×
2048³          ~571             ~770             1.35×
4096³          ~570             ~766             1.35×

- Maximum absolute difference between kernels: 0.0
- The tiled kernel consistently outperforms the naive implementation across all
  tested sizes.

-------------------------------------------------------------------------------

PROFILING WITH NSIGHT COMPUTE

The kernels were profiled using NVIDIA Nsight Compute with a single kernel launch
and full metric collection.

Key Observations:

1) Global Memory Access Patterns
- Nsight Compute reported low global load efficiency in the naive kernel.
- On average, only ~18 of 32 bytes per memory sector were utilized.
- This indicates inefficient or partially uncoalesced global memory accesses.
- The tiled kernel reduces global memory traffic by reusing data from shared memory.

2) Memory Stall Analysis
- Warps spent a significant number of cycles stalled on local/global (LG) memory
  instruction queues.
- Approximately 54% of the average instruction issue gap was attributed to
  memory-related stalls.
- This behavior is consistent with frequent global memory operations in the naive
  implementation.

3) Optimization Effect
- Shared-memory tiling reduced redundant global memory loads.
- Improved data reuse lowered memory instruction pressure.
- Resulted in higher effective throughput and consistent speedups.

-------------------------------------------------------------------------------

BUILD AND RUN

Compile:
nvcc -O3 -lineinfo -allow-unsupported-compiler -o matmul matmul.cu

Run:
./matmul          (defaults to 1024)
./matmul 2048
./matmul 4096

-------------------------------------------------------------------------------

NSIGHT COMPUTE COMMANDS

Profile a single kernel launch (full metrics):
ncu --kernel-name matmul_naive --launch-count 1 --set full matmul 1024
ncu --kernel-name matmul_tiled --launch-count 1 --set full matmul 1024

Targeted metric sets (faster):
ncu --set memory_workload_analysis matmul 1024
ncu --set occupancy matmul 1024
ncu --set speed_of_light matmul 1024

-------------------------------------------------------------------------------

SUMMARY

This project demonstrates how shared-memory tiling improves GPU kernel performance
by reducing inefficient global memory accesses. Profiling with Nsight Compute
highlights the impact of memory access patterns and instruction stalls on overall
performance, and validates the effectiveness of the optimization.

-------------------------------------------------------------------------------

FUTURE IMPROVEMENTS (OPTIONAL)

- Vectorized global loads (e.g., float4)
- Register tiling
- Tensor Core (WMMA) implementation
