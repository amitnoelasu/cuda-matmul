// matmul.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#define CHECK_CUDA(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    printf("CUDA Error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

__global__ void matmul_naive(const float* A, const float* B, float* C,
                             int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    // baseline: every multiply reads A and B from global memory
    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

constexpr int TILE = 16;

// shared memory tiling kernel
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float sum = 0.0f;

  // move along K in TILE chunks
  for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
    int kA = t * TILE + threadIdx.x; // column of A
    int kB = t * TILE + threadIdx.y; // row of B

    // load into shared memory (with bounds checks)
    As[threadIdx.y][threadIdx.x] = (row < M && kA < K) ? A[row * K + kA] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (kB < K && col < N) ? B[kB * N + col] : 0.0f;

    __syncthreads();

    // reuse the loaded tile
    #pragma unroll
    for (int k = 0; k < TILE; k++) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); i++) {
    m = std::max(m, std::fabs(a[i] - b[i]));
  }
  return m;
}

static float time_kernel_ms(void (*kernel_launch)(), int warmup, int iters) {
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; i++) kernel_launch();
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; i++) kernel_launch();
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  return ms / iters;
}

int main(int argc, char** argv) {
  // simple defaults; can pass one size arg: ./matmul 2048
  int M = 1024, N = 1024, K = 1024;
  if (argc >= 2) {
    int s = std::atoi(argv[1]);
    M = N = K = s;
  }

  printf("Running GEMM: C[%dx%d] = A[%dx%d] * B[%dx%d]\n", M, N, M, K, K, N);

  size_t bytesA = (size_t)M * K * sizeof(float);
  size_t bytesB = (size_t)K * N * sizeof(float);
  size_t bytesC = (size_t)M * N * sizeof(float);

  std::vector<float> hA((size_t)M * K);
  std::vector<float> hB((size_t)K * N);
  std::vector<float> hC1((size_t)M * N, 0.0f);
  std::vector<float> hC2((size_t)M * N, 0.0f);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : hA) x = dist(rng);
  for (auto& x : hB) x = dist(rng);

  float *dA, *dB, *dC;
  CHECK_CUDA(cudaMalloc(&dA, bytesA));
  CHECK_CUDA(cudaMalloc(&dB, bytesB));
  CHECK_CUDA(cudaMalloc(&dC, bytesC));

  CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  // 1) Naive
  auto launch_naive = [&]() {
    matmul_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
  };
  // function pointer wrapper
  auto naive_wrapper = +[]() {};
  (void)naive_wrapper;

  CHECK_CUDA(cudaMemset(dC, 0, bytesC));
  float naive_ms;
  {
    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    for (int i = 0; i < 5; i++) launch_naive();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(s));
    for (int i = 0; i < 50; i++) launch_naive();
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float total = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total, s, e));
    naive_ms = total / 50.0f;
    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
  }
  CHECK_CUDA(cudaMemcpy(hC1.data(), dC, bytesC, cudaMemcpyDeviceToHost));

  // 2) Tiled (shared memory)
  auto launch_tiled = [&]() {
    matmul_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
  };

  CHECK_CUDA(cudaMemset(dC, 0, bytesC));
  float tiled_ms;
  {
    cudaEvent_t s, e;
    CHECK_CUDA(cudaEventCreate(&s));
    CHECK_CUDA(cudaEventCreate(&e));
    for (int i = 0; i < 5; i++) launch_tiled();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(s));
    for (int i = 0; i < 50; i++) launch_tiled();
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float total = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total, s, e));
    tiled_ms = total / 50.0f;
    CHECK_CUDA(cudaEventDestroy(s));
    CHECK_CUDA(cudaEventDestroy(e));
  }
  CHECK_CUDA(cudaMemcpy(hC2.data(), dC, bytesC, cudaMemcpyDeviceToHost));

  // correctness (compare kernels against each other)
  float diff = max_abs_diff(hC1, hC2);
  printf("Max abs diff (naive vs tiled): %.6f\n", diff);

  // quick perf report
  double flops = 2.0 * (double)M * (double)N * (double)K;
  double naive_gflops = (flops / (naive_ms / 1e3)) / 1e9;
  double tiled_gflops = (flops / (tiled_ms / 1e3)) / 1e9;

  printf("\n=== Performance ===\n");
  printf("Naive: %.3f ms, %.2f GFLOP/s\n", naive_ms, naive_gflops);
  printf("Tiled: %.3f ms, %.2f GFLOP/s\n", tiled_ms, tiled_gflops);
  printf("Speedup: %.2fx\n", naive_ms / tiled_ms);

  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));
  return 0;
}
