// bank-ways.cu -- how many passes does a shared memory access take?
//
// One warp per block; thread t of the warp reads the word s[idx(t) + 32 p]
// (same bank for every p, so the index changes at every iteration and the
// compiler cannot hoist the load) a few hundred times. The patterns of the
// 1D exercise, plus the strides 4 and 8 for comparison:
//   s[t]             32 banks, 1 word each               1-way
//   s[2 t]           16 banks, 2 words each              2-way
//   s[4 t]            8 banks, 4 words each              4-way
//   s[32 (t % 4)]     bank 0, 4 distinct words, 8 threads read each one
//   s[8 t]            4 banks, 8 words each              8-way
//   s[32 (t % 8)]     bank 0, 8 distinct words, 4 threads read each one
//   s[32 t]           bank 0, 32 words                   32-way
//   s[0]              one word for the whole warp        broadcast
// The time of a pattern divided by the time of s[t] is its number of passes.
//
// Build:  nvcc -O2 -arch=native bank-ways.cu -o bank-ways
//         (godbolt: -O2)
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

#define ITERS 2048
#define BLOCKS 4096

template <int PATTERN>
__device__ int idx(int t)
{
  switch (PATTERN) {
    case 0: return t;              // 1-way
    case 1: return 2 * t;          // 2-way
    case 2: return 4 * t;          // 4-way
    case 3: return 32 * (t % 4);   // 4 words in bank 0, read by 8 threads each
    case 4: return 8 * t;          // 8-way
    case 5: return 32 * (t % 8);   // 8 words in bank 0, read by 4 threads each
    case 6: return 32 * t;         // 32-way
    default: return 0;             // broadcast
  }
}

template <int PATTERN>
__global__ void bankKernel(float *out)
{
  __shared__ float s[2048];
  int t = threadIdx.x;
  for (int w = t; w < 2048; w += 32) { s[w] = w; }
  __syncthreads();
  float x = 0.0f;
  int base = idx<PATTERN>(t);
  for (int p = 0; p < ITERS; p++) { x += s[(base + 32 * p) & 2047]; }
  out[blockIdx.x * 32 + t] = x;
}

double now(void)
{
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

template <int PATTERN>
double timeIt(float *d, const char *name)
{
  bankKernel<PATTERN><<<BLOCKS, 32>>>(d);   // warm-up
  CUDA_CHECK(cudaDeviceSynchronize());
  double best = 1e30;
  for (int r = 0; r < 5; r++) {
    double t0 = now();
    bankKernel<PATTERN><<<BLOCKS, 32>>>(d);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t = now() - t0;
    if (t < best) best = t;
  }
  printf("%-16s %8.3f ms\n", name, best);
  return best;
}

int main(void)
{
  float *d;
  CUDA_CHECK(cudaMalloc(&d, BLOCKS * 32 * sizeof(float)));
  double t1 = timeIt<0>(d, "s[t]");
  double t2 = timeIt<1>(d, "s[2t]");
  double t4 = timeIt<2>(d, "s[4t]");
  double tf = timeIt<3>(d, "s[32(t%4)]");
  double t8 = timeIt<4>(d, "s[8t]");
  double tg = timeIt<5>(d, "s[32(t%8)]");
  double t32 = timeIt<6>(d, "s[32t]");
  double tb = timeIt<7>(d, "s[0]");
  printf("relative to s[t]: 2t %.1f  4t %.1f  32(t%%4) %.1f  8t %.1f  32(t%%8) %.1f  32t %.1f  s[0] %.1f\n",
         t2 / t1, t4 / t1, tf / t1, t8 / t1, tg / t1, t32 / t1, tb / t1);
  CUDA_CHECK(cudaFree(d));
  return 0;
}
