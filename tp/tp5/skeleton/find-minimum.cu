// =============================================================================
//  TP5 -- The minimum of an array on the GPU: a reduction
//
//  a is an array of n floats (malloc on the CPU, cudaMalloc on the GPU),
//  filled with values between 0 and 999, with ONE element set to -1 at a
//  random position: the minimum is -1, and a kernel that skips a part of the
//  array is caught. n is any positive integer, not a multiple of the block
//  size: every kernel must guard the threads past the end of a.
//
//  A reduction reads every element once and does one comparison per
//  element: it is bound by the bandwidth of global memory. For each version
//  the program prints the time to get the minimum into a CPU variable
//  (kernel + copy of the partials + CPU finish) and the effective bandwidth,
//  4 n bytes / time. The read-only kernel (given) reads a once with
//  coalesced loads and computes nothing: its bandwidth is the ceiling.
//
//  EXERCISES -- one version each, in order:
//    1. min_interleaved : one element per thread; the block's minimum through
//                         a tree in shared memory, interleaved pairs; one
//                         partial per block, the CPU takes the minimum of the
//                         partials
//    2. min_contiguous  : the same tree with the busy threads packed at the
//                         front
//    3. min_multi       : each thread first takes the minimum of several
//                         elements (grid-stride loop), then the tree: G blocks
//    4. min_lastwarp    : version 3, the last steps by warp 0 alone, without
//                         __syncthreads()
//    5. two launches    : the partials of version 4 reduced by a second
//                         launch of the same kernel, so that the CPU reads one
//                         value (nothing to write but the launch dimensions)
//  For each version, write the kernel's body and its grid and block
//  dimensions in main() (the launches are written; as given, every kernel
//  runs with 1 block of 1 thread, so the program compiles and runs from the
//  start and prints WRONG until the kernels are written).
//
//  Build:  nvcc -O2 -arch=native find-minimum.cu -o find-minimum
//  Run:    ./find-minimum        (the array size n is set in main())
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <chrono>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
//  Error checking (wrap every CUDA runtime call and kernel launch)
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } } while (0)

// A kernel launch returns nothing: check the launch itself, then wait for it
// to finish and check what happened while it ran.
#define CUDA_CHECK_KERNEL() do { \
  CUDA_CHECK(cudaGetLastError()); \
  CUDA_CHECK(cudaDeviceSynchronize()); } while (0)

// ----------------------------------------------------------------------------
//  Block size and grid size
// ----------------------------------------------------------------------------
#define BS 256    // threads per block: a power of two (the tree halves it at each step)
#define G  1024   // blocks of versions 3, 4 and 5 (the grid-stride kernels)

// =============================================================================
//  Given: read a once, compute nothing -- the ceiling
// =============================================================================
//  G blocks, a grid-stride loop: thread t of block b reads the elements
//  b*BS + t, b*BS + t + G*BS, ... (a warp reads 32 consecutive floats at each
//  turn: coalesced). The test is never true, so nothing is written, but the
//  compiler cannot know it: the loads stay. Its time is the time to read a.
// =============================================================================
__global__ void read_only(int n, const float *a, float *partial)
{
  int t = threadIdx.x;
  float x = 0.0f;
  for (int i = blockIdx.x * BS + t; i < n; i += gridDim.x * BS) {
    x = (a[i] == -2.0f) ? 1.0f : x;
  }
  if (x != 0.0f) { partial[blockIdx.x] = x; }
}

// =============================================================================
//  1. The tree with interleaved pairs
// =============================================================================
//  One element per thread: thread t of block b loads a[b*BS + t] into s[t]
//  (FLT_MAX past the end of a: the identity of the minimum), then a barrier.
//  Then the tree: at step stride = 1, 2, 4, ..., BS/2, thread t replaces
//  s[t] by fminf(s[t], s[t + stride]) when t is a multiple of 2*stride, and
//  a barrier after every step. At the end s[0] is the minimum of the
//  block's BS elements: thread 0 writes it to partial[b].
//  Grid: ceil(n / BS) blocks of BS threads; the CPU takes the minimum of the
//  ceil(n / BS) partials (done by main()).
//  How many warps execute each step?
// =============================================================================
__global__ void min_interleaved(int n, const float *a, float *partial)
{
  // TODO
}

// =============================================================================
//  2. The tree with contiguous threads
// =============================================================================
//  The same load. Then at step stride = BS/2, BS/4, ..., 1, the threads
//  t < stride replace s[t] by fminf(s[t], s[t + stride]), a barrier after
//  every step. Which warps execute each step now?
//  Grid: ceil(n / BS) blocks of BS threads; the CPU finishes.
// =============================================================================
__global__ void min_contiguous(int n, const float *a, float *partial)
{
  // TODO
}

// =============================================================================
//  3. Several elements per thread, then the tree
// =============================================================================
//  G blocks only. Thread t of block b walks the array with a grid-stride
//  loop, i = b*BS + t, b*BS + t + G*BS, ... (a warp reads 32 consecutive
//  floats at each turn: coalesced), and keeps the minimum of its elements
//  in a register x (FLT_MAX at the start). Then s[t] = x, a barrier, and the
//  tree of version 2 on the BS values: one tree per block instead of one per
//  BS elements, and only G partials for the CPU.
//  Grid: G blocks of BS threads.
// =============================================================================
__global__ void min_multi(int n, const float *a, float *partial)
{
  // TODO
}

// =============================================================================
//  4. The last warp alone
// =============================================================================
//  Version 3, with the tree stopped at stride > 32: from then on only warp 0
//  (threads t < 32) has busy threads, so it finishes alone, without
//  __syncthreads() -- the other warps are done. Inside the warp, put
//  __syncwarp() after each step to keep the 32 threads in step (since Volta
//  the threads of a warp may run ahead of each other: the warp barrier is
//  required). Thread 0 writes fminf(s[0], s[1]) to partial[b].
//  Grid: G blocks of BS threads.
// =============================================================================
__global__ void min_lastwarp(int n, const float *a, float *partial)
{
  // TODO
}

// =============================================================================
//  5. Two launches: the partials reduced on the GPU
// =============================================================================
//  Nothing to write here: version 4 is launched on a (G blocks: G partials),
//  then launched AGAIN on the array of G partials, with ONE block of BS
//  threads, which writes the minimum of the G partials to result[0]. The
//  kernel works on any array of any length, so on the partials too. The CPU
//  copies one float. See main().
// =============================================================================

// ----------------------------------------------------------------------------
//  Timing, checking, reporting
// ----------------------------------------------------------------------------
double now(void)
{
  using namespace std::chrono;
  return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

// Copy the `count` partials back and take their minimum (for versions 1-4).
float finish_on_cpu(const float *dPartial, float *partial, int count)
{
  CUDA_CHECK(cudaMemcpy(partial, dPartial, count * sizeof(float), cudaMemcpyDeviceToHost));
  float m = FLT_MAX;
  for (int b = 0; b < count; b++) { if (partial[b] < m) { m = partial[b]; } }
  return m;
}

// Print one line: the version, the minimum found, ok / WRONG, the time and
// the bandwidth (bytes of a read / time).
void report(const char *name, int n, float found, double ms)
{
  double gbs = (double)n * sizeof(float) / (ms * 1e-3) / 1e9;
  if (found == -1.0f) { printf("%-24s ok    (min = %5.1f) %9.3f ms %8.1f GB/s\n", name, found, ms, gbs); }
  else                { printf("%-24s WRONG (min = %5.1f, expected -1.0)\n", name, found); }
}

// GPUs raise their clock after a few hundred ms of load: without this spin
// first, the first timings would run at a lower clock.
__global__ void settle_clock()
{
  long long t0 = clock64();
  while (clock64() - t0 < 400000000LL) { }
}

// =============================================================================
//  Driver
// =============================================================================
int main(void)
{
  const int n = 16000001;   // the array size: 16 M floats, not a multiple of BS
  size_t bytes = (size_t)n * sizeof(float);
  // The array: values in 0..999, one -1 at a random position in the second half
  srand(1234);
  float *a = (float *)malloc(bytes);
  for (int i = 0; i < n; i++) { a[i] = (float)(rand() % 1000); }
  a[n / 2 + rand() % (n / 2)] = -1.0f;
  int blocks1 = (n + BS - 1) / BS;    // one element per thread: versions 1 and 2
  float *partial = (float *)malloc(blocks1 * sizeof(float));
  for (int b = 0; b < blocks1; b++) { partial[b] = 0.0f; }   // touch the pages now, not inside a timing
  float *da, *dPartial, *dResult, result;
  CUDA_CHECK(cudaMalloc(&da, bytes));
  CUDA_CHECK(cudaMalloc(&dPartial, blocks1 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dResult, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(partial, dPartial, blocks1 * sizeof(float), cudaMemcpyDeviceToHost));   // a first copy back, untimed
  settle_clock<<<1, 32>>>();
  CUDA_CHECK_KERNEL();
  dim3 grid0(G), block0(BS);
  for (int k = 0; k < 2; k++) {   // two untimed launches: the first timed launch of a program pays a one-time cost
    read_only<<<grid0, block0>>>(n, da, dPartial);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(&result, dPartial, sizeof(float), cudaMemcpyDeviceToHost));
  }
  printf("n = %d floats (%zu MB), BS = %d, G = %d\n", n, bytes >> 20, BS, G);
  printf("%-24s %-21s %12s %13s\n", "version", "result", "time", "bandwidth");
  double t0, ms;
  float found;
  // Every version is launched twice: the first launch loads the kernel onto
  // the GPU and is not timed; the clock is read around the second one, up to
  // the moment the minimum is in a CPU variable.

  // ---- 1. interleaved tree -------------------------------------------------
  // TODO: grid and block of version 1 (as given: 1 block of 1 thread)
  dim3 grid1(1), block1(1);
  min_interleaved<<<grid1, block1>>>(n, da, dPartial);   // warm-up
  CUDA_CHECK_KERNEL();
  t0 = now();
  min_interleaved<<<grid1, block1>>>(n, da, dPartial);   // timed
  CUDA_CHECK_KERNEL();
  found = finish_on_cpu(dPartial, partial, grid1.x);
  ms = now() - t0;
  report("1 interleaved", n, found, ms);

  // ---- 2. contiguous tree --------------------------------------------------
  // TODO: grid and block of version 2 (as given: 1 block of 1 thread)
  dim3 grid2(1), block2(1);
  min_contiguous<<<grid2, block2>>>(n, da, dPartial);   // warm-up
  CUDA_CHECK_KERNEL();
  t0 = now();
  min_contiguous<<<grid2, block2>>>(n, da, dPartial);   // timed
  CUDA_CHECK_KERNEL();
  found = finish_on_cpu(dPartial, partial, grid2.x);
  ms = now() - t0;
  report("2 contiguous", n, found, ms);

  // ---- 3. several elements per thread --------------------------------------
  // TODO: grid and block of version 3 (as given: 1 block of 1 thread)
  dim3 grid3(1), block3(1);
  min_multi<<<grid3, block3>>>(n, da, dPartial);   // warm-up
  CUDA_CHECK_KERNEL();
  t0 = now();
  min_multi<<<grid3, block3>>>(n, da, dPartial);   // timed
  CUDA_CHECK_KERNEL();
  found = finish_on_cpu(dPartial, partial, grid3.x);
  ms = now() - t0;
  report("3 several per thread", n, found, ms);

  // ---- 4. the last warp alone ----------------------------------------------
  // TODO: grid and block of version 4 (as given: 1 block of 1 thread)
  dim3 grid4(1), block4(1);
  min_lastwarp<<<grid4, block4>>>(n, da, dPartial);   // warm-up
  CUDA_CHECK_KERNEL();
  t0 = now();
  min_lastwarp<<<grid4, block4>>>(n, da, dPartial);   // timed
  CUDA_CHECK_KERNEL();
  found = finish_on_cpu(dPartial, partial, grid4.x);
  ms = now() - t0;
  report("4 last warp", n, found, ms);

  // ---- 5. two launches: the partials reduced by the GPU --------------------
  // TODO: grid and block of the second launch, on the G partials (as given: 1 block of 1 thread)
  dim3 grid5(1), block5(1);
  t0 = now();
  min_lastwarp<<<grid4, block4>>>(n, da, dPartial);      // G partials
  CUDA_CHECK_KERNEL();
  min_lastwarp<<<grid5, block5>>>(grid4.x, dPartial, dResult);   // G -> 1
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpy(&result, dResult, sizeof(float), cudaMemcpyDeviceToHost));
  ms = now() - t0;
  report("5 two launches", n, result, ms);

  // ---- given: read only, the ceiling ---------------------------------------
  read_only<<<grid0, block0>>>(n, da, dPartial);   // warm-up
  CUDA_CHECK_KERNEL();
  t0 = now();
  read_only<<<grid0, block0>>>(n, da, dPartial);   // timed
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpy(&result, dPartial, sizeof(float), cudaMemcpyDeviceToHost));
  ms = now() - t0;
  report("read only (given)", n, -1.0f, ms);   // the ceiling: no reduction can beat it

  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(dPartial));
  CUDA_CHECK(cudaFree(dResult));
  free(a);
  free(partial);
  return 0;
}
