// =============================================================================
//  TP3 -- Matrix multiplication on the GPU: C = A B
//
//  A, B and C are N x N single-precision matrices, allocated as 1D arrays of
//  N*N floats (malloc on the CPU, cudaMalloc on the GPU), with these layouts:
//
//      A is row-major    : A(i, k) is at  A[i * N + k]
//      B is column-major : B(k, j) is at  B[k + j * N]
//      C is row-major    : C(i, j) is at  C[i * N + j]
//
//  and C(i, j) = sum over k of A(i, k) * B(k, j). N is any positive integer:
//  every kernel must work whatever N (round the grid up, guard the threads
//  that fall outside the matrix).
//
//  EXERCISES -- one kernel each; kernel 1 is given as a worked example, with
//  its comment block and its launch in main():
//    1. one thread per element of C, blocks of one thread, 2D grid
//    2. 1D blocks along a row of C: a block computes blockDim.x consecutive
//       elements of one row, one element per thread
//    3. 1D blocks along a column of C: a block computes blockDim.x
//       consecutive elements of one column
//    4. 2D blocks, warps horizontal: a block computes a blockDim.x x
//       blockDim.y tile of C, one element per thread, threadIdx.x running
//       along the row (a warp = consecutive columns)
//    5. 2D blocks, warps vertical: the same tile, threadIdx.x running down
//       the column (a warp = consecutive rows)
//    6. cuBLAS: the same product with cublasSgemm; then set n in main()
//       to 128, 256, ..., 4096 in turn, rerun, and compare your kernels
//       with cuBLAS
//
//  For every kernel except the first, write in its comment block the
//  ANALYSIS of its memory accesses: for each read of A and of B and for the
//  write of C, what do the 32 threads of a warp touch at one instruction,
//  and is the access coalesced or not?
//  In main(), set the grid and block dimensions of each kernel (the launch
//  is written; as given, kernels 2-5 run with 1 block of 1 thread, so the
//  program compiles and runs from the start and prints WRONG until they
//  are written).
//
//  The program checks every result against a CPU reference and prints the
//  time and the performance in GFLOP/s.
//
//  Build:  nvcc -O2 -arch=native matmul.cu -o matmul -lcublas
//  Run:    ./matmul            (the matrix size n is set in main())
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ----------------------------------------------------------------------------
//  Error checking (wrap every CUDA runtime call, kernel launch and cuBLAS call)
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

#define CUBLAS_CHECK(call) do { \
  cublasStatus_t st = (call); \
  if (st != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__, __LINE__, (int)st); \
    exit(EXIT_FAILURE); \
  } } while (0)

// ----------------------------------------------------------------------------
//  Block sizes
// ----------------------------------------------------------------------------
#define BS_1D  256    // threads per block of the 1D-block kernels (2, 3)
#define BS_2D  32     // side of the thread tile of the 2D-block kernels (4, 5)

// Test data: integer values that depend on both indices and are not
// symmetric, so that a transposed access of A or B, or a wrong column of B,
// changes the result. Every product is at most 6 * 4 = 24 and every partial
// sum at most 24 * N < 2^24, so all sums are exact in float whatever the
// order of the additions: the GPU result can be compared exactly.
inline float aval(int i, int k) { return (float)((i + 2 * k) % 7); }
inline float bval(int k, int j) { return (float)((3 * k + j) % 5); }

// =============================================================================
//  1. One thread per element of C, blocks of one thread, 2D grid of N x N
// =============================================================================
//  Thread (i, j) = (blockIdx.x, blockIdx.y) computes the dot product of row i
//  of A and column j of B: n multiply-adds, 2n loads from global memory.
//  The grid has exactly N x N blocks: no guard needed.
//  Given as a worked example. (One thread per block: a warp has a single
//  active thread, so the coalescing analysis of the other kernels does not
//  apply here -- every access is one float of one sector.)
// =============================================================================
__global__ void matmul_blocks(int n, const float *A, const float *B, float *C)
{
  // TODO
}

// =============================================================================
//  2. 1D blocks along a row of C: blockDim.x consecutive elements of one row
// =============================================================================
//  blockIdx.x = the row i; blockIdx.y = which piece of the row;
//  j = blockIdx.y * blockDim.x + threadIdx.x. Grid: N x ceil(N / blockDim.x),
//  guard on j.
//
//  ANALYSIS (to write here): for each global memory access of this kernel --
//  the reads of A and of B at one step k, the write of C -- say what the 32
//  threads of a warp touch at that instruction and whether the access is
//  coalesced.
// =============================================================================
__global__ void matmul_1d_row(int n, const float *A, const float *B, float *C)
{
  // TODO
}

// =============================================================================
//  3. 1D blocks along a column of C: blockDim.x consecutive elements of one column
// =============================================================================
//  blockIdx.x = the column j; blockIdx.y = which piece of the column;
//  i = blockIdx.y * blockDim.x + threadIdx.x. Grid: N x ceil(N / blockDim.x),
//  guard on i.
//
//  ANALYSIS (to write here): for each global memory access of this kernel --
//  the reads of A and of B at one step k, the write of C -- say what the 32
//  threads of a warp touch at that instruction and whether the access is
//  coalesced.
//  Compare with kernel 2.
// =============================================================================
__global__ void matmul_1d_column(int n, const float *A, const float *B, float *C)
{
  // TODO
}

// =============================================================================
//  4. 2D blocks, warps horizontal: a blockDim.x x blockDim.y tile of C,
//     threadIdx.x along the row
// =============================================================================
//  j = blockIdx.x * blockDim.x + threadIdx.x,  i = blockIdx.y * blockDim.y +
//  threadIdx.y: threadIdx.x runs along the row of C, so with blockDim.x = 32
//  a warp is 32 consecutive elements of one row. Grid: ceil(N / blockDim.x) x
//  ceil(N / blockDim.y), guard on i and j.
//
//  ANALYSIS (to write here): for each global memory access of this kernel --
//  the reads of A and of B at one step k, the write of C -- say what the 32
//  threads of a warp touch at that instruction and whether the access is
//  coalesced.
//  Compare with kernel 2.
// =============================================================================
__global__ void matmul_2d_horizontal(int n, const float *A, const float *B, float *C)
{
  // TODO
}

// =============================================================================
//  5. 2D blocks, warps vertical: the same tile, threadIdx.x down the column
// =============================================================================
//  i = blockIdx.x * blockDim.x + threadIdx.x,  j = blockIdx.y * blockDim.y +
//  threadIdx.y: threadIdx.x runs down the column of C, so with blockDim.x = 32
//  a warp is 32 consecutive elements of one column. Grid: ceil(N / blockDim.x)
//  x ceil(N / blockDim.y), guard on i and j.
//
//  ANALYSIS (to write here): for each global memory access of this kernel --
//  the reads of A and of B at one step k, the write of C -- say what the 32
//  threads of a warp touch at that instruction and whether the access is
//  coalesced.
//  Compare with kernels 3 and 4. Is there a mapping of the thread
//  indices that coalesces the reads of A and of B at the same time?
// =============================================================================
__global__ void matmul_2d_vertical(int n, const float *A, const float *B, float *C)
{
  // TODO
}

// =============================================================================
//  6. cuBLAS
// =============================================================================
//  Compute C = A B with cuBLAS, into dC in OUR layout (row-major), so that
//  the result can be compared with the kernels' (see main()).
//
//  cuBLAS is column-major, and
//      cublasSgemm(handle, opX, opY, m, n, k, &alpha, X, ldx, Y, ldy, &beta, C, ldc)
//  computes C = alpha * op(X) * op(Y) + beta * C for column-major X (m x k
//  after op), Y (k x n after op) and C (m x n), with op = CUBLAS_OP_N (as
//  is) or CUBLAS_OP_T (transposed); ld* are the leading dimensions (the
//  distance between consecutive columns). Think about what our three
//  arrays are when read column-major, and which product gives our C.
//
//  With FP32 data of small integers the result is exact, so it can be
//  compared element by element with our kernels.
// =============================================================================
void cublasMatmul(cublasHandle_t handle, int n, const float *dA, const float *dB, float *dC)
{
  // TODO: one cublasSgemm call (wrapped in CUBLAS_CHECK)
  (void)handle; (void)n; (void)dA; (void)dB; (void)dC;
}

// =============================================================================
//  Host side: clock, data, verification
// =============================================================================
// Wall-clock time in milliseconds. A kernel launch returns at once: call
// cudaDeviceSynchronize() (CUDA_CHECK_KERNEL does) before reading the clock.
double now(void)
{
  using namespace std::chrono;
  return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

void initialise(int n, float *A, float *B)
{
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < n; k++) { A[i * n + k] = aval(i, k); }
  }
  for (int k = 0; k < n; k++) {
    for (int j = 0; j < n; j++) { B[k + j * n] = bval(k, j); }
  }
}

// Reference value of C(i, j), on the CPU.
float reference(int n, const float *A, const float *B, int i, int j)
{
  float sum = 0.0f;
  for (int k = 0; k < n; k++) { sum += A[i * n + k] * B[k + j * n]; }
  return sum;
}

// Compare C with the reference. Every element is checked when n <= 512;
// above that, 96 whole rows are checked (the first 32, the last 32 and 32
// spread in between), so that the boundary tiles and every column are
// covered. Returns the number of wrong elements among those checked and
// prints the first one.
int verify(int n, const float *A, const float *B, const float *C)
{
  int errors = 0, firstI = -1, firstJ = -1;
  int rows[96], numRows = 0;
  if (n <= 512) {
    numRows = n;
  } else {
    for (int r = 0; r < 32; r++) { rows[numRows++] = r; }
    for (int r = 0; r < 32; r++) { rows[numRows++] = 32 + (int)((long long)r * (n - 64) / 32); }
    for (int r = 0; r < 32; r++) { rows[numRows++] = n - 32 + r; }
  }
  for (int r = 0; r < numRows; r++) {
    int i = (n <= 512) ? r : rows[r];
    for (int j = 0; j < n; j++) {
      float ref = reference(n, A, B, i, j);
      // written so that a NaN in C counts as wrong (NaN fails every comparison)
      if (!(fabsf(C[i * n + j] - ref) <= 1e-3f * fabsf(ref) + 1e-3f)) {
        if (errors == 0) { firstI = i; firstJ = j; }
        errors++;
      }
    }
  }
  if (errors) {
    printf("      first wrong element: C(%d, %d) = %g, expected %g\n",
           firstI, firstJ, C[firstI * n + firstJ], reference(n, A, B, firstI, firstJ));
  }
  return errors;
}

// After a kernel: copy dC back, check it, print one line (result, time in
// ms, GFLOP/s), then clear dC so that the next kernel starts from zeros.
void check(const char *name, int n, double ms, const float *A, const float *B, float *C, float *dC)
{
  size_t bytes = (size_t)n * n * sizeof(float);
  CUDA_CHECK(cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost));
  int errors = verify(n, A, B, C);
  double gflops = 2.0 * n * (double)n * n / (ms * 1e-3) / 1e9;
  if (errors) { printf("%-30s %6d  WRONG (%d)\n", name, n, errors); }
  else        { printf("%-30s %6d  ok      %10.3f ms %10.1f GFLOP/s\n", name, n, ms, gflops); }
  CUDA_CHECK(cudaMemset(dC, 0, bytes));
}

// =============================================================================
//  Driver
// =============================================================================
int main(void)
{
  const int n = 250;   // the matrix size: change it here and rerun
  size_t bytes = (size_t)n * n * sizeof(float);
  float *A = (float *)malloc(bytes), *B = (float *)malloc(bytes), *C = (float *)malloc(bytes);
  initialise(n, A, B);
  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));
  CUDA_CHECK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, bytes));
  printf("%-30s %6s  %-7s %13s %18s\n", "kernel", "N", "result", "time", "performance");
  double t0;
  // Every kernel is launched twice. The first launch loads the kernel onto
  // the GPU (and, for the first kernel, finishes setting the device up) and
  // is not timed; the clock is read around the second one.

  // ---- 1. One thread per element, blocks of one thread --------------------
  dim3 grid1(n, n), block1(1);
  matmul_blocks<<<grid1, block1>>>(n, dA, dB, dC);   // warm-up
  CUDA_CHECK_KERNEL();
  t0 = now();
  matmul_blocks<<<grid1, block1>>>(n, dA, dB, dC);   // timed
  CUDA_CHECK_KERNEL();
  check("1 one thread per block", n, now() - t0, A, B, C, dC);

  // ---- 2. 1D blocks along a row of C --------------------------------------
  // TODO: grid and block of kernel 2 (as given: 1 block of 1 thread)
  dim3 grid2(1), block2(1);
  matmul_1d_row<<<grid2, block2>>>(n, dA, dB, dC);   // warm-up
  CUDA_CHECK_KERNEL();
  t0 = now();
  matmul_1d_row<<<grid2, block2>>>(n, dA, dB, dC);   // timed
  CUDA_CHECK_KERNEL();
  check("2 1D blocks, row of C", n, now() - t0, A, B, C, dC);

  // ---- 3. 1D blocks along a column of C -----------------------------------
  // TODO: grid and block of kernel 3 (as given: 1 block of 1 thread)
  dim3 grid3(1), block3(1);
  matmul_1d_column<<<grid3, block3>>>(n, dA, dB, dC);   // warm-up
  CUDA_CHECK_KERNEL();
  t0 = now();
  matmul_1d_column<<<grid3, block3>>>(n, dA, dB, dC);   // timed
  CUDA_CHECK_KERNEL();
  check("3 1D blocks, column of C", n, now() - t0, A, B, C, dC);

  // ---- 4. 2D blocks, warps horizontal -------------------------------------
  // TODO: grid and block of kernel 4 (as given: 1 block of 1 thread)
  dim3 grid4(1), block4(1);
  matmul_2d_horizontal<<<grid4, block4>>>(n, dA, dB, dC);   // warm-up
  CUDA_CHECK_KERNEL();
  t0 = now();
  matmul_2d_horizontal<<<grid4, block4>>>(n, dA, dB, dC);   // timed
  CUDA_CHECK_KERNEL();
  check("4 2D blocks, warps horizontal", n, now() - t0, A, B, C, dC);

  // ---- 5. 2D blocks, warps vertical ---------------------------------------
  // TODO: grid and block of kernel 5 (as given: 1 block of 1 thread)
  dim3 grid5(1), block5(1);
  matmul_2d_vertical<<<grid5, block5>>>(n, dA, dB, dC);   // warm-up
  CUDA_CHECK_KERNEL();
  t0 = now();
  matmul_2d_vertical<<<grid5, block5>>>(n, dA, dB, dC);   // timed
  CUDA_CHECK_KERNEL();
  check("5 2D blocks, warps vertical", n, now() - t0, A, B, C, dC);

  // ---- 6. cuBLAS ----------------------------------------------------------
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  cublasMatmul(handle, n, dA, dB, dC);     // warm-up: cuBLAS sets itself up on its first call
  CUDA_CHECK(cudaDeviceSynchronize());
  t0 = now();
  cublasMatmul(handle, n, dA, dB, dC);
  CUDA_CHECK(cudaDeviceSynchronize());
  check("6 cuBLAS", n, now() - t0, A, B, C, dC);
  CUBLAS_CHECK(cublasDestroy(handle));

  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
  free(A); free(B); free(C);
  return 0;
}
