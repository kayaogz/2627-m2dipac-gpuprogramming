// =============================================================================
//  TP4 -- Matrix multiplication on the GPU, through shared memory
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
//  Kernels 1 to 5 are TP3's, given with their solutions, and so is the
//  cuBLAS call that serves as the benchmark:
//    1. one thread per element of C, blocks of one thread, 2D grid
//    2. 1D blocks along a row of C
//    3. 1D blocks along a column of C
//    4. 2D blocks of BSXY x BSXY threads, warps horizontal
//    5. 2D blocks of BSXY x BSXY threads, warps vertical
//    -- cuBLAS (cublasSgemm): not a kernel of ours, the benchmark
//  Kernel 4 and kernel 5 ended TP3 on a dead end: with A row-major and B
//  column-major, no mapping of one thread per element of C makes the reads
//  of BOTH operands coalesced.
//
//  EXERCISE -- the way out, kernel 6:
//    6. shared memory: 2D blocks of BSXY x BSXY threads, each block computes
//       a BSXY x BSXY tile of C by walking along the band of A and the band
//       of B it needs, BSXY x BSXY tiles at a time, through shared memory
//       (see the comment block of matmul_shmem). Then set n in main() to
//       256, 512, 1024, 2048 (and 4096 on a lab GPU) in turn, rerun, and
//       compare kernel 6 with kernel 4 and with cuBLAS.
//
//  In main(), set the grid and block dimensions of kernel 6 (the launch is
//  written; as given, it runs with 1 block of 1 thread, so the program
//  compiles and runs from the start and prints WRONG for kernel 6 until it
//  is written).
//
//  The program checks every result against a CPU reference and prints the
//  time and the performance in GFLOP/s.
//
//  Build:  nvcc -O2 -arch=native matmul-shmem.cu -o matmul-shmem -lcublas
//  Run:    ./matmul-shmem      (the matrix size n is set in main())
//  The program calls cuBLAS: it MUST be linked with -lcublas. On godbolt,
//  the compiler options of the pane that runs it must read -O2 -lcublas;
//  without -lcublas the program does not link.
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
#define BSXY   32     // side of the 2D blocks (4, 5, 6) and of the shared tiles (6)

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
//
//  Coalescing: a warp is one thread; every access is one sector fetched for
//  4 useful bytes (1/8 of the bytes used), and 31 of 32 threads idle. Not
//  coalesced for A, B or C -- not a question of layout but of block size.
// =============================================================================
__global__ void matmul_blocks(int n, const float *A, const float *B, float *C)
{
  int i = blockIdx.x;
  int j = blockIdx.y;
  float sum = 0.0f;
  for (int k = 0; k < n; k++) { sum += A[i * n + k] * B[k + j * n]; }
  C[i * n + j] = sum;
}

// =============================================================================
//  2. 1D blocks along a row of C: blockDim.x consecutive elements of one row
// =============================================================================
//  blockIdx.x = the row i; blockIdx.y = which piece of the row;
//  j = blockIdx.y * blockDim.x + threadIdx.x. Grid: N x ceil(N / blockDim.x),
//  guard on j (the last piece of each row has idle threads).
//
//  At step k, the 32 threads of a warp have the same i and 32 consecutive j:
//    A(i, k)  : the same address for all 32 threads -- a broadcast, 1 sector
//    B(k, j)  : B[k + j*n], consecutive j are 4n bytes apart -- 32 sectors,
//               one float used in each: NOT coalesced
//    C(i, j)  : C[i*n + j], 32 consecutive floats -- coalesced (one store per
//               thread, negligible next to the n loads of B)
// =============================================================================
__global__ void matmul_1d_row(int n, const float *A, const float *B, float *C)
{
  int i = blockIdx.x;
  int j = blockIdx.y * blockDim.x + threadIdx.x;
  if (j < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; k++) { sum += A[i * n + k] * B[k + j * n]; }
    C[i * n + j] = sum;
  }
}

// =============================================================================
//  3. 1D blocks along a column of C: blockDim.x consecutive elements of one column
// =============================================================================
//  blockIdx.x = which piece of the column, i = blockIdx.x * blockDim.x +
//  threadIdx.x; blockIdx.y = the column j. Grid: ceil(N / blockDim.x) x N,
//  guard on i.
//
//  At step k, the 32 threads of a warp have the same j and 32 consecutive i:
//    A(i, k)  : A[i*n + k], consecutive i are 4n bytes apart -- 32 sectors,
//               NOT coalesced
//    B(k, j)  : the same address for the whole warp -- a broadcast, 1 sector
//    C(i, j)  : C[i*n + j], consecutive i, 4n bytes apart -- NOT coalesced
//               (one store per thread)
//
//  The mirror image of kernel 2: the operand that was broadcast is now
//  strided and vice versa. The two kernels do the same number of loads;
//  what differs is which operand costs 32 sectors per warp per step.
// =============================================================================
__global__ void matmul_1d_column(int n, const float *A, const float *B, float *C)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y;
  if (i < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; k++) { sum += A[i * n + k] * B[k + j * n]; }
    C[i * n + j] = sum;
  }
}

// =============================================================================
//  4. 2D blocks, warps horizontal: a blockDim.x x blockDim.y tile of C,
//     threadIdx.x along the row
// =============================================================================
//  i = blockIdx.x * BSXY + threadIdx.y,  j = blockIdx.y * BSXY + threadIdx.x:
//  threadIdx.x is on the COLUMN index (the blocks are BSXY x BSXY, so the
//  tile side is BSXY in both directions). Grid: ceil(N/BSXY) x ceil(N/BSXY),
//  guard on i and j. With BSXY = 32, a warp is 32 consecutive elements of
//  one row of C.
//
//  At step k, a warp (32 consecutive threadIdx.x, same threadIdx.y) has the
//  same i and 32 consecutive j -- exactly the pattern of kernel 2:
//    A(i, k)  : broadcast, 1 sector
//    B(k, j)  : B[k + j*n], 4n bytes apart -- 32 sectors, NOT coalesced
//    C(i, j)  : 32 consecutive floats -- coalesced
//  The 2D block changes nothing to the memory traffic of a warp; it only
//  changes how many rows a block covers.
// =============================================================================
__global__ void matmul_2d_horizontal(int n, const float *A, const float *B, float *C)
{
  int i = blockIdx.x * BSXY + threadIdx.y;
  int j = blockIdx.y * BSXY + threadIdx.x;
  if (i < n && j < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; k++) { sum += A[i * n + k] * B[k + j * n]; }
    C[i * n + j] = sum;
  }
}

// =============================================================================
//  5. 2D blocks, warps vertical: the same tile, threadIdx.x down the column
// =============================================================================
//  i = blockIdx.x * BSXY + threadIdx.x,  j = blockIdx.y * BSXY + threadIdx.y:
//  threadIdx.x is on the ROW index. Grid: ceil(N/BSXY) x ceil(N/BSXY), guard
//  on i and j. With BSXY = 32, a warp is 32 consecutive elements of one
//  column of C.
//
//  At step k, a warp has 32 consecutive i and the same j -- the pattern of
//  kernel 3:
//    A(i, k)  : A[i*n + k], 4n bytes apart -- 32 sectors, NOT coalesced
//    B(k, j)  : broadcast, 1 sector
//    C(i, j)  : 4n bytes apart -- NOT coalesced (one store per thread)
//
//  Kernels 4 and 5 together show the dead end: with A row-major and B
//  column-major there is no assignment of the thread indices that makes
//  BOTH operands coalesced. A warp of 32 distinct elements of C must spread
//  over columns (then B is strided) or over rows (then A is strided). The
//  way out is to stop loading what a thread computes with: shared memory,
//  kernel 6.
// =============================================================================
__global__ void matmul_2d_vertical(int n, const float *A, const float *B, float *C)
{
  int i = blockIdx.x * BSXY + threadIdx.x;
  int j = blockIdx.y * BSXY + threadIdx.y;
  if (i < n && j < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; k++) { sum += A[i * n + k] * B[k + j * n]; }
    C[i * n + j] = sum;
  }
}

// =============================================================================
//  cuBLAS: the benchmark
// =============================================================================
//  cuBLAS is column-major. Our row-major A is, read column-major, the matrix
//  A^T; our row-major C is C^T; our B is column-major as it is. Since
//  C = A B is equivalent to C^T = B^T A^T, we ask cuBLAS for
//      C' = op(X) op(Y)  with  X = B (op = T), Y = A^T as stored (op = N)
//  and C' = B^T A^T = C^T column-major = our row-major C. All leading
//  dimensions are n.
//
//  With FP32 data of small integers the result is exact, so it can be
//  compared element by element with our kernels. (cuBLAS keeps full FP32
//  arithmetic unless the math mode is changed to allow TF32.)
// =============================================================================
void cublasMatmul(cublasHandle_t handle, int n, const float *dA, const float *dB, float *dC)
{
  const float one = 1.0f, zero = 0.0f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                           &one, dB, n, dA, n, &zero, dC, n));
}


// =============================================================================
//  6. Shared memory: BSXY x BSXY tiles of A and B
// =============================================================================
//  The same blocks and the same mapping as kernel 4: block (bx, by) computes
//  the BSXY x BSXY tile of C at rows bx*BSXY.. and columns by*BSXY..;
//  thread (tx, ty) owns the element C(i, j), i = bx*BSXY + ty, j = by*BSXY + tx.
//
//  Instead of walking along row i of A and column j of B alone, the block
//  walks along the band of rows of A and the band of columns of B it needs,
//  BSXY columns of A and BSXY rows of B at a time. At each step k0 = 0,
//  BSXY, 2*BSXY, ...:
//    1. the block copies the tile A(bx*BSXY.., k0..) and the tile
//       B(k0.., by*BSXY..) into shared memory, one element of each per
//       thread;
//    2. each thread multiplies row ty of the A tile by column tx of the B
//       tile (an inner product of length BSXY) and adds it to its C(i, j).
//  Do not forget to put barriers in the appropriate places, to avoid data
//  races among the warps of the block.
//  Where does the partial sum of C(i, j) live while the block walks: in
//  shared memory, or in a register / variable of the thread? (Who else
//  needs it?)
//
//  Requirements:
//    - both tile loads must be COALESCED: a warp (32 consecutive tx) must
//      read 32 consecutive floats of A, and 32 consecutive floats of B --
//      look at the layouts, and remember that the element a thread loads
//      need not be one it will multiply;
//    - the shared tiles must be PADDED, [BSXY][BSXY + 1], against bank
//      conflicts;
//    - N is any positive integer: the tiles of the last row and column of
//      blocks stick out of the matrices, and so does the last step k0.
//
//  Every element of A and B is then read from global memory N / BSXY times
//  instead of N times.
// =============================================================================
__global__ void matmul_shmem(int n, const float *A, const float *B, float *C)
{
  // TODO
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
  dim3 grid2(n, (n + BS_1D - 1) / BS_1D), block2(BS_1D);
  matmul_1d_row<<<grid2, block2>>>(n, dA, dB, dC);   // warm-up
  CUDA_CHECK_KERNEL();
  t0 = now();
  matmul_1d_row<<<grid2, block2>>>(n, dA, dB, dC);   // timed
  CUDA_CHECK_KERNEL();
  check("2 1D blocks, row of C", n, now() - t0, A, B, C, dC);

  // ---- 3. 1D blocks along a column of C -----------------------------------
  dim3 grid3((n + BS_1D - 1) / BS_1D, n), block3(BS_1D);
  matmul_1d_column<<<grid3, block3>>>(n, dA, dB, dC);   // warm-up
  CUDA_CHECK_KERNEL();
  t0 = now();
  matmul_1d_column<<<grid3, block3>>>(n, dA, dB, dC);   // timed
  CUDA_CHECK_KERNEL();
  check("3 1D blocks, column of C", n, now() - t0, A, B, C, dC);

  // ---- 4. 2D blocks, warps horizontal -------------------------------------
  dim3 grid4((n + BSXY - 1) / BSXY, (n + BSXY - 1) / BSXY), block4(BSXY, BSXY);
  matmul_2d_horizontal<<<grid4, block4>>>(n, dA, dB, dC);   // warm-up
  CUDA_CHECK_KERNEL();
  t0 = now();
  matmul_2d_horizontal<<<grid4, block4>>>(n, dA, dB, dC);   // timed
  CUDA_CHECK_KERNEL();
  check("4 2D blocks, warps horizontal", n, now() - t0, A, B, C, dC);

  // ---- 5. 2D blocks, warps vertical ---------------------------------------
  dim3 grid5((n + BSXY - 1) / BSXY, (n + BSXY - 1) / BSXY), block5(BSXY, BSXY);
  matmul_2d_vertical<<<grid5, block5>>>(n, dA, dB, dC);   // warm-up
  CUDA_CHECK_KERNEL();
  t0 = now();
  matmul_2d_vertical<<<grid5, block5>>>(n, dA, dB, dC);   // timed
  CUDA_CHECK_KERNEL();
  check("5 2D blocks, warps vertical", n, now() - t0, A, B, C, dC);

  // ---- cuBLAS, the benchmark ----------------------------------------------
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  cublasMatmul(handle, n, dA, dB, dC);     // warm-up: cuBLAS sets itself up on its first call
  CUDA_CHECK(cudaDeviceSynchronize());
  t0 = now();
  cublasMatmul(handle, n, dA, dB, dC);
  CUDA_CHECK(cudaDeviceSynchronize());
  check("cuBLAS (benchmark)", n, now() - t0, A, B, C, dC);
  CUBLAS_CHECK(cublasDestroy(handle));

  // ---- 6. Shared memory tiles ---------------------------------------------
  // TODO: grid and block of kernel 6 (as given: 1 block of 1 thread)
  dim3 grid6(1), block6(1);
  matmul_shmem<<<grid6, block6>>>(n, dA, dB, dC);   // warm-up
  CUDA_CHECK_KERNEL();
  t0 = now();
  matmul_shmem<<<grid6, block6>>>(n, dA, dB, dC);   // timed
  CUDA_CHECK_KERNEL();
  check("6 shared memory tiles", n, now() - t0, A, B, C, dC);

  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
  free(A); free(B); free(C);
  return 0;
}
