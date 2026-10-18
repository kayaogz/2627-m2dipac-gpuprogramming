// =============================================================================
//  TP2 -- saxpy on an N x N matrix: the same computation, eighteen mappings,
//         two memory layouts, and one question: which accesses are coalesced?
//
//  Y <- a * X + Y, where X and Y are N x N single-precision matrices.
//
//  The matrices are allocated as ONE-DIMENSIONAL arrays of N*N floats
//  (malloc on the CPU, cudaMalloc on the GPU). How the pair (i, j) of a
//  matrix element maps to a position in that array is OUR choice:
//
//      row-major    : element (i, j) is at  A[i * N + j]   (C convention)
//      column-major : element (i, j) is at  A[i + j * N]   (Fortran/BLAS)
//
//  Every mapping below comes as TWO kernels, one per layout, that differ in
//  exactly one line: the position idx of element (i, j) in the array. The
//  mappings are the fourteen of session 2, plus four where each thread handles
//  K x K elements; the two kernels of a mapping differ by their memory access
//  pattern alone.
//
//  For each kernel, the comment block above it answers, for both layouts:
//    * what a thread computes, and how (i, j) is obtained from the indices;
//    * what the 32 threads of a warp (32 consecutive threadIdx.x, same
//      threadIdx.y) touch in memory at one instruction;
//    * whether that access is COALESCED: the 32 addresses fall into the
//      minimum number of 32-byte sectors (4 sectors for 32 floats), so
//      that every fetched byte is used -- or NOT: up to 32 sectors, 1/8 of
//      the fetched bytes used.
//
//  The driver runs every kernel once, in both layouts, in TRACE mode: besides
//  computing, each thread records which (blockIdx.x, blockIdx.y, threadIdx.x,
//  threadIdx.y) touched each element -- the CHECK_INDEX() line after every
//  assignment to Y[idx]. The CPU then checks the result Y = a*X + Y, compares
//  the record with the mapping the kernel is supposed to implement (exact
//  check, element by element), and counts the 32-byte sectors touched by
//  each warp instruction -- the coalescing verdict, measured rather than
//  declared.
//
//  Build:  nvcc -O2 -arch=native saxpy2d.cu -o saxpy2d
//  Run:    ./saxpy2d
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
//  Error checking (wrap every CUDA runtime call and every kernel launch)
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } } while (0)

// A kernel launch returns nothing: check the launch itself, then the run.
#define CUDA_CHECK_KERNEL() do { \
  CUDA_CHECK(cudaGetLastError()); \
  CUDA_CHECK(cudaDeviceSynchronize()); } while (0)

// ----------------------------------------------------------------------------
//  Problem sizes and launch parameters
// ----------------------------------------------------------------------------
#define N 256                     // matrix size (small: the mapping does not depend on N, and the code must run on godbolt)
#define BLOCK_SIZE 128            // threads per block of the 1D-block kernels (N / BLOCK_SIZE = 2 blocks per row)
#define SIDE 32                   // tile side: the tile kernels use SIDE * SIDE = 1024 threads per block
#define WARP 32                   // threads per warp
#define KK 4                      // kernels 15-18: each thread computes a KK x KK patch

// ----------------------------------------------------------------------------
//  Layouts, initial values, trace
// ----------------------------------------------------------------------------
enum Layout { ROW_MAJOR = 0, COL_MAJOR = 1 };

// The matrices are initialised from their (i, j) coordinates, so that the
// expected result of Y <- a*X + Y can be recomputed on the CPU without a copy.
inline float xval(int i, int j) { return (float)((i + j) % 7); }
inline float yval(int i, int j) { return (float)((i * 3 + j) % 5); }

// Trace mode. When tr.count is non-NULL, each thread records, for the
// element idx it has just updated, its four indices exactly as CUDA gives
// them -- blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y (the .y ones are 0
// in a 1D launch, whose extent in y is 1) -- and increments the element's
// counter. Every kernel computes the position of the element it updates in a
// variable named idx, and calls CHECK_INDEX() right after the assignment to
// Y[idx]. (With tr.count == NULL the line costs one uniform branch, so the
// same kernels can be timed as they are.)
struct Trace { int *bx; int *by; int *tx; int *ty; int *count; };

#define CHECK_INDEX() do { if (tr.count != NULL) { \
    tr.bx[idx] = blockIdx.x;  tr.by[idx] = blockIdx.y; \
    tr.tx[idx] = threadIdx.x; tr.ty[idx] = threadIdx.y; \
    atomicAdd(&tr.count[idx], 1); } } while (0)

// =============================================================================
//  0. Reference: plain 1D saxpy over the N*N elements
// =============================================================================
//  One thread per element of the 1D array, in array order: thread k handles
//  position k, whatever (i, j) it corresponds to. The layout is irrelevant.
//
//  A warp = 32 consecutive k = 32 consecutive floats = 128 contiguous bytes,
//  one 128-byte line = 4 sectors, all bytes used.
//    row-major    : COALESCED
//    column-major : COALESCED  (the kernel never looks at i or j)
//  The reference: coalesced by construction, whatever the layout.
// =============================================================================
__global__ void saxpy_flat(int n, float a, const float *X, float *Y, Trace tr)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * n) {
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

// =============================================================================
//  1. One thread per ROW: thread i loops over j = 0 .. N-1
// =============================================================================
//  N threads in total (far too few to fill a GPU).
//  i = global 1D index; the thread walks its own row.
//
//  At one step of the loop (a fixed j), the 32 threads of a warp have 32
//  consecutive i and the SAME j, so they touch A(i..i+31, j):
//    row-major    : addresses i*N + j, (i+1)*N + j, ... : 4N bytes apart,
//                   32 different sectors for 128 useful bytes -> NOT coalesced
//    column-major : addresses i + j*N, i+1 + j*N, ...   : consecutive floats,
//                   4 sectors                            -> COALESCED
//  Even when coalesced, this kernel is slow: N threads in all (256 here, a
//  few thousand for a big matrix) cannot hide the memory latency of a GPU.
// =============================================================================
__global__ void saxpy_row_per_thread_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    for (int j = 0; j < n; j++) {
      int idx = i * n + j;   // row-major
      Y[idx] = a * X[idx] + Y[idx];
      CHECK_INDEX();
    }
  }
}

__global__ void saxpy_row_per_thread_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    for (int j = 0; j < n; j++) {
      int idx = i + j * n;   // column-major
      Y[idx] = a * X[idx] + Y[idx];
      CHECK_INDEX();
    }
  }
}

// =============================================================================
//  2. One thread per COLUMN: thread j loops over i = 0 .. N-1
// =============================================================================
//  The mirror image of kernel 1: a warp = 32 consecutive j, same i, touching
//  A(i, j..j+31) at each step.
//    row-major    : i*N + j .. i*N + j+31 : consecutive floats -> COALESCED
//    column-major : 4N bytes apart, 32 sectors                -> NOT coalesced
// =============================================================================
__global__ void saxpy_col_per_thread_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) {
    for (int i = 0; i < n; i++) {
      int idx = i * n + j;   // row-major
      Y[idx] = a * X[idx] + Y[idx];
      CHECK_INDEX();
    }
  }
}

__global__ void saxpy_col_per_thread_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) {
    for (int i = 0; i < n; i++) {
      int idx = i + j * n;   // column-major
      Y[idx] = a * X[idx] + Y[idx];
      CHECK_INDEX();
    }
  }
}

// =============================================================================
//  3. (a) 1D grid, 1D blocks, threads ROW aligned -- 1D indices
// =============================================================================
//  A block handles BLOCK_SIZE consecutive elements of a row. Blocks are
//  numbered row by row: blocksPerRow = ceil(N / BLOCK_SIZE) blocks per row.
//    b_i = blockIdx.x / blocksPerRow   (which row)
//    b_j = blockIdx.x % blocksPerRow   (which piece of it)
//    i = b_i,   j = b_j * blockDim.x + threadIdx.x     (guard: j < n)
//
//  A warp = 32 consecutive threadIdx.x = 32 consecutive j, same i:
//    row-major    : consecutive floats -> COALESCED
//    column-major : 4N bytes apart     -> NOT coalesced
// =============================================================================
__global__ void saxpy_1d_a_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int blocksPerRow = (n + blockDim.x - 1) / blockDim.x;
  int bi = blockIdx.x / blocksPerRow;
  int bj = blockIdx.x % blocksPerRow;
  int i = bi;
  int j = bj * blockDim.x + threadIdx.x;
  if (j < n) {
    int idx = i * n + j;   // row-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

__global__ void saxpy_1d_a_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int blocksPerRow = (n + blockDim.x - 1) / blockDim.x;
  int bi = blockIdx.x / blocksPerRow;
  int bj = blockIdx.x % blocksPerRow;
  int i = bi;
  int j = bj * blockDim.x + threadIdx.x;
  if (j < n) {
    int idx = i + j * n;   // column-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

// =============================================================================
//  4. (b) 1D grid, 1D blocks, threads COLUMN aligned -- 1D indices
// =============================================================================
//  A block handles BLOCK_SIZE consecutive elements of a column; blocks are
//  numbered column by column (blocksPerCol pieces per column).
//    b_j = blockIdx.x / blocksPerCol,  b_i = blockIdx.x % blocksPerCol
//    i = b_i * blockDim.x + threadIdx.x,   j = b_j     (guard: i < n)
//
//  A warp = 32 consecutive i, same j:
//    row-major    : NOT coalesced (32 sectors)
//    column-major : COALESCED
// =============================================================================
__global__ void saxpy_1d_b_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int blocksPerCol = (n + blockDim.x - 1) / blockDim.x;
  int bj = blockIdx.x / blocksPerCol;
  int bi = blockIdx.x % blocksPerCol;
  int i = bi * blockDim.x + threadIdx.x;
  int j = bj;
  if (i < n) {
    int idx = i * n + j;   // row-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

__global__ void saxpy_1d_b_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int blocksPerCol = (n + blockDim.x - 1) / blockDim.x;
  int bj = blockIdx.x / blocksPerCol;
  int bi = blockIdx.x % blocksPerCol;
  int i = bi * blockDim.x + threadIdx.x;
  int j = bj;
  if (i < n) {
    int idx = i + j * n;   // column-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

// =============================================================================
//  5. (a) 2D grid, 1D blocks, threads ROW aligned -- dim3 indices
// =============================================================================
//  Same mapping as kernel 3; blockIdx.x and blockIdx.y simply replace b_i
//  and b_j -- no division/modulus. Grid: N x ceil(N / BLOCK_SIZE).
//  Same warp behaviour as kernel 3:
//    row-major : COALESCED       column-major : NOT coalesced
// =============================================================================
__global__ void saxpy_2d_a_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int i = blockIdx.x;
  int j = blockIdx.y * blockDim.x + threadIdx.x;
  if (j < n) {
    int idx = i * n + j;   // row-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

__global__ void saxpy_2d_a_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int i = blockIdx.x;
  int j = blockIdx.y * blockDim.x + threadIdx.x;
  if (j < n) {
    int idx = i + j * n;   // column-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

// =============================================================================
//  6. (b) 2D grid, 1D blocks, threads COLUMN aligned -- dim3 indices
// =============================================================================
//  Same mapping as kernel 4 with b_i = blockIdx.x (which piece), b_j =
//  blockIdx.y (which column). Grid: ceil(N / BLOCK_SIZE) x N.
//    row-major : NOT coalesced   column-major : COALESCED
// =============================================================================
__global__ void saxpy_2d_b_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y;
  if (i < n) {
    int idx = i * n + j;   // row-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

__global__ void saxpy_2d_b_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y;
  if (i < n) {
    int idx = i + j * n;   // column-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

// =============================================================================
//  7. (c) 1D grid, tile blocks, threads COLUMN aligned -- 1D indices
// =============================================================================
//  A block handles a SIDE x SIDE tile. Tiles are numbered row by row
//  (tilesPerRow = ceil(N / SIDE) tiles per row of tiles):
//    b_i = blockIdx.x / tilesPerRow,   b_j = blockIdx.x % tilesPerRow
//  Inside the tile, threads are column aligned: consecutive threadIdx.x go
//  down a column of the tile:
//    t_i = threadIdx.x % SIDE,   t_j = threadIdx.x / SIDE
//    i = b_i * SIDE + t_i,       j = b_j * SIDE + t_j    (guard on both)
//
//  A warp (32 consecutive threadIdx.x, SIDE = 32) = one COLUMN of the tile
//  = 32 consecutive i, same j:
//    row-major    : NOT coalesced (32 sectors)
//    column-major : COALESCED
// =============================================================================
__global__ void saxpy_1d_c_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int tilesPerRow = (n + SIDE - 1) / SIDE;
  int bi = blockIdx.x / tilesPerRow;
  int bj = blockIdx.x % tilesPerRow;
  int ti = threadIdx.x % SIDE;
  int tj = threadIdx.x / SIDE;
  int i = bi * SIDE + ti;
  int j = bj * SIDE + tj;
  if (i < n && j < n) {
    int idx = i * n + j;   // row-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

__global__ void saxpy_1d_c_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int tilesPerRow = (n + SIDE - 1) / SIDE;
  int bi = blockIdx.x / tilesPerRow;
  int bj = blockIdx.x % tilesPerRow;
  int ti = threadIdx.x % SIDE;
  int tj = threadIdx.x / SIDE;
  int i = bi * SIDE + ti;
  int j = bj * SIDE + tj;
  if (i < n && j < n) {
    int idx = i + j * n;   // column-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

// =============================================================================
//  8. (d) 1D grid, tile blocks, threads ROW aligned -- 1D indices
// =============================================================================
//  Same tiles as kernel 7; threads are row aligned, consecutive threadIdx.x
//  go along a row of the tile:
//    t_i = threadIdx.x / SIDE,   t_j = threadIdx.x % SIDE
//  A warp = one ROW of the tile = 32 consecutive j, same i:
//    row-major    : COALESCED
//    column-major : NOT coalesced
// =============================================================================
__global__ void saxpy_1d_d_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int tilesPerRow = (n + SIDE - 1) / SIDE;
  int bi = blockIdx.x / tilesPerRow;
  int bj = blockIdx.x % tilesPerRow;
  int ti = threadIdx.x / SIDE;
  int tj = threadIdx.x % SIDE;
  int i = bi * SIDE + ti;
  int j = bj * SIDE + tj;
  if (i < n && j < n) {
    int idx = i * n + j;   // row-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

__global__ void saxpy_1d_d_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int tilesPerRow = (n + SIDE - 1) / SIDE;
  int bi = blockIdx.x / tilesPerRow;
  int bj = blockIdx.x % tilesPerRow;
  int ti = threadIdx.x / SIDE;
  int tj = threadIdx.x % SIDE;
  int i = bi * SIDE + ti;
  int j = bj * SIDE + tj;
  if (i < n && j < n) {
    int idx = i + j * n;   // column-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

// =============================================================================
//  9. (c) alternative: tiles numbered COLUMN by column, threads column aligned
// =============================================================================
//  Only the numbering of the blocks changes (consecutive blocks walk down a
//  column of tiles): b_j = blockIdx.x / tilesPerCol, b_i = blockIdx.x %
//  tilesPerCol. What a block computes and how its threads are aligned is
//  exactly as in kernel 7, so what a warp touches is unchanged:
//    row-major : NOT coalesced   column-major : COALESCED
//  The block numbering is a logical convenience; it has no effect on the
//  memory accesses of a warp.
// =============================================================================
__global__ void saxpy_1d_c_alt_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int tilesPerCol = (n + SIDE - 1) / SIDE;
  int bj = blockIdx.x / tilesPerCol;
  int bi = blockIdx.x % tilesPerCol;
  int ti = threadIdx.x % SIDE;
  int tj = threadIdx.x / SIDE;
  int i = bi * SIDE + ti;
  int j = bj * SIDE + tj;
  if (i < n && j < n) {
    int idx = i * n + j;   // row-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

__global__ void saxpy_1d_c_alt_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int tilesPerCol = (n + SIDE - 1) / SIDE;
  int bj = blockIdx.x / tilesPerCol;
  int bi = blockIdx.x % tilesPerCol;
  int ti = threadIdx.x % SIDE;
  int tj = threadIdx.x / SIDE;
  int i = bi * SIDE + ti;
  int j = bj * SIDE + tj;
  if (i < n && j < n) {
    int idx = i + j * n;   // column-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

// =============================================================================
// 10. (d) alternative: tiles numbered column by column, threads ROW aligned
// =============================================================================
//  Kernel 8 with the block numbering of kernel 9. A warp = one row of the
//  tile:
//    row-major : COALESCED       column-major : NOT coalesced
// =============================================================================
__global__ void saxpy_1d_d_alt_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int tilesPerCol = (n + SIDE - 1) / SIDE;
  int bj = blockIdx.x / tilesPerCol;
  int bi = blockIdx.x % tilesPerCol;
  int ti = threadIdx.x / SIDE;
  int tj = threadIdx.x % SIDE;
  int i = bi * SIDE + ti;
  int j = bj * SIDE + tj;
  if (i < n && j < n) {
    int idx = i * n + j;   // row-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

__global__ void saxpy_1d_d_alt_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int tilesPerCol = (n + SIDE - 1) / SIDE;
  int bj = blockIdx.x / tilesPerCol;
  int bi = blockIdx.x % tilesPerCol;
  int ti = threadIdx.x / SIDE;
  int tj = threadIdx.x % SIDE;
  int i = bi * SIDE + ti;
  int j = bj * SIDE + tj;
  if (i < n && j < n) {
    int idx = i + j * n;   // column-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

// =============================================================================
// 11. (c) 2D grid, 2D blocks, threads COLUMN aligned -- dim3 indices
// =============================================================================
//  dimBlock = (SIDE, SIDE): the thread coordinates are read directly,
//    t_i = threadIdx.x,  t_j = threadIdx.y,  b_i = blockIdx.x,  b_j = blockIdx.y
//    i = b_i * SIDE + t_i,   j = b_j * SIDE + t_j     (SIDE = blockDim.x = blockDim.y)
//  A warp = 32 consecutive threadIdx.x, same threadIdx.y = one column of
//  the tile = 32 consecutive i, same j:
//    row-major : NOT coalesced   column-major : COALESCED
// =============================================================================
__global__ void saxpy_2d_c_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int i = blockIdx.x * SIDE + threadIdx.x;
  int j = blockIdx.y * SIDE + threadIdx.y;
  if (i < n && j < n) {
    int idx = i * n + j;   // row-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

__global__ void saxpy_2d_c_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int i = blockIdx.x * SIDE + threadIdx.x;
  int j = blockIdx.y * SIDE + threadIdx.y;
  if (i < n && j < n) {
    int idx = i + j * n;   // column-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

// =============================================================================
// 12. (d) 2D grid, 2D blocks, threads ROW aligned -- dim3 indices
// =============================================================================
//  Same blocks and grid; the thread coordinates are interchanged,
//    t_i = threadIdx.y,  t_j = threadIdx.x
//    i = b_i * SIDE + t_i,   j = b_j * SIDE + t_j
//  A warp = one row of the tile = 32 consecutive j, same i:
//    row-major : COALESCED       column-major : NOT coalesced
// =============================================================================
__global__ void saxpy_2d_d_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int i = blockIdx.x * SIDE + threadIdx.y;
  int j = blockIdx.y * SIDE + threadIdx.x;
  if (i < n && j < n) {
    int idx = i * n + j;   // row-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

__global__ void saxpy_2d_d_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int i = blockIdx.x * SIDE + threadIdx.y;
  int j = blockIdx.y * SIDE + threadIdx.x;
  if (i < n && j < n) {
    int idx = i + j * n;   // column-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

// =============================================================================
// 13. (c) alternative with dim3: block indices interchanged, threads column aligned
// =============================================================================
//  b_i = blockIdx.y, b_j = blockIdx.x (the grid is transposed); threads as in
//  kernel 11. A warp = one column of the tile:
//    row-major : NOT coalesced   column-major : COALESCED
// =============================================================================
__global__ void saxpy_2d_c_alt_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int i = blockIdx.y * SIDE + threadIdx.x;
  int j = blockIdx.x * SIDE + threadIdx.y;
  if (i < n && j < n) {
    int idx = i * n + j;   // row-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

__global__ void saxpy_2d_c_alt_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int i = blockIdx.y * SIDE + threadIdx.x;
  int j = blockIdx.x * SIDE + threadIdx.y;
  if (i < n && j < n) {
    int idx = i + j * n;   // column-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

// =============================================================================
// 14. (d) alternative with dim3: block indices interchanged, threads row aligned
// =============================================================================
//  b_i = blockIdx.y, b_j = blockIdx.x; threads as in kernel 12. A warp = one
//  row of the tile:
//    row-major : COALESCED       column-major : NOT coalesced
// =============================================================================
__global__ void saxpy_2d_d_alt_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int i = blockIdx.y * SIDE + threadIdx.y;
  int j = blockIdx.x * SIDE + threadIdx.x;
  if (i < n && j < n) {
    int idx = i * n + j;   // row-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

__global__ void saxpy_2d_d_alt_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  int i = blockIdx.y * SIDE + threadIdx.y;
  int j = blockIdx.x * SIDE + threadIdx.x;
  if (i < n && j < n) {
    int idx = i + j * n;   // column-major
    Y[idx] = a * X[idx] + Y[idx];
    CHECK_INDEX();
  }
}

// =============================================================================
// 15. (c) 2D blocks, threads column aligned, K x K elements per thread
// =============================================================================
//  Same thread and block coordinates as kernel 11, but each thread now
//  computes K x K elements instead of one, INTERLEAVED with the other threads
//  of its block: a block covers a (32K) x (32K) tile, and thread (t_i, t_j)
//  handles the elements
//      i = b_i*32K + t_i + 32*di,   j = b_j*32K + t_j + 32*dj,   di, dj < K
//  i.e. at every step (di, dj) the 32 threads of a warp are still 32
//  CONSECUTIVE i, same j, exactly as in kernel 11. The grid is
//  ceil(N / 32K) x ceil(N / 32K).
//
//  Because the threads stay consecutive, the K x K variant keeps the
//  coalescing behaviour of kernel 11:
//    row-major    : NOT coalesced (32 sectors)
//    column-major : COALESCED
//  (Giving each thread a CONTIGUOUS K x K patch instead -- i0 = (b_i*32 +
//  t_i)*K -- would put consecutive threads K floats apart and destroy the
//  coalescing: 4K sectors per warp instruction instead of 4.)
// =============================================================================
__global__ void saxpy_2d_c_kxk_rowmajor(int n, int K, float a, const float *X, float *Y, Trace tr)
{
  int i0 = blockIdx.x * SIDE * K + threadIdx.x;
  int j0 = blockIdx.y * SIDE * K + threadIdx.y;
  for (int di = 0; di < K; di++) {
    for (int dj = 0; dj < K; dj++) {
      int i = i0 + di * SIDE;
      int j = j0 + dj * SIDE;
      if (i < n && j < n) {
        int idx = i * n + j;   // row-major
        Y[idx] = a * X[idx] + Y[idx];
        CHECK_INDEX();
      }
    }
  }
}

__global__ void saxpy_2d_c_kxk_colmajor(int n, int K, float a, const float *X, float *Y, Trace tr)
{
  int i0 = blockIdx.x * SIDE * K + threadIdx.x;
  int j0 = blockIdx.y * SIDE * K + threadIdx.y;
  for (int di = 0; di < K; di++) {
    for (int dj = 0; dj < K; dj++) {
      int i = i0 + di * SIDE;
      int j = j0 + dj * SIDE;
      if (i < n && j < n) {
        int idx = i + j * n;   // column-major
        Y[idx] = a * X[idx] + Y[idx];
        CHECK_INDEX();
      }
    }
  }
}

// =============================================================================
// 16. (d) 2D blocks, threads row aligned, K x K elements per thread
// =============================================================================
//  Same thread and block coordinates as kernel 12, but each thread now
//  computes K x K elements instead of one, INTERLEAVED with the other threads
//  of its block: a block covers a (32K) x (32K) tile, and thread (t_i, t_j)
//  handles the elements
//      i = b_i*32K + t_i + 32*di,   j = b_j*32K + t_j + 32*dj,   di, dj < K
//  i.e. at every step (di, dj) the 32 threads of a warp are still 32
//  CONSECUTIVE j, same i, exactly as in kernel 12. The grid is
//  ceil(N / 32K) x ceil(N / 32K).
//
//  Because the threads stay consecutive, the K x K variant keeps the
//  coalescing behaviour of kernel 12:
//    row-major    : COALESCED
//    column-major : NOT coalesced (32 sectors)
//  (Giving each thread a CONTIGUOUS K x K patch instead -- i0 = (b_i*32 +
//  t_i)*K -- would put consecutive threads K floats apart and destroy the
//  coalescing: 4K sectors per warp instruction instead of 4.)
// =============================================================================
__global__ void saxpy_2d_d_kxk_rowmajor(int n, int K, float a, const float *X, float *Y, Trace tr)
{
  int i0 = blockIdx.x * SIDE * K + threadIdx.y;
  int j0 = blockIdx.y * SIDE * K + threadIdx.x;
  for (int di = 0; di < K; di++) {
    for (int dj = 0; dj < K; dj++) {
      int i = i0 + di * SIDE;
      int j = j0 + dj * SIDE;
      if (i < n && j < n) {
        int idx = i * n + j;   // row-major
        Y[idx] = a * X[idx] + Y[idx];
        CHECK_INDEX();
      }
    }
  }
}

__global__ void saxpy_2d_d_kxk_colmajor(int n, int K, float a, const float *X, float *Y, Trace tr)
{
  int i0 = blockIdx.x * SIDE * K + threadIdx.y;
  int j0 = blockIdx.y * SIDE * K + threadIdx.x;
  for (int di = 0; di < K; di++) {
    for (int dj = 0; dj < K; dj++) {
      int i = i0 + di * SIDE;
      int j = j0 + dj * SIDE;
      if (i < n && j < n) {
        int idx = i + j * n;   // column-major
        Y[idx] = a * X[idx] + Y[idx];
        CHECK_INDEX();
      }
    }
  }
}

// =============================================================================
// 17. (c) alternative (grid transposed), K x K elements per thread
// =============================================================================
//  Same thread and block coordinates as kernel 13, but each thread now
//  computes K x K elements instead of one, INTERLEAVED with the other threads
//  of its block: a block covers a (32K) x (32K) tile, and thread (t_i, t_j)
//  handles the elements
//      i = b_i*32K + t_i + 32*di,   j = b_j*32K + t_j + 32*dj,   di, dj < K
//  i.e. at every step (di, dj) the 32 threads of a warp are still 32
//  CONSECUTIVE i, same j, exactly as in kernel 13. The grid is
//  ceil(N / 32K) x ceil(N / 32K).
//
//  Because the threads stay consecutive, the K x K variant keeps the
//  coalescing behaviour of kernel 13:
//    row-major    : NOT coalesced (32 sectors)
//    column-major : COALESCED
//  (Giving each thread a CONTIGUOUS K x K patch instead -- i0 = (b_i*32 +
//  t_i)*K -- would put consecutive threads K floats apart and destroy the
//  coalescing: 4K sectors per warp instruction instead of 4.)
// =============================================================================
__global__ void saxpy_2d_c_alt_kxk_rowmajor(int n, int K, float a, const float *X, float *Y, Trace tr)
{
  int i0 = blockIdx.y * SIDE * K + threadIdx.x;
  int j0 = blockIdx.x * SIDE * K + threadIdx.y;
  for (int di = 0; di < K; di++) {
    for (int dj = 0; dj < K; dj++) {
      int i = i0 + di * SIDE;
      int j = j0 + dj * SIDE;
      if (i < n && j < n) {
        int idx = i * n + j;   // row-major
        Y[idx] = a * X[idx] + Y[idx];
        CHECK_INDEX();
      }
    }
  }
}

__global__ void saxpy_2d_c_alt_kxk_colmajor(int n, int K, float a, const float *X, float *Y, Trace tr)
{
  int i0 = blockIdx.y * SIDE * K + threadIdx.x;
  int j0 = blockIdx.x * SIDE * K + threadIdx.y;
  for (int di = 0; di < K; di++) {
    for (int dj = 0; dj < K; dj++) {
      int i = i0 + di * SIDE;
      int j = j0 + dj * SIDE;
      if (i < n && j < n) {
        int idx = i + j * n;   // column-major
        Y[idx] = a * X[idx] + Y[idx];
        CHECK_INDEX();
      }
    }
  }
}

// =============================================================================
// 18. (d) alternative (grid transposed), K x K elements per thread
// =============================================================================
//  Same thread and block coordinates as kernel 14, but each thread now
//  computes K x K elements instead of one, INTERLEAVED with the other threads
//  of its block: a block covers a (32K) x (32K) tile, and thread (t_i, t_j)
//  handles the elements
//      i = b_i*32K + t_i + 32*di,   j = b_j*32K + t_j + 32*dj,   di, dj < K
//  i.e. at every step (di, dj) the 32 threads of a warp are still 32
//  CONSECUTIVE j, same i, exactly as in kernel 14. The grid is
//  ceil(N / 32K) x ceil(N / 32K).
//
//  Because the threads stay consecutive, the K x K variant keeps the
//  coalescing behaviour of kernel 14:
//    row-major    : COALESCED
//    column-major : NOT coalesced (32 sectors)
//  (Giving each thread a CONTIGUOUS K x K patch instead -- i0 = (b_i*32 +
//  t_i)*K -- would put consecutive threads K floats apart and destroy the
//  coalescing: 4K sectors per warp instruction instead of 4.)
// =============================================================================
__global__ void saxpy_2d_d_alt_kxk_rowmajor(int n, int K, float a, const float *X, float *Y, Trace tr)
{
  int i0 = blockIdx.y * SIDE * K + threadIdx.y;
  int j0 = blockIdx.x * SIDE * K + threadIdx.x;
  for (int di = 0; di < K; di++) {
    for (int dj = 0; dj < K; dj++) {
      int i = i0 + di * SIDE;
      int j = j0 + dj * SIDE;
      if (i < n && j < n) {
        int idx = i * n + j;   // row-major
        Y[idx] = a * X[idx] + Y[idx];
        CHECK_INDEX();
      }
    }
  }
}

__global__ void saxpy_2d_d_alt_kxk_colmajor(int n, int K, float a, const float *X, float *Y, Trace tr)
{
  int i0 = blockIdx.y * SIDE * K + threadIdx.y;
  int j0 = blockIdx.x * SIDE * K + threadIdx.x;
  for (int di = 0; di < K; di++) {
    for (int dj = 0; dj < K; dj++) {
      int i = i0 + di * SIDE;
      int j = j0 + dj * SIDE;
      if (i < n && j < n) {
        int idx = i + j * n;   // column-major
        Y[idx] = a * X[idx] + Y[idx];
        CHECK_INDEX();
      }
    }
  }
}

// =============================================================================
//  Kernel table: names, launch configurations, expected mappings
// =============================================================================

// Launch configuration of kernel `kernel` for a matrix of size n.
struct Config { dim3 grid; dim3 block; };

Config config(int kernel, int n)
{
  const int nbLinear = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;   // pieces per row/column
  const int nbTiles = (n + SIDE - 1) / SIDE;                 // tiles per row/column
  Config c;
  c.block = dim3(BLOCK_SIZE);                                // 1D block of BLOCK_SIZE threads, unless said otherwise
  switch (kernel) {
    case 0:  c.grid = dim3((n * n + BLOCK_SIZE - 1) / BLOCK_SIZE); break;
    case 1:  case 2: c.grid = dim3(nbLinear); break;
    case 3:  case 4: c.grid = dim3(n * nbLinear); break;
    case 5:  c.grid = dim3(n, nbLinear); break;
    case 6:  c.grid = dim3(nbLinear, n); break;
    case 7:  case 8: case 9: case 10: c.grid = dim3(nbTiles * nbTiles); c.block = dim3(SIDE * SIDE); break;
    case 11: case 12: case 13: case 14: c.grid = dim3(nbTiles, nbTiles); c.block = dim3(SIDE, SIDE); break;
    default: {                                                 // 15-18: K x K per thread
      int nbBig = (n + SIDE * KK - 1) / (SIDE * KK);
      c.grid = dim3(nbBig, nbBig); c.block = dim3(SIDE, SIDE); break;
    }
  }
  return c;
}

// Launch mapping `kernel` with the given layout on an n x n matrix.
#define LAUNCH(fn) do { if (layout == ROW_MAJOR) fn##_rowmajor<<<c.grid, c.block>>>(n, a, dX, dY, tr); \
                        else                     fn##_colmajor<<<c.grid, c.block>>>(n, a, dX, dY, tr); } while (0)
#define LAUNCH_K(fn) do { if (layout == ROW_MAJOR) fn##_rowmajor<<<c.grid, c.block>>>(n, KK, a, dX, dY, tr); \
                          else                     fn##_colmajor<<<c.grid, c.block>>>(n, KK, a, dX, dY, tr); } while (0)
void launch(int kernel, int n, float a, const float *dX, float *dY, int layout, Trace tr)
{
  Config c = config(kernel, n);
  switch (kernel) {
    case 0:  saxpy_flat<<<c.grid, c.block>>>(n, a, dX, dY, tr); break;   // layout-free
    case 1:  LAUNCH(saxpy_row_per_thread); break;
    case 2:  LAUNCH(saxpy_col_per_thread); break;
    case 3:  LAUNCH(saxpy_1d_a); break;
    case 4:  LAUNCH(saxpy_1d_b); break;
    case 5:  LAUNCH(saxpy_2d_a); break;
    case 6:  LAUNCH(saxpy_2d_b); break;
    case 7:  LAUNCH(saxpy_1d_c); break;
    case 8:  LAUNCH(saxpy_1d_d); break;
    case 9:  LAUNCH(saxpy_1d_c_alt); break;
    case 10: LAUNCH(saxpy_1d_d_alt); break;
    case 11: LAUNCH(saxpy_2d_c); break;
    case 12: LAUNCH(saxpy_2d_d); break;
    case 13: LAUNCH(saxpy_2d_c_alt); break;
    case 14: LAUNCH(saxpy_2d_d_alt); break;
    case 15: LAUNCH_K(saxpy_2d_c_kxk); break;
    case 16: LAUNCH_K(saxpy_2d_d_kxk); break;
    case 17: LAUNCH_K(saxpy_2d_c_alt_kxk); break;
    case 18: LAUNCH_K(saxpy_2d_d_alt_kxk); break;
  }
  CUDA_CHECK_KERNEL();
}

// The mapping each kernel is supposed to implement, written the other way
// round: which (blockIdx, threadIdx) must touch element (i, j)? This is the
// reference against which the trace is checked, element by element.
struct Who { int bx, by, tx, ty; };

Who expected(int kernel, int i, int j, int n)
{
  const int nbLinear = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int nbTiles = (n + SIDE - 1) / SIDE;
  Who w = { 0, 0, 0, 0 };
  int bi, bj, ti, tj;
  switch (kernel) {
    case 0: {                                   // flat: position k, whatever the layout
      // (i, j) -> k depends on the layout; handled by the caller (see checkTrace)
      break;
    }
    case 1:  w.bx = i / BLOCK_SIZE; w.tx = i % BLOCK_SIZE; break;   // thread i, all j
    case 2:  w.bx = j / BLOCK_SIZE; w.tx = j % BLOCK_SIZE; break;   // thread j, all i
    case 3:  w.bx = i * nbLinear + j / BLOCK_SIZE; w.tx = j % BLOCK_SIZE; break;
    case 4:  w.bx = j * nbLinear + i / BLOCK_SIZE; w.tx = i % BLOCK_SIZE; break;
    case 5:  w.bx = i; w.by = j / BLOCK_SIZE; w.tx = j % BLOCK_SIZE; break;
    case 6:  w.bx = i / BLOCK_SIZE; w.by = j; w.tx = i % BLOCK_SIZE; break;
    case 7:  bi = i / SIDE; bj = j / SIDE; ti = i % SIDE; tj = j % SIDE;
             w.bx = bi * nbTiles + bj; w.tx = tj * SIDE + ti; break;          // column aligned
    case 8:  bi = i / SIDE; bj = j / SIDE; ti = i % SIDE; tj = j % SIDE;
             w.bx = bi * nbTiles + bj; w.tx = ti * SIDE + tj; break;          // row aligned
    case 9:  bi = i / SIDE; bj = j / SIDE; ti = i % SIDE; tj = j % SIDE;
             w.bx = bj * nbTiles + bi; w.tx = tj * SIDE + ti; break;          // tiles column by column
    case 10: bi = i / SIDE; bj = j / SIDE; ti = i % SIDE; tj = j % SIDE;
             w.bx = bj * nbTiles + bi; w.tx = ti * SIDE + tj; break;
    case 11: w.bx = i / SIDE; w.by = j / SIDE; w.tx = i % SIDE; w.ty = j % SIDE; break;
    case 12: w.bx = i / SIDE; w.by = j / SIDE; w.tx = j % SIDE; w.ty = i % SIDE; break;
    case 13: w.bx = j / SIDE; w.by = i / SIDE; w.tx = i % SIDE; w.ty = j % SIDE; break;
    case 14: w.bx = j / SIDE; w.by = i / SIDE; w.tx = j % SIDE; w.ty = i % SIDE; break;
    // K x K per thread, interleaved: tiles of 32K; within the tile, thread coordinate = index % 32
    case 15: w.bx = i / (SIDE * KK); w.by = j / (SIDE * KK); w.tx = i % SIDE; w.ty = j % SIDE; break;
    case 16: w.bx = i / (SIDE * KK); w.by = j / (SIDE * KK); w.tx = j % SIDE; w.ty = i % SIDE; break;
    case 17: w.bx = j / (SIDE * KK); w.by = i / (SIDE * KK); w.tx = i % SIDE; w.ty = j % SIDE; break;
    case 18: w.bx = j / (SIDE * KK); w.by = i / (SIDE * KK); w.tx = j % SIDE; w.ty = i % SIDE; break;
  }
  return w;
}

// Names of the mappings, and the coalescing verdicts claimed in the comments above.
struct KernelInfo { const char *name; const char *rowMajor; const char *colMajor; };
static const KernelInfo kernels[] = {
  { "0  1D reference (flat)",                    "yes", "yes" },
  { "1  one thread per row",                      "no",  "yes" },
  { "2  one thread per column",                   "yes", "no"  },
  { "3  (a) 1D blocks, row aligned, 1D idx",      "yes", "no"  },
  { "4  (b) 1D blocks, col aligned, 1D idx",      "no",  "yes" },
  { "5  (a) 1D blocks, row aligned, dim3",        "yes", "no"  },
  { "6  (b) 1D blocks, col aligned, dim3",        "no",  "yes" },
  { "7  (c) tiles, col aligned, 1D idx",          "no",  "yes" },
  { "8  (d) tiles, row aligned, 1D idx",          "yes", "no"  },
  { "9  (c) alt: tiles col by col, col aligned",  "no",  "yes" },
  { "10 (d) alt: tiles col by col, row aligned",  "yes", "no"  },
  { "11 (c) 2D blocks, col aligned, dim3",        "no",  "yes" },
  { "12 (d) 2D blocks, row aligned, dim3",        "yes", "no"  },
  { "13 (c) alt: grid transposed, col aligned",   "no",  "yes" },
  { "14 (d) alt: grid transposed, row aligned",   "yes", "no"  },
  { "15 (c) 2D blocks, col aligned, KxK/thread",  "no",  "yes" },
  { "16 (d) 2D blocks, row aligned, KxK/thread",  "yes", "no"  },
  { "17 (c) alt, KxK/thread",                     "no",  "yes" },
  { "18 (d) alt, KxK/thread",                     "yes", "no"  },
};
static const int numKernels = sizeof(kernels) / sizeof(kernels[0]);

// =============================================================================
//  Trace analysis (CPU)
// =============================================================================
//  From the recorded (blockIdx, threadIdx) of every element:
//    * coverage: every element touched exactly once;
//    * mapping: the recorded (blockIdx.x, .y, threadIdx.x, .y) is the
//      expected one, element by element;
//    * sectors: group the elements by warp instruction -- same block, same
//      warp within the block (threads numbered x first, 32 per warp), and
//      for the two loop kernels the same loop step (j for one thread per
//      row, i for one thread per column) -- and count the distinct 32-byte
//      sectors (8 floats) touched; 4 means coalesced, 32 means one sector
//      per thread.
struct TraceResult { int coverageErrors; int mappingErrors; Who firstBad; int badI, badJ; Who firstGot; double sectorsPerWarp; int minSectors, maxSectors; };

static int cmpInt3(const void *a, const void *b)
{
  const int *x = (const int *)a, *y = (const int *)b;
  for (int t = 0; t < 3; t++) { if (x[t] != y[t]) return x[t] < y[t] ? -1 : 1; }
  return 0;
}

TraceResult checkTrace(int kernel, int n, int layout, const int *bx, const int *by, const int *tx, const int *ty, const int *count)
{
  TraceResult r; memset(&r, 0, sizeof(r)); r.minSectors = 1 << 30;
  Config c = config(kernel, n);
  const int n2 = n * n;
  // --- coverage and exact mapping
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      int k = (layout == ROW_MAJOR) ? i * n + j : i + j * n;
      if (count[k] != 1) { r.coverageErrors++; continue; }
      Who w = expected(kernel, i, j, n);
      if (kernel == 0) { w.bx = k / BLOCK_SIZE; w.tx = k % BLOCK_SIZE; }
      if (bx[k] != w.bx || by[k] != w.by || tx[k] != w.tx || ty[k] != w.ty) {
        if (r.mappingErrors == 0) {
          r.firstBad = w; r.badI = i; r.badJ = j;
          r.firstGot.bx = bx[k]; r.firstGot.by = by[k];
          r.firstGot.tx = tx[k]; r.firstGot.ty = ty[k];
        }
        r.mappingErrors++;
      }
    }
  }
  // --- sectors per warp instruction: sort the elements by (warp, step, k).
  // A warp is identified by its block and its rank within the block; threads
  // are numbered threadIdx.x first, then threadIdx.y, 32 per warp. The loop
  // step of an element is known from the kernel: j for one thread per row,
  // i for one thread per column, (di, dj) inside the patch for the K x K
  // kernels, 0 for the one-element-per-thread kernels.
  int *rec = (int *)malloc((size_t)n2 * 3 * sizeof(int));
  const int warpsPerBlock = (c.block.x * c.block.y + WARP - 1) / WARP;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      int k = (layout == ROW_MAJOR) ? i * n + j : i + j * n;
      int block = bx[k] + by[k] * c.grid.x;
      int thread = tx[k] + ty[k] * c.block.x;
      rec[3 * k] = block * warpsPerBlock + thread / WARP;   // global warp id
      rec[3 * k + 1] = (kernel == 1) ? j : (kernel == 2) ? i
                     : (kernel >= 15) ? ((i % (SIDE * KK)) / SIDE) * KK + (j % (SIDE * KK)) / SIDE : 0;   // K x K: step (di, dj)
      rec[3 * k + 2] = k;
    }
  }
  qsort(rec, n2, 3 * sizeof(int), cmpInt3);
  long long totalSectors = 0; int groups = 0;
  int g = 0;
  while (g < n2) {
    int h = g, sectors = 0, lastSector = -1;   // elements of a group are sorted by k
    while (h < n2 && rec[3 * h] == rec[3 * g] && rec[3 * h + 1] == rec[3 * g + 1]) {
      int sector = rec[3 * h + 2] / 8;          // 8 floats = 32 bytes per sector
      if (sector != lastSector) { sectors++; lastSector = sector; }
      h++;
    }
    if (h - g == WARP) {                        // full warps only (tails are shorter)
      totalSectors += sectors; groups++;
      if (sectors < r.minSectors) r.minSectors = sectors;
      if (sectors > r.maxSectors) r.maxSectors = sectors;
    }
    g = h;
  }
  free(rec);
  r.sectorsPerWarp = groups ? (double)totalSectors / groups : 0.0;
  if (groups == 0) { r.minSectors = 0; }          // nothing recorded (kernel not written yet)
  return r;
}

// =============================================================================
//  Initial values and result check (CPU side)
// =============================================================================
// Fill X and Y from the (i, j) coordinates, in the given layout.
void initialise(float *X, float *Y, int n, int layout)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      int k = (layout == ROW_MAJOR) ? i * n + j : i + j * n;
      X[k] = xval(i, j);
      Y[k] = yval(i, j);
    }
  }
}

// Check Y == a * X + Y0 for every element; return the number of errors.
int verify(const float *Y, float a, int n, int layout)
{
  int errors = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      int k = (layout == ROW_MAJOR) ? i * n + j : i + j * n;
      if (Y[k] != a * xval(i, j) + yval(i, j)) { errors++; }
    }
  }
  return errors;
}

int main(void)
{
  const float a = 2.0f;
  const int n = N, n2 = n * n;
  const size_t bytes = (size_t)n2 * sizeof(float);
  float *X = (float *)malloc(bytes);        // CPU arrays: ONE-dimensional, N*N floats
  float *Y = (float *)malloc(bytes);
  float *dX, *dY;                           // GPU arrays, same shape
  CUDA_CHECK(cudaMalloc(&dX, bytes));
  CUDA_CHECK(cudaMalloc(&dY, bytes));
  Trace d; int *h[5];                       // trace arrays on the GPU, and their host copies: bx, by, tx, ty, count
  for (int t = 0; t < 5; t++) { h[t] = (int *)malloc(n2 * sizeof(int)); }
  CUDA_CHECK(cudaMalloc(&d.bx, n2 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d.by, n2 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d.tx, n2 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d.ty, n2 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d.count, n2 * sizeof(int)));

  printf("== saxpy on a %d x %d matrix (%d-thread 1D blocks, %d x %d tiles): result, mapping and sectors per warp\n\n", n, n, BLOCK_SIZE, SIDE, SIDE);
  printf("%-44s %-12s %-8s %-8s %-9s %-14s %s\n", "kernel", "layout", "result", "covered", "mapping", "sectors/warp", "claimed");
  for (int kernel = 0; kernel < numKernels; kernel++) {
    for (int layout = ROW_MAJOR; layout <= COL_MAJOR; layout++) {
      initialise(X, Y, n, layout);
      CUDA_CHECK(cudaMemcpy(dX, X, bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dY, Y, bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(d.count, 0, n2 * sizeof(int)));
      CUDA_CHECK(cudaMemset(d.bx, 0xff, n2 * sizeof(int)));   // -1: "nobody came here"
      CUDA_CHECK(cudaMemset(d.by, 0xff, n2 * sizeof(int)));
      CUDA_CHECK(cudaMemset(d.tx, 0xff, n2 * sizeof(int)));
      CUDA_CHECK(cudaMemset(d.ty, 0xff, n2 * sizeof(int)));
      launch(kernel, n, a, dX, dY, layout, d);
      CUDA_CHECK(cudaMemcpy(Y, dY, bytes, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h[0], d.bx, n2 * sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h[1], d.by, n2 * sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h[2], d.tx, n2 * sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h[3], d.ty, n2 * sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h[4], d.count, n2 * sizeof(int), cudaMemcpyDeviceToHost));
      int errors = verify(Y, a, n, layout);
      TraceResult r = checkTrace(kernel, n, layout, h[0], h[1], h[2], h[3], h[4]);
      const char *claimed = (layout == ROW_MAJOR) ? kernels[kernel].rowMajor : kernels[kernel].colMajor;
      const char *measured = (r.maxSectors == 4) ? "yes" : "no";
      char sectors[32]; snprintf(sectors, sizeof sectors, "%.1f [%d-%d]", r.sectorsPerWarp, r.minSectors, r.maxSectors);
      printf("%-44s %-12s %-8s %-8s %-9s %-14s %s%s\n", kernels[kernel].name,
             layout == ROW_MAJOR ? "row-major" : "column-major",
             errors ? "WRONG" : "ok",
             r.coverageErrors ? "NO" : "ok",
             r.mappingErrors ? "WRONG" : "ok",
             sectors, claimed,
             strcmp(claimed, measured) ? "  ** claim does not match **" : "");
      if (r.mappingErrors) {
        printf("      %d elements touched by an unexpected thread; e.g. (i,j) = (%d,%d): expected block (%d,%d) thread (%d,%d), got block (%d,%d) thread (%d,%d)\n",
               r.mappingErrors, r.badI, r.badJ, r.firstBad.bx, r.firstBad.by, r.firstBad.tx, r.firstBad.ty,
               r.firstGot.bx, r.firstGot.by, r.firstGot.tx, r.firstGot.ty);
      }
    }
  }
  for (int t = 0; t < 5; t++) { free(h[t]); }
  CUDA_CHECK(cudaFree(d.bx)); CUDA_CHECK(cudaFree(d.by));
  CUDA_CHECK(cudaFree(d.tx)); CUDA_CHECK(cudaFree(d.ty));
  CUDA_CHECK(cudaFree(d.count));
  CUDA_CHECK(cudaFree(dX)); CUDA_CHECK(cudaFree(dY));
  free(X);
  free(Y);
  return 0;
}
