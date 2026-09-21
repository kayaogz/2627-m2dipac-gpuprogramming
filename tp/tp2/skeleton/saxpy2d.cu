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
//  Every mapping comes as TWO kernels, one per layout, that differ in exactly
//  one line: the position idx of element (i, j) in the array. The mappings
//  are the fourteen of session 2, plus four where each thread handles K x K
//  elements, so any difference in running time between two kernels comes
//  from the memory access pattern alone.
//
//  WHAT YOU HAVE TO DO, for every kernel marked TODO:
//    1. write the body of the two kernels (_rowmajor and _colmajor) following
//       the mapping described in the comment block above them: compute i and
//       j from the block/thread indices, then idx, then
//           Y[idx] = a * X[idx] + Y[idx];
//           CHECK_INDEX();
//       -- CHECK_INDEX() must follow EVERY assignment to Y[idx]: in the trace
//       run it records which block/thread touched the element, and the
//       driver checks that this is the thread the mapping prescribes;
//    2. write its launch configuration in config();
//    3. in the comment block, answer for both layouts: what do the 32 threads
//       of a warp touch in memory at one instruction, and is the access
//       COALESCED (the 32 addresses fall into 4 sectors of 32 bytes) or NOT
//       (up to 32 sectors)? Write your verdict ("yes"/"no") in the table
//       kernels[] as well: the driver compares it with the number of sectors
//       it measures.
//  Kernels 0 and 1 are given as worked examples.
//
//  The driver runs every kernel once, in both layouts, and prints for each:
//  the result check (Y = a*X + Y), coverage (every element touched once), the
//  exact mapping check, and the number of 32-byte sectors touched by each
//  warp instruction -- the coalescing verdict, measured.
//  Do not modify the driver (everything after the kernels).
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
//  This is the speed every other kernel is compared with.
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
//  Even when coalesced, this kernel is slow: N = 4096 threads cannot hide
//  the memory latency of a whole GPU.
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
//  The mirror image of kernel 1: thread j = global 1D index walks its own
//  column, i = 0 .. N-1. N threads, grid ceil(N / BLOCK_SIZE).
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_col_per_thread_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_col_per_thread_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

// =============================================================================
//  3. (a) 1D grid, 1D blocks, threads ROW aligned -- 1D indices
// =============================================================================
//  A block handles BLOCK_SIZE consecutive elements of a row. Blocks are
//  numbered row by row: blocksPerRow = ceil(N / BLOCK_SIZE) blocks per row,
//  grid N * blocksPerRow.
//    b_i = blockIdx.x / blocksPerRow   (which row)
//    b_j = blockIdx.x % blocksPerRow   (which piece of it)
//    i = b_i,   j = b_j * blockDim.x + threadIdx.x     (guard: j < n)
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_1d_a_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_1d_a_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

// =============================================================================
//  4. (b) 1D grid, 1D blocks, threads COLUMN aligned -- 1D indices
// =============================================================================
//  A block handles BLOCK_SIZE consecutive elements of a COLUMN. Blocks are
//  numbered column by column: blocksPerCol = ceil(N / BLOCK_SIZE),
//  grid N * blocksPerCol.
//    b_j = blockIdx.x / blocksPerCol,   b_i = blockIdx.x % blocksPerCol
//    i = b_i * blockDim.x + threadIdx.x,   j = b_j     (guard: i < n)
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_1d_b_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_1d_b_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

// =============================================================================
//  5. (a) 2D grid, 1D blocks, threads ROW aligned -- dim3 indices
// =============================================================================
//  Same mapping as kernel 3; blockIdx.x and blockIdx.y simply replace b_i
//  and b_j -- no division/modulus. Grid: N x ceil(N / BLOCK_SIZE).
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_2d_a_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_2d_a_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

// =============================================================================
//  6. (b) 2D grid, 1D blocks, threads COLUMN aligned -- dim3 indices
// =============================================================================
//  Same mapping as kernel 4 with dim3 indices: b_i = blockIdx.x,
//  b_j = blockIdx.y. Grid: ceil(N / BLOCK_SIZE) x N.
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_2d_b_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_2d_b_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

// =============================================================================
//  7. (c) 1D grid, tile blocks, threads COLUMN aligned -- 1D indices
// =============================================================================
//  A block handles a SIDE x SIDE tile. Tiles are numbered row by row
//  (tilesPerRow = ceil(N / SIDE) tiles per row of tiles), grid tilesPerRow^2:
//    b_i = blockIdx.x / tilesPerRow,   b_j = blockIdx.x % tilesPerRow
//  Inside the tile, threads are column aligned: consecutive threadIdx.x go
//  down a column of the tile:
//    t_i = threadIdx.x % SIDE,   t_j = threadIdx.x / SIDE
//    i = b_i * SIDE + t_i,       j = b_j * SIDE + t_j    (guard on both)
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_1d_c_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_1d_c_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

// =============================================================================
//  8. (d) 1D grid, tile blocks, threads ROW aligned -- 1D indices
// =============================================================================
//  Same tiles and same block numbering as kernel 7, but the threads are ROW
//  aligned: consecutive threadIdx.x walk along a row of the tile:
//    t_i = threadIdx.x / SIDE,   t_j = threadIdx.x % SIDE
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_1d_d_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_1d_d_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

// =============================================================================
//  9. (c) alternative: tiles numbered COLUMN by column, threads column aligned
// =============================================================================
//  Only the numbering of the blocks changes with respect to kernel 7
//  (consecutive blocks walk down a column of tiles):
//    b_j = blockIdx.x / tilesPerCol,   b_i = blockIdx.x % tilesPerCol
//  Threads column aligned as in kernel 7.
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_1d_c_alt_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_1d_c_alt_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

// =============================================================================
// 10. (d) alternative: tiles numbered column by column, threads ROW aligned
// =============================================================================
//  Tiles numbered column by column as in kernel 9, threads row aligned as in
//  kernel 8.
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_1d_d_alt_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_1d_d_alt_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

// =============================================================================
// 11. (c) 2D grid, 2D blocks, threads COLUMN aligned -- dim3 indices
// =============================================================================
//  dimBlock = (SIDE, SIDE), grid ceil(N / SIDE) x ceil(N / SIDE): the
//  coordinates are read directly,
//    t_i = threadIdx.x,  t_j = threadIdx.y,  b_i = blockIdx.x,  b_j = blockIdx.y
//    i = b_i * blockDim.x + t_i,   j = b_j * blockDim.y + t_j
//  (each coordinate is multiplied by the extent of the index it contains).
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_2d_c_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_2d_c_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

// =============================================================================
// 12. (d) 2D grid, 2D blocks, threads ROW aligned -- dim3 indices
// =============================================================================
//  Same grid and blocks as kernel 11, threads ROW aligned:
//    t_i = threadIdx.y,  t_j = threadIdx.x,  b_i = blockIdx.x,  b_j = blockIdx.y
//    i = b_i * blockDim.y + t_i,   j = b_j * blockDim.x + t_j
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_2d_d_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_2d_d_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

// =============================================================================
// 13. (c) alternative with dim3: block indices interchanged, threads column aligned
// =============================================================================
//  b_i = blockIdx.y, b_j = blockIdx.x (the grid is transposed); threads as
//  in kernel 11.
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_2d_c_alt_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_2d_c_alt_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

// =============================================================================
// 14. (d) alternative with dim3: block indices interchanged, threads row aligned
// =============================================================================
//  b_i = blockIdx.y, b_j = blockIdx.x; threads as in kernel 12.
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_2d_d_alt_rowmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_2d_d_alt_colmajor(int n, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

// =============================================================================
// 15. (c) 2D blocks, threads column aligned, K x K elements per thread
// =============================================================================
//  Same thread and block coordinates as kernel 11, but each thread now
//  computes K x K elements instead of one, INTERLEAVED with the other threads
//  of its block: a block covers a (32K) x (32K) tile, and thread (t_i, t_j)
//  handles the elements
//      i = b_i*32K + t_i + 32*di,   j = b_j*32K + t_j + 32*dj,   di, dj < K
//  The grid is ceil(N / 32K) x ceil(N / 32K), blocks (SIDE, SIDE).
//  (Do NOT give each thread a contiguous K x K patch: think about what the
//  warp would touch then.)
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_2d_c_kxk_rowmajor(int n, int K, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_2d_c_kxk_colmajor(int n, int K, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

// =============================================================================
// 16. (d) 2D blocks, threads row aligned, K x K elements per thread
// =============================================================================
//  Same thread and block coordinates as kernel 12, but each thread now
//  computes K x K elements instead of one, INTERLEAVED with the other threads
//  of its block: a block covers a (32K) x (32K) tile, and thread (t_i, t_j)
//  handles the elements
//      i = b_i*32K + t_i + 32*di,   j = b_j*32K + t_j + 32*dj,   di, dj < K
//  The grid is ceil(N / 32K) x ceil(N / 32K), blocks (SIDE, SIDE).
//  (Do NOT give each thread a contiguous K x K patch: think about what the
//  warp would touch then.)
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_2d_d_kxk_rowmajor(int n, int K, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_2d_d_kxk_colmajor(int n, int K, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

// =============================================================================
// 17. (c) alternative (grid transposed), K x K elements per thread
// =============================================================================
//  Same thread and block coordinates as kernel 13, but each thread now
//  computes K x K elements instead of one, INTERLEAVED with the other threads
//  of its block: a block covers a (32K) x (32K) tile, and thread (t_i, t_j)
//  handles the elements
//      i = b_i*32K + t_i + 32*di,   j = b_j*32K + t_j + 32*dj,   di, dj < K
//  The grid is ceil(N / 32K) x ceil(N / 32K), blocks (SIDE, SIDE).
//  (Do NOT give each thread a contiguous K x K patch: think about what the
//  warp would touch then.)
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_2d_c_alt_kxk_rowmajor(int n, int K, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_2d_c_alt_kxk_colmajor(int n, int K, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

// =============================================================================
// 18. (d) alternative (grid transposed), K x K elements per thread
// =============================================================================
//  Same thread and block coordinates as kernel 14, but each thread now
//  computes K x K elements instead of one, INTERLEAVED with the other threads
//  of its block: a block covers a (32K) x (32K) tile, and thread (t_i, t_j)
//  handles the elements
//      i = b_i*32K + t_i + 32*di,   j = b_j*32K + t_j + 32*dj,   di, dj < K
//  The grid is ceil(N / 32K) x ceil(N / 32K), blocks (SIDE, SIDE).
//  (Do NOT give each thread a contiguous K x K patch: think about what the
//  warp would touch then.)
//
//  TODO: at one instruction, the 32 threads of a warp touch ... :
//    row-major    : ?
//    column-major : ?
// =============================================================================
__global__ void saxpy_2d_d_alt_kxk_rowmajor(int n, int K, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (row-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
}

__global__ void saxpy_2d_d_alt_kxk_colmajor(int n, int K, float a, const float *X, float *Y, Trace tr)
{
  // TODO: compute i and j, then idx (column-major), Y[idx] = a * X[idx] + Y[idx], CHECK_INDEX()
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
  c.block = dim3(BLOCK_SIZE);                                // 1D block, unless said otherwise
  switch (kernel) {
    case 0:  c.grid = dim3((n * n + BLOCK_SIZE - 1) / BLOCK_SIZE); break;
    case 1:  c.grid = dim3(nbLinear); break;
    // TODO: one case per kernel: the grid, and the block when it is not a 1D block of BLOCK_SIZE threads
    //       (tile kernels: SIDE * SIDE threads, as a 1D block for kernels 7-10, as a SIDE x SIDE 2D block for 11-18)
    case 2:  c.grid = dim3(1); break;
    case 3:  c.grid = dim3(1); break;
    case 4:  c.grid = dim3(1); break;
    case 5:  c.grid = dim3(1); break;
    case 6:  c.grid = dim3(1); break;
    case 7:  c.grid = dim3(1); break;
    case 8:  c.grid = dim3(1); break;
    case 9:  c.grid = dim3(1); break;
    case 10: c.grid = dim3(1); break;
    case 11: c.grid = dim3(1); break;
    case 12: c.grid = dim3(1); break;
    case 13: c.grid = dim3(1); break;
    case 14: c.grid = dim3(1); break;
    case 15: c.grid = dim3(1); break;
    case 16: c.grid = dim3(1); break;
    case 17: c.grid = dim3(1); break;
    case 18: c.grid = dim3(1); break;
  }
  (void)nbLinear; (void)nbTiles;
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
// reference against which the trace is checked, element by element -- the
// formal specification of every kernel above. Do not modify.
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

// Names of the mappings, and YOUR coalescing verdicts ("yes"/"no"), one per layout.
struct KernelInfo { const char *name; const char *rowMajor; const char *colMajor; };
static const KernelInfo kernels[] = {
  { "0  1D reference (flat)",                    "yes", "yes" },
  { "1  one thread per row",                      "no",  "yes" },
  { "2  one thread per column",                    "?",   "?"   },
  { "3  (a) 1D blocks, row aligned, 1D idx",       "?",   "?"   },
  { "4  (b) 1D blocks, col aligned, 1D idx",       "?",   "?"   },
  { "5  (a) 1D blocks, row aligned, dim3",         "?",   "?"   },
  { "6  (b) 1D blocks, col aligned, dim3",         "?",   "?"   },
  { "7  (c) tiles, col aligned, 1D idx",           "?",   "?"   },
  { "8  (d) tiles, row aligned, 1D idx",           "?",   "?"   },
  { "9  (c) alt: tiles col by col, col aligned",   "?",   "?"   },
  { "10 (d) alt: tiles col by col, row aligned",   "?",   "?"   },
  { "11 (c) 2D blocks, col aligned, dim3",         "?",   "?"   },
  { "12 (d) 2D blocks, row aligned, dim3",         "?",   "?"   },
  { "13 (c) alt: grid transposed, col aligned",    "?",   "?"   },
  { "14 (d) alt: grid transposed, row aligned",    "?",   "?"   },
  { "15 (c) 2D blocks, col aligned, KxK/thread",   "?",   "?"   },
  { "16 (d) 2D blocks, row aligned, KxK/thread",   "?",   "?"   },
  { "17 (c) alt, KxK/thread",                      "?",   "?"   },
  { "18 (d) alt, KxK/thread",                      "?",   "?"   },
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
             !strcmp(claimed, "?") ? "  ** verdict to fill in **"
             : strcmp(claimed, measured) ? "  ** claim does not match **" : "");
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
