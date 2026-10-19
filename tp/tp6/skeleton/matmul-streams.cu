// =============================================================================
//  TP6 -- Streaming matrix multiplication using CUDA: by columns, then by panels
//
//  C = A B for n x n matrices, all three COLUMN-MAJOR:
//
//      X(i, j) is at X[i + j * n]      -- a column is n consecutive values
//
//  A is copied to the GPU once. B then goes in, and C comes out, a chunk at a
//  time: that is the shape of a program whose data does not fit on the GPU or
//  arrives over time, and the point of the lab: overlap the copies and the
//  computation with streams, then choose the size of the chunks.
//
//  EXERCISE 1 -- single precision, n = 2048: column by column
//    1. matvec + version1: the kernel (one thread per row i of A: thread i
//       walks row i, A[i + k * n] for k = 0..n-1 -- consecutive threads read
//       consecutive addresses: coalesced) and the sequential loop over the
//       columns: cudaMemcpy, kernel, cudaMemcpy, column after column
//    2. version2: the same loop with S streams and S slots on the GPU:
//       column j goes to stream j % S and slot j % S, cudaMemcpyAsync in,
//       kernel, cudaMemcpyAsync out, all in that stream; wait for every
//       stream at the end. main() runs it with S = 1, 2, 4, 8, 16
//
//  EXERCISE 2 -- double precision, n = 1536: panels of M columns
//    3. version3: panels of M columns instead of single columns (M
//       consecutive columns are M * n consecutive values), and cublasDgemm
//       on each panel in the stream of the panel (cublasSetStream):
//       C(:, p) = A B(:, p) is an n x M product. main() runs it with
//       M = 8, 32, 128, 512, S = 4 streams
//    4. version4: the CPU joins in. Of the n / M panels, the last npcpu are
//       computed by the CPU with cpu_dgemm (given: AVX2, all the cores with
//       OpenMP), the others by the GPU as in version 3, at the same time --
//       four things at once: copies in, the GPU's dgemm, copies out, the
//       CPU's dgemm. main() runs it with M = 128, S = 4, npcpu = 0, 1, 2, 3, 4
//  Why double precision: GPUs like the lab's compute in double 32 to 64
//  times slower than in single, while a CPU core loses only a factor 2.
//  Next to the GPU in single precision the CPU would be almost useless; in
//  double it takes a real share of the work.
//
//  Given: the data (small integers, so that every result is exact and is
//  compared element by element with the reference), the references --
//  cuBLAS's sgemm (exercise 1) and dgemm (exercise 2) on the whole matrices,
//  timed with the copies of B and C --, the CPU dgemm (at the end of the
//  file), the timing (the host clock), the checks, and main(). The host
//  arrays B and C are PINNED (cudaMallocHost): cudaMemcpyAsync is
//  asynchronous only from pinned memory (set PINNED to 0 to see what pageable
//  memory does).
//
//  Build:  nvcc -O2 -arch=native matmul-streams.cu -o matmul-streams -lcublas
//              -Xcompiler -fopenmp -lgomp
//  Run:    ./matmul-streams      (OMP_NUM_THREADS sets the CPU's threads:
//                                 one per core is best)
//  The program calls cuBLAS and OpenMP: it MUST be linked with -lcublas, and
//  built with -Xcompiler -fopenmp (which passes -fopenmp to the host
//  compiler, for the OpenMP pragmas of cpu_dgemm) and -lgomp (the OpenMP
//  runtime). Without -fopenmp, cpu_dgemm runs on one core.
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <immintrin.h>
#include <omp.h>
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

#define CUBLAS_CHECK(call) do { \
  cublasStatus_t st = (call); \
  if (st != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__, __LINE__, (int)st); \
    exit(EXIT_FAILURE); \
  } } while (0)

// ----------------------------------------------------------------------------
//  Sizes
// ----------------------------------------------------------------------------
#define BS     256    // threads per block of matvec
#define MAXS   16     // streams at most (versions 2 to 4)
#define MAXM   512    // columns per panel at most (versions 3 and 4, S = 4)
#define PINNED 1      // 1: B and C in pinned host memory; 0: pageable (malloc)

// ----------------------------------------------------------------------------
//  Given: the CPU's matrix product, used in version 4
// ----------------------------------------------------------------------------
//  C = A B on all the cores of the CPU, for column-major doubles: C is m x n
//  (leading dimension ldc), A is m x k (lda), B is k x n (ldb). Its
//  implementation, with AVX2 and OpenMP, is at the end of the file.
void cpu_dgemm(int m, int n, int k, const double *A, int lda,
               const double *B, int ldb, double *C, int ldc);

// #############################################################################
//  EXERCISE 1 -- single precision, column by column
// #############################################################################

// =============================================================================
//  1. The matrix-vector product: one thread per row of A
// =============================================================================
//  c = A b for one column b of B (n floats on the GPU) into one column c of C.
//  Thread i computes c[i] = sum over k of A(i, k) * b[k] = A[i + k * n] * b[k]:
//  at step k the threads of a warp read 32 consecutive floats of A (coalesced)
//  and all the same b[k] (a broadcast). n / BS blocks of BS threads, rounded
//  up; the threads past n do nothing.
// =============================================================================
__global__ void matvec(int n, const float *A, const float *b, float *c)
{
  // TODO
}

// =============================================================================
//  Version 1: column by column, sequentially
// =============================================================================
//  dB and dC each hold one column on the GPU (n floats). For each column j:
//  copy B(:, j) = B + j * n into dB (cudaMemcpy), launch matvec on it, copy
//  dC back into C(:, j) = C + j * n. Each cudaMemcpy waits for everything
//  before it: the copies and the kernels take turns.
// =============================================================================
void version1(int n, const float *dA, const float *B, float *C, float *dB, float *dC)
{
  // TODO: for each column j, cudaMemcpy in, matvec, cudaMemcpy out
}

// =============================================================================
//  Version 2: S streams, S slots
// =============================================================================
//  dB and dC now hold S columns each (slot s at dB + s * n). Column j goes to
//  stream j % S and slot j % S: cudaMemcpyAsync of B(:, j) into the slot,
//  matvec on the slot (4th launch parameter: the stream), cudaMemcpyAsync of
//  the slot into C(:, j) -- all three in that stream, so they run in order;
//  what runs at the same time is the work of different streams. After the
//  loop, wait for every stream (cudaDeviceSynchronize). Create the S streams
//  at the start (cudaStreamCreate) and destroy them at the end.
// =============================================================================
void version2(int n, int S, const float *dA, const float *B, float *C, float *dB, float *dC)
{
  // TODO: S streams, S slots; column j in stream j % S, cudaMemcpyAsync
  // in, matvec in the stream, cudaMemcpyAsync out; wait at the end
}

// #############################################################################
//  EXERCISE 2 -- double precision, panels of M columns
// #############################################################################

// =============================================================================
//  Version 3: panels of M columns, cuBLAS on each panel
// =============================================================================
//  The columns j = p * M .. p * M + M - 1 of B are M * n consecutive
//  doubles at B + p * M * n: one cudaMemcpyAsync moves a whole panel. The
//  slots are now M * n doubles (slot s at dB + s * M * n). On the panel,
//  C(:, panel) = A B(:, panel) is an n x M product: cublasDgemm with
//  m = n, n = M, k = n, no transposition, leading dimensions n:
//      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, M, n,
//                  &one, dA, n, dBs, n, &zero, dCs, n)   (dBs, dCs: the slot)
//  cuBLAS queues its kernels in the stream set by cublasSetStream(handle,
//  stream): set it before each call. n / M panels, panel p in stream p % S.
//  After the loop, wait for every stream, then destroy them.
// =============================================================================
void version3(int n, int S, int M, cublasHandle_t handle, const double *dA,
              const double *B, double *C, double *dB, double *dC)
{
  // TODO: panels of M columns, cublasSetStream + cublasDgemm on each
  // panel in stream p % S
}

// =============================================================================
//  Version 4: the CPU joins in -- GPU panels and CPU panels at the same time
// =============================================================================
//  Of the n / M panels, the first n / M - npcpu go to the GPU exactly as in
//  version 3 (panel p in stream p % S), the last npcpu to the CPU: one call
//  of cpu_dgemm (declared at the top of the file) on their npcpu * M
//  columns, which reads A, B and writes C in host memory -- no copy for them.
//  Issue all the GPU work first, then call cpu_dgemm, then wait for the
//  streams: the copy engines, the GPU and the CPU then all work at the same
//  time.
// =============================================================================
void version4(int n, int S, int M, int npcpu, cublasHandle_t handle, const double *A,
              const double *dA, const double *B, double *C, double *dB, double *dC)
{
  // TODO: the first n / M - npcpu panels on the GPU as in version 3 (issued
  // first), cpu_dgemm on the last npcpu panels, then wait for the streams
}

// #############################################################################
//  Given: timing, references, data, checks, driver
// #############################################################################

// ----------------------------------------------------------------------------
//  The host clock, in ms. Every operation timed with it ends with the CPU
//  waiting for the GPU (a cudaMemcpy, or cudaDeviceSynchronize).
// ----------------------------------------------------------------------------
double now(void)
{
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// =============================================================================
//  The references, cuBLAS on the whole matrices
// =============================================================================
//  B copied whole, one gemm, C copied whole (into the same pinned C as the
//  versions): the three times are printed, and their sum is the reference
//  pipeline to beat. The result, kept in Cref, is the reference for the
//  checks (exact: the data are small integers). The gemm runs in the default
//  stream: the handle is put back on it first, whatever a version left.
// =============================================================================
void reference32(cublasHandle_t handle, int n, const float *dA, const float *B, float *C,
                 float *Cref, float *dBfull, float *dCfull, double *ms)
{
  const float one = 1.0f, zero = 0.0f;
  size_t bytes = (size_t)n * n * sizeof(float);
  CUBLAS_CHECK(cublasSetStream(handle, 0));
  double t0 = now();
  CUDA_CHECK(cudaMemcpy(dBfull, B, bytes, cudaMemcpyHostToDevice));
  double t1 = now();
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                           &one, dA, n, dBfull, n, &zero, dCfull, n));
  CUDA_CHECK(cudaDeviceSynchronize());     // the sgemm returns before it is done
  double t2 = now();
  CUDA_CHECK(cudaMemcpy(C, dCfull, bytes, cudaMemcpyDeviceToHost));
  double t3 = now();
  ms[0] = t1 - t0; ms[1] = t2 - t1; ms[2] = t3 - t2;
  for (size_t x = 0; x < (size_t)n * n; x++) { Cref[x] = C[x]; C[x] = -1.0f; }
}

void reference64(cublasHandle_t handle, int n, const double *dA, const double *B, double *C,
                 double *Cref, double *dBfull, double *dCfull, double *ms)
{
  const double one = 1.0, zero = 0.0;
  size_t bytes = (size_t)n * n * sizeof(double);
  CUBLAS_CHECK(cublasSetStream(handle, 0));
  double t0 = now();
  CUDA_CHECK(cudaMemcpy(dBfull, B, bytes, cudaMemcpyHostToDevice));
  double t1 = now();
  CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                           &one, dA, n, dBfull, n, &zero, dCfull, n));
  CUDA_CHECK(cudaDeviceSynchronize());
  double t2 = now();
  CUDA_CHECK(cudaMemcpy(C, dCfull, bytes, cudaMemcpyDeviceToHost));
  double t3 = now();
  ms[0] = t1 - t0; ms[1] = t2 - t1; ms[2] = t3 - t2;
  for (size_t x = 0; x < (size_t)n * n; x++) { Cref[x] = C[x]; C[x] = -1.0; }
}

// ----------------------------------------------------------------------------
//  Data, check, report -- templates: the same code serves the floats of
//  exercise 1 and the doubles of exercise 2
// ----------------------------------------------------------------------------
// GPUs raise their clock after a few hundred ms of load: without this spin
// first, the first timings would run at a lower clock.
__global__ void settle_clock()
{
  long long t0 = clock64();
  while (clock64() - t0 < 400000000LL) { }
}

// Small integers: every C(i, j) is an integer below 2^24, exact in float and
// in double whatever the order of the additions, so the results compare
// exactly.
template <class T> void initialise(int n, T *A, T *B)
{
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      A[i + j * n] = (T)((i * 7 + j) % 4);
      B[i + j * n] = (T)((i * 3 + j) % 5);
    }
  }
}

// The number of elements of C that differ from the reference; C is reset
// to -1 afterwards, so that a version that skips a column is caught.
template <class T> int check(int n, T *C, const T *Cref)
{
  int errors = 0;
  for (size_t x = 0; x < (size_t)n * n; x++) { if (C[x] != Cref[x]) errors++; }
  for (size_t x = 0; x < (size_t)n * n; x++) { C[x] = (T)-1; }
  return errors;
}

template <class T> void report(const char *name, int n, double ms, T *C, const T *Cref)
{
  int errors = check(n, C, Cref);
  double gflops = 2.0 * n * (double)n * n / (ms * 1e-3) / 1e9;
  if (errors) { printf("%-28s WRONG (%d elements)\n", name, errors); }
  else        { printf("%-28s ok  %10.3f ms %9.1f GFLOP/s\n", name, ms, gflops); }
}

// Host memory, pinned or not (PINNED)
template <class T> T *hostAlloc(size_t count)
{
  T *p;
#if PINNED
  CUDA_CHECK(cudaMallocHost(&p, count * sizeof(T)));   // the copy engine reads it directly
#else
  p = (T *)malloc(count * sizeof(T));
#endif
  return p;
}
template <class T> void hostFree(T *p)
{
#if PINNED
  CUDA_CHECK(cudaFreeHost(p));
#else
  free(p);
#endif
}

// =============================================================================
//  Driver
// =============================================================================
int main(void)
{
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  settle_clock<<<1, 32>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  double ms[3], t0;

  // ======================================================== EXERCISE 1
  {
    const int n = 2048;                    // single precision
    size_t count = (size_t)n * n, bytes = count * sizeof(float);
    float *A = (float *)malloc(bytes), *Cref = (float *)malloc(bytes);
    float *B = hostAlloc<float>(count), *C = hostAlloc<float>(count);
    initialise(n, A, B);
    float *dA, *dBfull, *dCfull, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dBfull, bytes));
    CUDA_CHECK(cudaMalloc(&dCfull, bytes));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)MAXS * n * sizeof(float)));   // the slots
    CUDA_CHECK(cudaMalloc(&dC, (size_t)MAXS * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));    // A: once
    printf("EXERCISE 1: single precision, n = %d (%zu MB per matrix), B and C %s\n",
           n, bytes >> 20, PINNED ? "pinned" : "pageable");

    // the reference (twice: the first call of cuBLAS sets it up, not reported)
    reference32(handle, n, dA, B, C, Cref, dBfull, dCfull, ms);
    reference32(handle, n, dA, B, C, Cref, dBfull, dCfull, ms);
    printf("%-28s     %10.3f ms  = copy B %.3f + sgemm %.3f + copy C %.3f\n",
           "reference (cuBLAS sgemm)", ms[0] + ms[1] + ms[2], ms[0], ms[1], ms[2]);
    matvec<<<(n + BS - 1) / BS, BS>>>(n, dA, dBfull, dC);   // loads the kernel, untimed
    CUDA_CHECK(cudaDeviceSynchronize());
    // Timing: the host clock around each version, after cudaDeviceSynchronize()
    // -- the CPU stops the clock only when every stream is done.

    // ---- 1. sequential, column by column
    t0 = now();
    version1(n, dA, B, C, dB, dC);
    CUDA_CHECK(cudaDeviceSynchronize());
    report("1 sequential", n, now() - t0, C, Cref);

    // ---- 2. S streams
    int Ss[] = {1, 2, 4, 8, 16};
    for (int k = 0; k < 5; k++) {
      char name[32];
      snprintf(name, sizeof name, "2 streams, S = %d", Ss[k]);
      t0 = now();
      version2(n, Ss[k], dA, B, C, dB, dC);
      CUDA_CHECK(cudaDeviceSynchronize());
      report(name, n, now() - t0, C, Cref);
    }
    // everything freed before exercise 2, so that less host memory is
    // pinned at a time
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dBfull)); CUDA_CHECK(cudaFree(dCfull));
    CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    hostFree(B); hostFree(C);
    free(A); free(Cref);
  }

  // ======================================================== EXERCISE 2
  {
    const int n = 1536;                    // double precision (a multiple of MAXM)
    size_t count = (size_t)n * n, bytes = count * sizeof(double);
    double *A = (double *)malloc(bytes), *Cref = (double *)malloc(bytes);
    double *B = hostAlloc<double>(count), *C = hostAlloc<double>(count);
    initialise(n, A, B);
    double *dA, *dBfull, *dCfull, *dB, *dC;
    size_t slots = (size_t)4 * MAXM * n * sizeof(double);           // S = 4 slots
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dBfull, bytes));
    CUDA_CHECK(cudaMalloc(&dCfull, bytes));
    CUDA_CHECK(cudaMalloc(&dB, slots));
    CUDA_CHECK(cudaMalloc(&dC, slots));
    CUDA_CHECK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
    printf("EXERCISE 2: double precision, n = %d (%zu MB per matrix), B and C %s\n",
           n, bytes >> 20, PINNED ? "pinned" : "pageable");

    reference64(handle, n, dA, B, C, Cref, dBfull, dCfull, ms);
    reference64(handle, n, dA, B, C, Cref, dBfull, dCfull, ms);
    printf("%-28s     %10.3f ms  = copy B %.3f + dgemm %.3f + copy C %.3f\n",
           "reference (cuBLAS dgemm)", ms[0] + ms[1] + ms[2], ms[0], ms[1], ms[2]);
    // the CPU's dgemm alone, on one panel of 128 columns: its speed, for version 4
    {
      double best = 1e30;
      for (int r = 0; r < 3; r++) {
        double t = now();
        cpu_dgemm(n, 128, n, A, n, B, n, C, n);
        t = now() - t;
        if (t < best) { best = t; }
      }
      printf("%-28s     %10.3f ms  = %.1f GFLOP/s, %d threads\n", "CPU dgemm, one panel",
             best, 2.0 * n * (double)n * 128 / best / 1e6, omp_get_max_threads());
      for (size_t x = 0; x < count; x++) { C[x] = -1.0; }
    }

    // ---- 3. panels of M columns, cuBLAS on each, S = 4 streams
    int Ms[] = {8, 32, 128, 512};
    for (int k = 0; k < 4; k++) {
      char name[32];
      snprintf(name, sizeof name, "3 panels, M = %d, S = 4", Ms[k]);
      t0 = now();
      version3(n, 4, Ms[k], handle, dA, B, C, dB, dC);
      CUDA_CHECK(cudaDeviceSynchronize());
      report(name, n, now() - t0, C, Cref);
    }

    // ---- 4. panels of M = 128 columns, the last npcpu of them on the CPU
    int npcpus[] = {0, 1, 2, 3, 4};
    for (int k = 0; k < 5; k++) {
      char name[32];
      snprintf(name, sizeof name, "4 GPU + CPU, npcpu = %d", npcpus[k]);
      t0 = now();
      version4(n, 4, 128, npcpus[k], handle, A, dA, B, C, dB, dC);
      CUDA_CHECK(cudaDeviceSynchronize());
      report(name, n, now() - t0, C, Cref);
    }
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dBfull)); CUDA_CHECK(cudaFree(dCfull));
    CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    hostFree(B); hostFree(C);
    free(A); free(Cref);
  }
  CUBLAS_CHECK(cublasSetStream(handle, 0));   // the versions' streams are gone
  CUBLAS_CHECK(cublasDestroy(handle));
  return 0;
}

// #############################################################################
// #############################################################################
//
//  CPU DGEMM IMPLEMENTATION, WITH AVX2 AND OPENMP
//  (given, used by version 4 -- nothing to write below this line)
//
// #############################################################################
// #############################################################################
//
//  cpu_dgemm: C (m x n, leading dimension ldc) = A (m x k, lda) B (k x n, ldb),
//  all column-major doubles, on all the cores. The classic organisation of a
//  fast gemm (as in BLIS / OpenBLAS), in short:
//    - a MICRO-KERNEL computes an 8 x 6 tile of C in 12 AVX registers of 4
//      doubles: at each step p it loads 8 values of A (2 registers),
//      broadcasts 6 values of B, and does 12 fused multiply-adds -- 96 flops
//      for 8 loads, so the FMA units, not the memory, set the pace;
//    - A and B are PACKED first into the order the micro-kernel reads them
//      (8 rows of A at each step, 6 columns of B at each step), in blocks of
//      GEMM_MC x GEMM_KC of A (128 KB: stays in the L2 cache) and
//      GEMM_KC x n of B;
//    - the threads share the packed B and take the blocks of GEMM_MC rows of
//      A in turn (#pragma omp for schedule(dynamic)).
//  The kernel is compiled for AVX2 + FMA whatever the flags (target
//  attribute): any x86 CPU since 2013 (Haswell) runs it.
// =============================================================================
#define GEMM_MR 8      // rows of the micro-tile (2 registers of 4 doubles)
#define GEMM_NR 6      // columns of the micro-tile
#define GEMM_MC 64     // rows of a block of A (a thread's unit of work)
#define GEMM_KC 256    // depth of a block

// C(0:8, 0:6) += a (8 x kc, packed) b (kc x 6, packed)
__attribute__((target("avx2,fma")))
static void gemm_kernel(int kc, const double *a, const double *b, double *c, int ldc)
{
  __m256d c00 = _mm256_setzero_pd(), c10 = _mm256_setzero_pd();
  __m256d c01 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
  __m256d c02 = _mm256_setzero_pd(), c12 = _mm256_setzero_pd();
  __m256d c03 = _mm256_setzero_pd(), c13 = _mm256_setzero_pd();
  __m256d c04 = _mm256_setzero_pd(), c14 = _mm256_setzero_pd();
  __m256d c05 = _mm256_setzero_pd(), c15 = _mm256_setzero_pd();
  for (int p = 0; p < kc; p++) {
    __m256d a0 = _mm256_load_pd(a), a1 = _mm256_load_pd(a + 4), bj;
#define STEP(j) bj = _mm256_broadcast_sd(b + j); \
    c0##j = _mm256_fmadd_pd(a0, bj, c0##j); c1##j = _mm256_fmadd_pd(a1, bj, c1##j);
    STEP(0) STEP(1) STEP(2) STEP(3) STEP(4) STEP(5)
#undef STEP
    a += GEMM_MR; b += GEMM_NR;
  }
#define STORE(j) { double *cj = c + (size_t)j * ldc; \
    _mm256_storeu_pd(cj, _mm256_add_pd(_mm256_loadu_pd(cj), c0##j)); \
    _mm256_storeu_pd(cj + 4, _mm256_add_pd(_mm256_loadu_pd(cj + 4), c1##j)); }
  STORE(0) STORE(1) STORE(2) STORE(3) STORE(4) STORE(5)
#undef STORE
}

// A(0:mc, 0:kc) into micro-panels of 8 rows: for each p, 8 consecutive values
static void pack_a(int mc, int kc, const double *A, int lda, double *ap)
{
  for (int i0 = 0; i0 < mc; i0 += GEMM_MR) {
    int mr = mc - i0 < GEMM_MR ? mc - i0 : GEMM_MR;
    for (int p = 0; p < kc; p++) {
      const double *src = A + i0 + (size_t)p * lda;
      for (int i = 0; i < mr; i++) { ap[i] = src[i]; }
      for (int i = mr; i < GEMM_MR; i++) { ap[i] = 0.0; }    // zero padding
      ap += GEMM_MR;
    }
  }
}

// B(0:kc, 0:nr), nr <= 6 columns, into one micro-panel: for each p, 6 values
static void pack_b(int kc, int nr, const double *B, int ldb, double *bp)
{
  for (int p = 0; p < kc; p++) {
    for (int j = 0; j < nr; j++) { bp[j] = B[p + (size_t)j * ldb]; }
    for (int j = nr; j < GEMM_NR; j++) { bp[j] = 0.0; }      // zero padding
    bp += GEMM_NR;
  }
}

void cpu_dgemm(int m, int n, int k, const double *A, int lda,
               const double *B, int ldb, double *C, int ldc)
{
  for (int j = 0; j < n; j++) { memset(C + (size_t)j * ldc, 0, (size_t)m * sizeof(double)); }
  int nb = (n + GEMM_NR - 1) / GEMM_NR;  // micro-panels of B
  double *bp = (double *)aligned_alloc(64, (size_t)GEMM_KC * nb * GEMM_NR * sizeof(double));
#pragma omp parallel
  {
    double *ap = (double *)aligned_alloc(64, (size_t)GEMM_MC * GEMM_KC * sizeof(double));
    alignas(32) double t[GEMM_MR * GEMM_NR];  // a tile at the edges of C
    for (int pc = 0; pc < k; pc += GEMM_KC) {
      int kc = k - pc < GEMM_KC ? k - pc : GEMM_KC;
#pragma omp for
      for (int jb = 0; jb < nb; jb++) {   // pack B(pc:pc+kc, :), shared by all
        int nr = n - jb * GEMM_NR < GEMM_NR ? n - jb * GEMM_NR : GEMM_NR;
        pack_b(kc, nr, B + pc + (size_t)jb * GEMM_NR * ldb, ldb, bp + (size_t)jb * GEMM_NR * kc);
      }
#pragma omp for schedule(dynamic)
      for (int ic = 0; ic < m; ic += GEMM_MC) {   // a block of rows per thread
        int mc = m - ic < GEMM_MC ? m - ic : GEMM_MC;
        pack_a(mc, kc, A + ic + (size_t)pc * lda, lda, ap);
        for (int jb = 0; jb < nb; jb++) {
          int nr = n - jb * GEMM_NR < GEMM_NR ? n - jb * GEMM_NR : GEMM_NR;
          const double *b = bp + (size_t)jb * GEMM_NR * kc;
          for (int i0 = 0; i0 < mc; i0 += GEMM_MR) {
            int mr = mc - i0 < GEMM_MR ? mc - i0 : GEMM_MR;
            const double *a = ap + (size_t)i0 * kc;
            double *c = C + ic + i0 + (size_t)jb * GEMM_NR * ldc;
            if (mr == GEMM_MR && nr == GEMM_NR) {
              gemm_kernel(kc, a, b, c, ldc);
            } else {                      // an edge: through the tile t
              memset(t, 0, sizeof t);
              gemm_kernel(kc, a, b, t, GEMM_MR);
              for (int j = 0; j < nr; j++) {
                for (int i = 0; i < mr; i++) { c[i + (size_t)j * ldc] += t[i + j * GEMM_MR]; }
              }
            }
          }
        }
      }
    }
    free(ap);
  }
  free(bp);
}
