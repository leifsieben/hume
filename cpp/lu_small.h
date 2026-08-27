// lu_small.h -- dense LU solve and a small GEMM, no BLAS, no LAPACK.
//
// WHY THIS EXISTS. The resistance block solves (L + J/k) X = I once per connected component and
// multiplies matrix powers in rw_returns. Both went to Accelerate, and Accelerate is not a
// portable dependency: `-framework Accelerate` does not exist off macOS, and hume_blocks.h asked
// for `dgesv$NEWLAPACK` / `dgemm$NEWLAPACK` BY NAME through asm labels because Accelerate ships
// two LAPACKs that round differently. Those symbols exist nowhere else, so the build refused to
// configure off Apple rather than silently producing different numbers.
//
// THE CONSTRAINT IS BIT-IDENTITY, NOT ACCURACY, and that is the whole difficulty. The resistance
// bins have edges at 0.1 / 0.5 / 1.0 / 2.0 and real molecular Omega values land EXACTLY on them
// -- on tetra-tert-butyl tetrahedrane every pair has Omega = 2/n = 0.5 exactly, so delta sits on
// a bin boundary and a one-ulp difference moves a pair from RPAIR2 to RPAIR1 and takes four
// RATSC columns with it. A solver that is "accurate enough" in the usual sense still changes an
// integer column. So this header does not aim to be a good solver; it aims to be THE SAME
// SOLVER, in the same order, so that agreement is structural rather than lucky.
//
// NOT CHOLESKY. (L + J/k) is symmetric positive definite and a Cholesky factorisation is both
// shorter and better conditioned, which is why it is the obvious thing to reach for -- and it is
// the wrong thing. numpy.linalg.inv, which the resistance block is verified against, is
// gesv(A, I): LU with partial pivoting. Cholesky computes a DIFFERENT rounding of the same
// mathematical object. This was not left as an argument: see the measurement in the header
// comment of resistance() in hume_blocks.h, and the check reported below. Matching the ALGORITHM
// is what makes the arithmetic identical instead of merely close.
//
// WHAT IS IMPLEMENTED. Reference LAPACK/BLAS, column-major, unblocked:
//
//   getf2   -- LAPACK dgetf2: right-looking LU with partial pivoting (idamax + dswap + dscal +
//              dger), the unblocked kernel dgetrf itself calls when n <= the block size.
//   getrs_n -- LAPACK dgetrs TRANS='N': dlaswp, then dtrsm(L, unit) and dtrsm(U, non-unit).
//   gesv    -- the two above, drop-in for dgesv with TRANS='N'.
//   gemm_nn -- reference BLAS dgemm, TRANSA=TRANSB='N', alpha=1, beta=0, in dgemm's own
//              j / l / i loop order so the summation order over l matches.
//
// MEASURED, AND IT DOES NOT REACH BIT-IDENTITY WITH ACCELERATE. Every entry of every k x k
// inverse over the whole corpus was compared as raw bits, not as %.12g text -- 122,671
// components, k = 2 .. 245, 81,235,616 doubles -- and 79,929,316 of them (98.4%) differ from
// dgesv$NEWLAPACK's in the last bits. Max absolute difference 9.5e-12. This is not a bug here
// and it is not fixable by matching reference LAPACK harder: on a 5-atom path graph the results
// of this header agree almost entirely with Accelerate's LEGACY `dgesv_`, and both differ from
// `dgesv$NEWLAPACK`, which is a blocked, vectorised, closed kernel whose summation order is not
// reproducible from the outside. The pivot sequences agree; only the rounding does not.
//
// WHY THAT IS TOLERABLE, and where it is not: see the long note above resistance() in
// hume_blocks.h. Short version -- the atom pairs that change bin sit EXACTLY on a bin edge in
// exact rational arithmetic, so no floating-point solver resolves them, and Accelerate's own two
// LAPACKs disagree with each other on MORE of the corpus (9.00%) than this header disagrees with
// either. The bins were never a property of the molecule alone.
//
// gemm_nn does far better: 511,294 of 609,332,690 entries (0.084%) differ from dgemm$NEWLAPACK,
// all in the last bit. Six RW columns move as a result, all on molecules whose value is
// essentially zero (worst 4.5e-17 ABSOLUTE, on 44 molecules) -- see the note at the mul() lambda
// in rw_returns. Compiling with -ffp-contract=off makes the disagreement fifty times worse (30.4M
// entries), which is how we know Accelerate's kernel contracts too -- so the default -O3
// contraction is deliberately not suppressed.
//
// Header-only C++17. No BLAS, no LAPACK, no intrinsics, no -march=native requirement.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace hume_lin {

// BLAS idamax: 0-based index of the FIRST element of largest absolute value. "First" is not a
// detail -- it decides the pivot when two candidates tie, and a different tie-break is a
// different factorisation.
inline int idamax(int n, const double *x) {
  if (n < 1) return 0;
  int imax = 0;
  double dmax = std::fabs(x[0]);
  for (int i = 1; i < n; i++) {
    const double v = std::fabs(x[i]);
    if (v > dmax) { dmax = v; imax = i; }
  }
  return imax;
}

// LAPACK dgetf2: A = P * L * U, column-major, in place. L is unit lower, U upper, and ipiv[j]
// is the 0-based row that row j was swapped with. Returns 0, or j+1 for the first exactly zero
// pivot -- the caller must check, exactly as it checked dgesv's INFO.
//
// THE RECIPROCAL BRANCH IS LAPACK'S, AND IT IS OBSERVABLE. dscal multiplies by 1/pivot rather
// than dividing by pivot, which is one rounding of the reciprocal followed by one of each
// product instead of one rounding per quotient; the two disagree in the last bit. LAPACK falls
// back to true division only when |pivot| < sfmin, where forming the reciprocal would overflow.
inline int getf2(int n, double *A, int lda, int *ipiv) {
  int info = 0;
  const double sfmin = std::numeric_limits<double>::min();   // dlamch('S')
  for (int j = 0; j < n; j++) {
    const int jp = j + idamax(n - j, &A[(std::size_t)j * lda + j]);
    ipiv[j] = jp;
    if (A[(std::size_t)j * lda + jp] != 0.0) {
      if (jp != j)                                            // dswap: WHOLE rows, all n columns
        for (int c = 0; c < n; c++)
          std::swap(A[(std::size_t)c * lda + j], A[(std::size_t)c * lda + jp]);
      if (j < n - 1) {
        const double piv = A[(std::size_t)j * lda + j];
        if (std::fabs(piv) >= sfmin) {
          const double r = 1.0 / piv;                          // dscal
          for (int i = j + 1; i < n; i++) A[(std::size_t)j * lda + i] *= r;
        } else {
          for (int i = j + 1; i < n; i++) A[(std::size_t)j * lda + i] /= piv;
        }
      }
    } else if (info == 0) {
      info = j + 1;
    }
    if (j < n - 1) {
      // dger, alpha = -1: A(j+1:, j+1:) -= A(j+1:, j) * A(j, j+1:). The `y != 0` guard is
      // reference dger's own; it skips a column rather than adding zeros to it.
      for (int c = j + 1; c < n; c++) {
        const double y = A[(std::size_t)c * lda + j];
        if (y == 0.0) continue;
        const double temp = -y;
        for (int i = j + 1; i < n; i++)
          A[(std::size_t)c * lda + i] += A[(std::size_t)j * lda + i] * temp;
      }
    }
  }
  return info;
}

// LAPACK dgetrs, TRANS='N': solve A X = B given the factors from getf2. B is overwritten by X,
// column-major, nrhs right-hand sides.
inline void getrs_n(int n, int nrhs, const double *A, int lda, const int *ipiv, double *B,
                    int ldb) {
  for (int i = 0; i < n; i++) {                                // dlaswp, forwards
    const int ip = ipiv[i];
    if (ip != i)
      for (int c = 0; c < nrhs; c++)
        std::swap(B[(std::size_t)c * ldb + i], B[(std::size_t)c * ldb + ip]);
  }
  for (int c = 0; c < nrhs; c++) {
    double *b = &B[(std::size_t)c * ldb];
    // dtrsm SIDE='L' UPLO='L' TRANSA='N' DIAG='U', alpha = 1 (so no scaling pass).
    for (int k = 0; k < n; k++) {
      if (b[k] == 0.0) continue;
      for (int i = k + 1; i < n; i++) b[i] -= b[k] * A[(std::size_t)k * lda + i];
    }
    // dtrsm SIDE='L' UPLO='U' TRANSA='N' DIAG='N'.
    for (int k = n - 1; k >= 0; k--) {
      if (b[k] == 0.0) continue;
      b[k] /= A[(std::size_t)k * lda + k];
      for (int i = 0; i < k; i++) b[i] -= b[k] * A[(std::size_t)k * lda + i];
    }
  }
}

// LAPACK dgesv: factor and solve. A is destroyed, B overwritten with the solution, ipiv must
// hold n ints. Returns dgesv's INFO.
inline int gesv(int n, int nrhs, double *A, int lda, int *ipiv, double *B, int ldb) {
  const int info = getf2(n, A, lda, ipiv);
  if (info == 0) getrs_n(n, nrhs, A, lda, ipiv, B, ldb);
  return info;
}

// Reference BLAS dgemm, TRANSA = TRANSB = 'N', alpha = 1, beta = 0: C = A * B, all n x n and
// column-major. The j / l / i nesting is dgemm's own, so the sum over l accumulates into C in
// the same order the library would use.
inline void gemm_nn(int n, const double *A, int lda, const double *B, int ldb, double *C,
                    int ldc) {
  for (int j = 0; j < n; j++) {
    double *c = &C[(std::size_t)j * ldc];
    for (int i = 0; i < n; i++) c[i] = 0.0;                    // beta = 0
    for (int l = 0; l < n; l++) {
      const double temp = B[(std::size_t)j * ldb + l];         // alpha = 1
      if (temp == 0.0) continue;
      const double *a = &A[(std::size_t)l * lda];
      for (int i = 0; i < n; i++) c[i] += temp * a[i];
    }
  }
}

}  // namespace hume_lin
