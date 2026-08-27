// eigen_small.h -- extremal eigenvalues of a small dense real symmetric matrix, no BLAS.
//
// WHY THIS EXISTS. BCUT2D calls LAPACK four times per molecule (dsytd2 + dsterf). On macOS that
// resolves to Accelerate, on Linux to OpenBLAS or MKL, and the SAME HUME source measures
// 138.09 us/mol against 218.93 us/mol depending purely on which one it found -- a 1.6x swing in
// the block that is ~57% of the C++ descriptor time. A shipping package cannot have its headline
// number decided by the host's BLAS, and `-framework Accelerate` is not a portable build line.
//
// THE BET. At n ~ 27 (the corpus median heavy-atom count) LAPACK is mostly wrapper. This repo
// already measured that the UNBLOCKED dsytd2 beats the blocked dsyevd, because ILAENV lookups,
// workspace queries and a divide-and-conquer that falls through to QR below SMLSIZ cost more
// than the blocking saves. If that is true, the tuned BLAS underneath is not buying much either,
// and a self-contained reduction can match it -- while being byte-identical on every platform.
//
// WHAT IT IS. Householder tridiagonalisation (the LAPACK dsytd2 UPLO='U' algorithm, with the
// reference-BLAS dsymv/dsyr2 inner kernels written out) followed by implicit-shift QL/QR on the
// tridiagonal. Header-only C++17. No BLAS, no LAPACK, no intrinsics, no -march=native
// requirement. The arithmetic is the SAME arithmetic LAPACK does, in the same order, which is
// why it agrees with Accelerate to ~1e-15 relative rather than to "some tolerance".
//
// SCOPE. Eigenvalues only, and only the two extremes are returned -- that is all BCUT2D wants.
// The QL sweep still resolves the whole spectrum because deflation proceeds from the ends and
// an extremal-only stop would change which numbers come out; the sweep is O(n^2) against the
// reduction's O(n^3), so nothing is lost by finishing it.
//
// NOT WIRED INTO bcut2d. This is a candidate under measurement; see cpp/bench_eigen.cpp.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace hume_eig {

// Reusable scratch. ONLY EVER GROWS -- the same trap hume.cpp's BcutWork documents: a plain
// resize() shrinks and regrows as n varies molecule to molecule, and growing value-initialises
// the new tail, which is a per-call cost that has nothing to do with the solver being timed.
struct Work {
  std::vector<double> a, d, e, tau, wk;
  void ensure(int n) {
    const std::size_t nn = (std::size_t)n * (std::size_t)n;
    if (a.size() < nn) a.resize(nn);
    if ((int)d.size() < n) {
      d.resize(n);
      e.resize(n);
      tau.resize(n);
      wk.resize(n);
    }
  }
};

// LAPACK dlapy2: sqrt(x^2 + y^2) without spurious overflow. Not std::hypot -- hypot is
// correctly rounded and therefore slower, and LAPACK's own rounding is what we are matching.
inline double lapy2(double x, double y) {
  const double xa = std::fabs(x), ya = std::fabs(y);
  const double w = xa > ya ? xa : ya;
  const double z = xa > ya ? ya : xa;
  if (z == 0.0) return w;
  const double q = z / w;
  return w * std::sqrt(1.0 + q * q);
}

// LAPACK dlarfg, incx = 1. Generates H = I - tau*v*v' with H * (alpha, x)' = (beta, 0)'.
// On exit alpha holds beta and x holds v(2:n); v(1) is implicitly 1.
//
// The norm is the plain sum of squares rather than dnrm2's incremental scaling. A Burden matrix
// entry is a mass, a Crippen logP/MR contribution, a Gasteiger charge or 1/sqrt(bond order) --
// all within a couple of orders of 1 -- so the scaled path can never be reached; the guard below
// catches it anyway rather than assuming, and costs one predictable branch.
inline double larfg(int n, double &alpha, double *x) {
  if (n <= 1) return 0.0;
  const int m = n - 1;
  double ssq = 0.0;
  for (int k = 0; k < m; k++) ssq += x[k] * x[k];
  double xnorm;
  if (ssq > 0.0 && std::isfinite(ssq)) {
    xnorm = std::sqrt(ssq);
  } else {
    // Scaled fallback: only reachable on input this library will never see from a Burden
    // matrix, but a header that claims to be general must not silently return garbage.
    double scale = 0.0;
    for (int k = 0; k < m; k++) scale = std::max(scale, std::fabs(x[k]));
    if (scale == 0.0) return 0.0;
    double s = 0.0;
    for (int k = 0; k < m; k++) { const double t = x[k] / scale; s += t * t; }
    xnorm = scale * std::sqrt(s);
  }
  if (xnorm == 0.0) return 0.0;

  double beta = -std::copysign(lapy2(alpha, xnorm), alpha);
  // LAPACK rescales when |beta| < safmin so that tau does not lose all precision. Kept for the
  // same reason as the norm guard: unreachable here, cheap, and the difference between a
  // general routine and one that happens to work on our data.
  const double safmin = std::numeric_limits<double>::min() /
                        std::numeric_limits<double>::epsilon();
  if (std::fabs(beta) < safmin) {
    const double rsafmn = 1.0 / safmin;
    int cnt = 0;
    do {
      cnt++;
      for (int k = 0; k < m; k++) x[k] *= rsafmn;
      beta *= rsafmn;
      alpha *= rsafmn;
    } while (std::fabs(beta) < safmin && cnt < 20);
    double s2 = 0.0;
    for (int k = 0; k < m; k++) s2 += x[k] * x[k];
    xnorm = std::sqrt(s2);
    beta = -std::copysign(lapy2(alpha, xnorm), alpha);
    const double tau = (beta - alpha) / beta;
    const double sc = 1.0 / (alpha - beta);
    for (int k = 0; k < m; k++) x[k] *= sc;
    for (int k = 0; k < cnt; k++) beta *= safmin;
    alpha = beta;
    return tau;
  }
  const double tau = (beta - alpha) / beta;
  const double sc = 1.0 / (alpha - beta);
  for (int k = 0; k < m; k++) x[k] *= sc;
  alpha = beta;
  return tau;
}

// y := alpha * A * x, A symmetric n x n, UPPER triangle, column-major, beta = 0.
// This is reference dsymv's loop order verbatim: one pass over each column j accumulates both
// the contribution of x[j] to y[0..j-1] and the dot product of column j with x[0..j-1]. One
// traversal of the triangle, both strides unit, and the same summation order LAPACK gets.
inline void symv_upper(int n, double alpha, const double *A, int lda, const double *x,
                       double *y) {
  for (int k = 0; k < n; k++) y[k] = 0.0;
  for (int j = 0; j < n; j++) {
    const double *col = A + (std::size_t)j * lda;
    const double t1 = alpha * x[j];
    double t2 = 0.0;
    for (int k = 0; k < j; k++) {
      y[k] += t1 * col[k];
      t2 += col[k] * x[k];
    }
    y[j] += t1 * col[j] + alpha * t2;
  }
}

// A := A + alpha*(x*y' + y*x'), UPPER triangle only, column-major. Reference dsyr2.
inline void syr2_upper(int n, double alpha, const double *x, const double *y, double *A,
                       int lda) {
  for (int j = 0; j < n; j++) {
    double *col = A + (std::size_t)j * lda;
    const double t1 = alpha * y[j];
    const double t2 = alpha * x[j];
    for (int k = 0; k <= j; k++) col[k] += x[k] * t1 + y[k] * t2;
  }
}

// LAPACK dsytd2, UPLO='U': reduce A to symmetric tridiagonal by orthogonal similarity.
// Works from the bottom-right corner up, so the trailing submatrix shrinks and the O(i^2) inner
// work falls away -- 4/3 n^3 flops total, with no blocking, which is the point: at n ~ 27 a
// blocked dsytrd has nothing to block and pays ILAENV to find that out.
//
// A is destroyed. d[0..n-1] gets the diagonal, e[0..n-2] the off-diagonal.
inline void sytd2_upper(double *A, int n, int lda, double *d, double *e, double *tau,
                        double *wk) {
  for (int ii = n - 2; ii >= 0; --ii) {
    double *col = A + (std::size_t)(ii + 1) * lda;   // column ii+1
    double alpha = col[ii];
    const double taui = larfg(ii + 1, alpha, col);   // annihilates col[0..ii-1]
    e[ii] = alpha;
    if (taui != 0.0) {
      col[ii] = 1.0;
      const int m = ii + 1;
      symv_upper(m, taui, A, lda, col, wk);          // wk := tau * A * v
      double dot = 0.0;
      for (int k = 0; k < m; k++) dot += wk[k] * col[k];
      const double a2 = -0.5 * taui * dot;
      for (int k = 0; k < m; k++) wk[k] += a2 * col[k];
      syr2_upper(m, -1.0, col, wk, A, lda);          // A := A - v*w' - w*v'
      col[ii] = e[ii];
    }
    d[ii + 1] = A[(std::size_t)(ii + 1) * lda + (ii + 1)];
    tau[ii] = taui;
  }
  d[0] = A[0];
}

// LAPACK dlae2: both eigenvalues of the 2x2 [[a,b],[b,c]], computed so that the small root
// comes from the product rather than from a cancelling difference. Used for genuine 2x2 inputs
// AND for the last block of every QL/QR sweep, which is where dsteqr uses it too -- skipping it
// and letting the sweep grind the 2x2 down loses digits on the near-degenerate pairs that
// high-symmetry cages (cubane, prismane, adamantane) are full of.
inline void lae2(double a, double b, double c, double *rt1, double *rt2) {
  const double sm = a + c, df = a - c, adf = std::fabs(df);
  const double tb = b + b, ab = std::fabs(tb);
  double acmx, acmn;
  if (std::fabs(a) > std::fabs(c)) { acmx = a; acmn = c; } else { acmx = c; acmn = a; }
  double rt;
  if (adf > ab) { const double q = ab / adf; rt = adf * std::sqrt(1.0 + q * q); }
  else if (adf < ab) { const double q = adf / ab; rt = ab * std::sqrt(1.0 + q * q); }
  else rt = ab * std::sqrt(2.0);
  if (sm < 0.0) {
    *rt1 = 0.5 * (sm - rt);
    *rt2 = (acmx / *rt1) * acmn - (b / *rt1) * b;
  } else if (sm > 0.0) {
    *rt1 = 0.5 * (sm + rt);
    *rt2 = (acmx / *rt1) * acmn - (b / *rt1) * b;
  } else {
    *rt1 = 0.5 * rt;
    *rt2 = -0.5 * rt;
  }
}

// LAPACK dlartg: the plane rotation [c s; -s c] * (f, g)' = (r, 0)'. The zero cases are not
// cosmetic -- they are how the sweep survives an exactly-deflated off-diagonal without a
// divide by zero, which is the failure mode a naive r = hypot(f,g); c = f/r costs you.
inline void lartg(double f, double g, double *cs, double *sn, double *r) {
  if (g == 0.0) { *cs = 1.0; *sn = 0.0; *r = f; return; }
  if (f == 0.0) { *cs = 0.0; *sn = 1.0; *r = g; return; }
  *r = lapy2(f, g);
  *cs = f / *r;
  *sn = g / *r;
  if (std::fabs(f) > std::fabs(g) && *cs < 0.0) { *cs = -*cs; *sn = -*sn; *r = -*r; }
}

// Implicit-shift QL/QR on a symmetric tridiagonal, eigenvalues only -- the dsterf job, written
// out as LAPACK's dsteqr performs it with ICOMPZ = 0.
//
// Direction is chosen per unreduced block the way LAPACK does: QL when the small-|d| end is at
// the bottom, QR otherwise, so the Wilkinson shift chases the end that is already nearly
// converged. A single-direction sweep gets the same eigenvalues but takes measurably more
// iterations on the Burden spectra, which are strongly graded -- a mass diagonal runs from 12
// to 127 while the off-diagonals sit at 0.001 to 1.
//
// The split test is dsteqr's: e^2 <= eps^2 * |d_m| * |d_{m+1}| + safmin. A bare
// |e| <= eps*(|d_m| + |d_{m+1}|) -- the textbook/Numerical-Recipes test -- is WRONG for this
// application: the Gasteiger-charge Burden matrix has a diagonal of order 0.1 against
// off-diagonals fixed at 0.001, and on a near-zero diagonal a relative test with no absolute
// floor never terminates. That floor is why safmin is in there and not decoration.
//
// NOT IMPLEMENTED, deliberately: dsteqr's per-block rescaling to [sqrt(safmin), sqrt(safmax)].
// It guards against overflow in the rotations for spectra spanning ~1e150, which a matrix of
// atomic masses, logP contributions and Gasteiger charges cannot produce. The convergence
// counter below turns any surprise into a false return rather than a wrong number.
//
// Returns false if a block fails to converge within 30n sweeps.
inline bool sterf_min_max(int n, double *d, double *e, double *lo, double *hi) {
  if (n <= 0) return false;
  if (n == 1) { *lo = *hi = d[0]; return true; }

  const double eps = std::numeric_limits<double>::epsilon();
  const double eps2 = eps * eps;
  const double safmin = std::numeric_limits<double>::min();
  const int maxit = 30 * n;
  int iter = 0;
  double c, s, r, g, p, f, b, rt1, rt2;

  int l1 = 0;
  while (l1 < n) {
    if (l1 > 0) e[l1 - 1] = 0.0;
    int mblk = l1;
    while (mblk < n - 1) {
      const double t = std::fabs(e[mblk]);
      if (t * t <= (eps2 * std::fabs(d[mblk])) * std::fabs(d[mblk + 1]) + safmin) break;
      mblk++;
    }
    int l = l1, lend = mblk;
    l1 = mblk + 1;
    if (l == lend) continue;

    // Orient the block so the sweep deflates at the SMALL-|d| end: QL (indices increase) if
    // that end is already at the top, QR (indices decrease) after the swap if it is at the
    // bottom. This is dsteqr's choice, and on a mass-diagonal Burden matrix -- 12 at carbon,
    // 127 at iodine -- getting it backwards costs iterations, not accuracy.
    if (std::fabs(d[lend]) < std::fabs(d[l])) std::swap(l, lend);
    const bool ql = l < lend;

    while (l != -1) {
      if (ql) {
        if (l > lend) break;
        int m = l;
        while (m < lend) {
          const double t = std::fabs(e[m]);
          if (t * t <= (eps2 * std::fabs(d[m])) * std::fabs(d[m + 1]) + safmin) break;
          m++;
        }
        if (m < lend) e[m] = 0.0;
        p = d[l];
        if (m == l) { l++; continue; }
        if (m == l + 1) {
          lae2(d[l], e[l], d[l + 1], &rt1, &rt2);
          d[l] = rt1; d[l + 1] = rt2; e[l] = 0.0;
          l += 2;
          continue;
        }
        if (++iter > maxit) return false;
        g = (d[l + 1] - p) / (2.0 * e[l]);
        r = lapy2(g, 1.0);
        g = d[m] - p + (e[l] / (g + std::copysign(r, g)));
        s = 1.0; c = 1.0; p = 0.0;
        for (int i = m - 1; i >= l; --i) {
          f = s * e[i];
          b = c * e[i];
          lartg(g, f, &c, &s, &r);
          if (i != m - 1) e[i + 1] = r;
          g = d[i + 1] - p;
          r = (d[i] - g) * s + 2.0 * c * b;
          p = s * r;
          d[i + 1] = g + p;
          g = c * r - b;
        }
        e[l] = g;
        d[l] = d[l] - p;
      } else {
        if (l < lend) break;
        int m = l;
        while (m > lend) {
          const double t = std::fabs(e[m - 1]);
          if (t * t <= (eps2 * std::fabs(d[m])) * std::fabs(d[m - 1]) + safmin) break;
          m--;
        }
        if (m > lend) e[m - 1] = 0.0;
        p = d[l];
        if (m == l) { l--; continue; }
        if (m == l - 1) {
          lae2(d[l - 1], e[l - 1], d[l], &rt1, &rt2);
          d[l - 1] = rt1; d[l] = rt2; e[l - 1] = 0.0;
          l -= 2;
          continue;
        }
        if (++iter > maxit) return false;
        g = (d[l - 1] - p) / (2.0 * e[l - 1]);
        r = lapy2(g, 1.0);
        g = d[m] - p + (e[l - 1] / (g + std::copysign(r, g)));
        s = 1.0; c = 1.0; p = 0.0;
        for (int i = m; i <= l - 1; ++i) {
          f = s * e[i];
          b = c * e[i];
          lartg(g, f, &c, &s, &r);
          if (i != m) e[i - 1] = r;
          g = d[i] - p;
          r = (d[i + 1] - g) * s + 2.0 * c * b;
          p = s * r;
          d[i] = g + p;
          g = c * r - b;
        }
        e[l - 1] = g;
        d[l] = d[l] - p;
      }
    }
  }

  double mn = d[0], mx = d[0];
  for (int i = 1; i < n; i++) {
    if (d[i] < mn) mn = d[i];
    if (d[i] > mx) mx = d[i];
  }
  *lo = mn;
  *hi = mx;
  return true;
}

// ------------------------------------------------------------------ the one entry point
//
// Extremal eigenvalues of the n x n real symmetric matrix A. For a symmetric matrix column-major
// and row-major are the same bytes, so a row-major caller needs no transpose. A is COPIED, not
// destroyed. Returns false only if the QL/QR sweep fails to converge, which no Burden matrix in
// the 98,905-molecule corpus or the adversarial set has done.
inline bool extremal(const double *A, int n, int lda, double *lo, double *hi, Work &W) {
  if (n <= 0) return false;
  if (n == 1) { *lo = *hi = A[0]; return true; }
  if (n == 2) {
    double rt1, rt2;
    lae2(A[0], A[(std::size_t)lda], A[(std::size_t)lda + 1], &rt1, &rt2);
    *hi = rt1 > rt2 ? rt1 : rt2;
    *lo = rt1 > rt2 ? rt2 : rt1;
    return true;
  }
  W.ensure(n);
  double *M = W.a.data();
  // Pack into a contiguous n x n. Only the upper triangle is read by sytd2_upper.
  for (int j = 0; j < n; j++)
    for (int k = 0; k <= j; k++) M[(std::size_t)j * n + k] = A[(std::size_t)j * lda + k];
  sytd2_upper(M, n, n, W.d.data(), W.e.data(), W.tau.data(), W.wk.data());
  return sterf_min_max(n, W.d.data(), W.e.data(), lo, hi);
}

// Convenience overload for a packed n x n with lda == n.
inline bool extremal(const double *A, int n, double *lo, double *hi, Work &W) {
  return extremal(A, n, n, lo, hi, W);
}

}  // namespace hume_eig
