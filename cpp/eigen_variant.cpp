// eigen_variant.cpp -- one translation unit of eigen_small.h, compiled three times with three
// different -march/-mcpu settings and linked into ONE binary.
//
// WHY NOT THREE BINARIES. The question is what `-march=native` buys, and the answer this repo
// needs is worth a few percent. Three separate process invocations cannot resolve a few percent
// on this machine -- hume.cpp records the SAME unchanged code reading 122.7 to 136.0 us/mol
// across builds, and during these runs the box carried a load average of 28 on 12 cores. One
// binary holding all three, alternated within a cycle, can.
//
// WHY THE NAMESPACE WRAPPER. eigen_small.h is header-only, so its functions have inline linkage
// and identical mangled names in every TU. Compile the same header three ways and the linker is
// free to keep whichever COMDAT copy it saw first and discard the other two -- you would then
// "measure" three builds and time one. Wrapping each TU's include in a distinct namespace gives
// the three copies distinct symbols so all three actually survive into the binary. This is a
// harness concern only; nothing in the shipped header changes.
//
// WHAT IT FOUND, so the next person does not repeat the build. On arm64 macOS all three flag
// sets resolve to -target-cpu apple-m1 with an identical 27-feature set -- plain -O3 already
// targets apple-m1, and `native` on an M4 Pro falls back to apple-m1 because clang does not
// know the part. Compiled with identical symbol names, the three objects are BYTE-IDENTICAL
// (same md5). -march=native is a literal no-op here, so HUME can drop it from the build line
// on this platform at zero cost, and `march` mode is really a noise-floor calibration for the
// rest of the harness. None of this transfers to x86-64, where native unlocks AVX2/AVX-512.
//
// Build (see bench_eigen's `march` mode):
//   clang++ -O3                  -DVNS=v_plain   -DVFN=eig_plain   -c eigen_variant.cpp -o vp.o
//   clang++ -O3 -mcpu=apple-m1   -DVNS=v_m1      -DVFN=eig_m1      -c eigen_variant.cpp -o vm.o
//   clang++ -O3 -march=native    -DVNS=v_native  -DVFN=eig_native  -c eigen_variant.cpp -o vn.o

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace VNS {
#include "eigen_small.h"
}

extern "C" bool VFN(const double *A, int n, double *lo, double *hi) {
  static thread_local VNS::hume_eig::Work W;
  return VNS::hume_eig::extremal(A, n, lo, hi, W);
}
