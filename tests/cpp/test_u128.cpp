// Does the two-limb fallback agree with `unsigned __int128` exactly? Compile the fallback under
// a different namespace by hiding __SIZEOF_INT128__ from it, then compare on the operations
// MidWalk actually performs: multiply-by-u64, compare against LIM, and convert to double.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <random>

#define __SIZEOF_INT128__ __SIZEOF_INT128__
namespace native { using u128 = unsigned __int128;
  inline double to_double(u128 v) { return (double)v; }
  constexpr u128 make(uint64_t h, uint64_t l) { return ((u128)h << 64) | (u128)l; } }

#undef __SIZEOF_INT128__
#define hume fallback
#include "../../src/hume_core/u128.h"
#undef hume

int main() {
  std::mt19937_64 rng(12345);
  const native::u128 nLIM = native::make(5ULL, 7766279631452225536ULL);
  const fallback::u128 fLIM = fallback::u128_make(5ULL, 7766279631452225536ULL);
  long long n = 0, bad_mul = 0, bad_cmp = 0, bad_dbl = 0, bad_ulp = 0;

  for (long long t = 0; t < 4000000; ++t) {
    // shapes MidWalk produces: an accumulator up to ~1e20+ and a small multiplier
    const int bits = 1 + (int)(rng() % 90);
    const uint64_t hi = bits > 64 ? (rng() >> (128 - bits)) : 0;
    const uint64_t lo = rng();
    const uint64_t mul = 1 + (rng() % 81);
    native::u128 na = native::make(hi, lo);
    fallback::u128 fa = fallback::u128_make(hi, lo);

    native::u128 nb = na * (native::u128)mul;
    fallback::u128 fb = fa * mul;
    if ((uint64_t)nb != fb.lo || (uint64_t)(nb >> 64) != fb.hi) { ++bad_mul; continue; }
    if ((nb < nLIM) != (fb < fLIM)) ++bad_cmp;
    const double nd = native::to_double(nb), fd = fallback::to_double(fb);
    if (nd != fd) { ++bad_dbl; if (std::abs(nd - fd) > std::abs(nd) * 1e-16) ++bad_ulp; }
    ++n;
  }
  printf("  %lld cases\n", n);
  printf("  multiply differs : %lld\n", bad_mul);
  printf("  compare  differs : %lld\n", bad_cmp);
  printf("  to_double differs: %lld  (of which >1ulp: %lld)\n", bad_dbl, bad_ulp);
  // and the two static_assert digits
  printf("  LIM/1e10 = %llu  LIM%%1e10 = %llu\n",
         (unsigned long long)fallback::u128_low(fLIM / 10000000000ULL),
         (unsigned long long)(fLIM % 10000000000ULL));
  return (bad_mul || bad_cmp || bad_dbl) ? 1 : 0;
}
