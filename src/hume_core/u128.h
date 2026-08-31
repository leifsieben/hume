#pragma once
//! A 128-bit unsigned integer that exists on MSVC.
//!
//! `MolecularId` (misc_ext.h, S9) walks paths accumulating a product of per-half-edge degree
//! products and stops when it reaches `int(1.0 / 1e-10**2)` ~ 1e20. The running product can
//! reach `lim * 81 > 2^63`, so it needs more than 64 bits. gcc and clang have
//! `unsigned __int128`; MSVC does not, and there is no `/std:` flag that adds it.
//!
//! So: `unsigned __int128` where it exists, and a two-limb struct where it does not. The
//! native path is unchanged and still produces the verified values bit for bit -- this header
//! adds a fallback, it does not replace the type that was measured.
//!
//! WHAT THE FALLBACK HAS TO GET EXACTLY RIGHT is the conversion to double, not the arithmetic.
//! Truncating multiply and unsigned compare are exact in any correct implementation. The
//! conversion is not: `(double)hi * 2^64 + (double)lo` rounds twice and can land one ulp away
//! from the single correctly-rounded conversion the native type performs, and the result feeds
//! `1.0 / sqrt(nw)` into a descriptor graded at rtol 1e-9. `to_double` below therefore rounds
//! once, by normalizing to 64 significant bits with a sticky bit and letting the hardware's
//! own u64 -> double conversion do the rounding.

#include <cmath>
#include <cstdint>

namespace hume {

#if defined(__SIZEOF_INT128__)

using u128 = unsigned __int128;

constexpr u128 u128_make(uint64_t hi, uint64_t lo) { return ((u128)hi << 64) | (u128)lo; }
inline double to_double(u128 v) { return (double)v; }

#else  // MSVC and anything else without __int128

struct u128 {
  uint64_t lo = 0, hi = 0;

  constexpr u128() = default;
  constexpr u128(uint64_t v) : lo(v), hi(0) {}
  constexpr u128(uint64_t h, uint64_t l) : lo(l), hi(h) {}
};

constexpr u128 u128_make(uint64_t hi, uint64_t lo) { return u128(hi, lo); }

constexpr bool operator<(u128 a, u128 b) { return a.hi != b.hi ? a.hi < b.hi : a.lo < b.lo; }
constexpr bool operator==(u128 a, u128 b) { return a.hi == b.hi && a.lo == b.lo; }

//! 64 x 64 -> 128, by 32-bit halves. MSVC has `_umul128`, but only on x64, and this is not hot.
constexpr u128 u128_mul64(uint64_t a, uint64_t b) {
  const uint64_t a0 = a & 0xFFFFFFFFULL, a1 = a >> 32;
  const uint64_t b0 = b & 0xFFFFFFFFULL, b1 = b >> 32;
  const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  const uint64_t mid = (p00 >> 32) + (p01 & 0xFFFFFFFFULL) + (p10 & 0xFFFFFFFFULL);
  return u128(p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32),
              (mid << 32) | (p00 & 0xFFFFFFFFULL));
}

//! 128 x 64 -> 128, truncating on overflow -- which is what the native type does.
constexpr u128 operator*(u128 a, uint64_t b) {
  u128 r = u128_mul64(a.lo, b);
  r.hi += a.hi * b;
  return r;
}
constexpr u128 operator*(u128 a, u128 b) { return a * b.lo; }  // callers only ever scale by u64

//! Long division by a 64-bit divisor. constexpr because the only callers are static_asserts.
constexpr u128 u128_divmod(u128 a, uint64_t d, uint64_t *rem) {
  uint64_t r = 0, qh = 0, ql = 0;
  for (int i = 127; i >= 0; --i) {
    const uint64_t bit = i >= 64 ? (a.hi >> (i - 64)) & 1ULL : (a.lo >> i) & 1ULL;
    r = (r << 1) | bit;
    uint64_t q = 0;
    if (r >= d) { r -= d; q = 1; }
    if (i >= 64) qh |= q << (i - 64); else ql |= q << i;
  }
  if (rem) *rem = r;
  return u128(qh, ql);
}
constexpr u128 operator/(u128 a, uint64_t d) { uint64_t r = 0; return u128_divmod(a, d, &r); }
constexpr uint64_t operator%(u128 a, uint64_t d) {
  uint64_t r = 0;
  u128_divmod(a, d, &r);
  return r;
}

constexpr uint64_t u128_low(u128 a) { return a.lo; }

//! Correctly rounded, by rounding ONCE. See the header note.
inline double to_double(u128 v) {
  if (v.hi == 0) return (double)v.lo;             // one exact hardware conversion
  int lz = 0;
  for (uint64_t probe = v.hi; !(probe & 0x8000000000000000ULL); probe <<= 1) ++lz;
  const uint64_t top = (v.hi << lz) | (lz ? (v.lo >> (64 - lz)) : 0ULL);
  const uint64_t rest = lz ? (v.lo << lz) : v.lo;
  // `top` holds 64 significant bits and double keeps 53, so the low 11 are discarded by the
  // conversion. OR-ing the dropped remainder into the lowest of them makes the single
  // conversion round the way the full 128-bit value would: a tie is only a tie when nothing
  // was dropped. OR, not assign -- `(top & ~1) | sticky` would silently subtract one from any
  // odd `top` with an empty remainder, which is a 1-in-2^11 wrong answer, not a rounding
  // difference. The differential test against `unsigned __int128` found exactly that.
  const uint64_t mant = top | (rest != 0 ? 1ULL : 0ULL);
  return std::ldexp((double)mant, 64 - lz);
}

#endif

#if defined(__SIZEOF_INT128__)
constexpr uint64_t u128_low(u128 a) { return (uint64_t)a; }
#endif

}  // namespace hume
