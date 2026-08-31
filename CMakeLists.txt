# Build for src/molhume/_core, the pybind11 extension around the verified blocks.
#
# NO -march=native. It compiles for whatever core happens to be present, so the binary cannot
# ship; plain -O3 is what the verification harness is built with
# (`c++ -O3 -std=c++17 hume.cpp`) and matching it is what makes the bit-identity claim mean
# something. It also costs nothing to leave out: on arm64 the two flags produce BYTE-IDENTICAL
# object files for this source, re-checked after the BLAS removal -- see PACKAGING.md.
cmake_minimum_required(VERSION 3.15...3.31)
project(hume_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core src/hume_core/bindings.cpp)
target_compile_options(_core PRIVATE -O3)
target_include_directories(_core PRIVATE src/hume_core)

# NO BLAS, NO LAPACK, NO PLATFORM CHECK -- and that is the point of this file's history.
#
# This build used to REFUSE to configure off macOS. src/hume_core/hume_blocks.h asked Accelerate
# for `dgesv$NEWLAPACK` and `dgemm$NEWLAPACK` by asm label, because Accelerate ships two LAPACKs
# that do not round the same way and the resistance block had been verified against the modern
# one. Those symbols exist in no other library, so failing loudly was better than shipping a
# wheel whose numbers quietly differed. The cost was that HUME was a macOS-only package.
#
# Both call sites are now gone, replaced by two header-only implementations with no external
# dependency at all: cpp/eigen_small.h (BCUT2D's extremal eigenvalues) and cpp/lu_small.h
# (resistance's per-component solve and rw_returns' matrix powers). There is no longer any
# `target_link_libraries` here, and no `if(APPLE)`.
#
# WHAT CHANGED NUMERICALLY, because "portable now" is not the same as "same numbers":
#   * BCUT2D moved in the last bits on 795 of 98,905 molecules, max 1.9e-10 relative -- inside
#     the rtol 1e-9 the block is graded at.
#   * The resistance BIN columns (RATSC*/RPAIR*) moved on up to 8.1% of molecules. Those
#     columns turn out not to be well defined in floating point at all: exact rational
#     arithmetic shows the flipping pairs sit EXACTLY on a bin edge, and Accelerate's own two
#     LAPACKs disagree with each other on 9.00% of the corpus -- more than reference LU
#     disagrees with either. See the note above resistance() in hume_blocks.h.
# Neither of those is caused by the build system, but this is where the dependency was declared
# and this is where someone will come looking.
#
# PROVEN, not assumed. Both of these configure and compile clean, and the resulting extension
# links no BLAS (checked with otool -L / ldd):
#   cmake -S . -B build/nolapack -DCMAKE_DISABLE_FIND_PACKAGE_BLAS=ON \
#         -DCMAKE_DISABLE_FIND_PACKAGE_LAPACK=ON -DCMAKE_FIND_FRAMEWORK=NEVER \
#         -DCMAKE_SYSTEM_FRAMEWORK_PATH= -DCMAKE_OSX_SYSROOT=
#   c++ -O3 -std=c++17 -o hume cpp/hume.cpp     # no -framework Accelerate
#
# The two headers live under cpp/ and are pulled in by relative path from hume_blocks.h, so no
# include directory is needed here -- but they MUST be present in the sdist. Both are tracked by
# git, which is what scikit-build-core uses to decide what an sdist contains.

install(TARGETS _core DESTINATION molhume)
