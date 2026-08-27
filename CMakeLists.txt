# Build for src/hume/_core, the pybind11 extension around the verified blocks.
#
# NO -march=native. It compiles for whatever core happens to be present, so the binary cannot
# ship; plain -O3 is what the verification harness is built with
# (`c++ -O3 -std=c++17 hume.cpp -framework Accelerate`) and matching it is what makes the
# bit-identity claim mean something. Portability and runtime dispatch are being measured
# separately -- see PACKAGING.md.
cmake_minimum_required(VERSION 3.15...3.31)
project(hume_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core src/hume_core/bindings.cpp)
target_compile_options(_core PRIVATE -O3)
target_include_directories(_core PRIVATE src/hume_core)

# LAPACK. This is the hard part of shipping, not an afterthought -- see PACKAGING.md.
#
# hume_blocks.h asks Accelerate for `dgesv$NEWLAPACK` and `dgemm$NEWLAPACK` BY NAME, via asm
# labels. Accelerate ships two LAPACKs that do not round the same way, and the resistance block
# was verified against the modern one (the bare `dgesv_` symbol resolves to Apple's legacy
# implementation, which returned 0.18749999999999994 where the new one returns 0.1875 exactly --
# the ulp that moved 94 atom pairs across a resistance bin edge). Those asm labels do not exist
# outside Accelerate, so a non-Apple build does not link today. Fixing that means picking a
# reference LAPACK and RE-VERIFYING resistance and BCUT2D against it, which is a separate piece
# of work from building the bridge; failing loudly here is better than producing a wheel whose
# numbers quietly differ from the ones the paper verified.
if(APPLE)
  target_link_libraries(_core PRIVATE "-framework Accelerate")
else()
  message(FATAL_ERROR
    "hume currently builds on macOS only: src/hume_core/hume_blocks.h binds Accelerate's "
    "dgesv$NEWLAPACK / dgemm$NEWLAPACK by asm label, and the resistance and BCUT2D blocks are "
    "verified against that specific LAPACK. See the LAPACK note in PACKAGING.md.")
endif()

install(TARGETS _core DESTINATION hume)
