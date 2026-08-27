// Standalone harness for src/hume_core/frag_matcher.h, in the shape of cpp/estate_typer.cpp:
// read a flat dump of the boundary columns, print one line of counts per molecule, and let
// cpp/verify_frag.py compare against the pinned RDKit.
//
// THE HEADER IS NOW WIRED INTO THE EXTENSION and this file is no longer the only way to reach it:
// bindings.cpp's all_row() fills the same fragmatch::Mol from the boundary arrays.  This harness
// stays because it grades the MATCHER on a graph built independently of that wiring, and because
// it is the file the "ALL EXACT on 100,000" claim was made through.  Note it can no longer be the
// wiring's oracle: cpp/verify_wiring.py grades the 76 columns against RDKit's own `Descriptors`
// in-process, because this binary and the wiring share frag_matcher.h and could only agree.
//
//   c++ -O3 -std=c++17 -o cpp/frag cpp/frag.cpp
//   ./cpp/frag < dump.txt > counts.txt
//
// DUMP FORMAT (all integers, whitespace separated):
//   nmol
//   repeated nmol times:
//     n nb
//     n rows of:  z deg nH fchg arom nring tval
//     nb rows of: u v border bring
//
// `tval` is RDKit's GetTotalValence().  It is the column that made the boundary (n_atoms, 10):
// it cannot be derived -- see the header comment in frag_matcher.h for the 11,238-atom
// counterexample -- so it is carried, as `Atom.GetTotalValence()` on the reference path and as
// the pickle's own explicit + implicit valence on the fast one.
#include <cstdio>
#include <vector>

#include "../src/hume_core/frag_matcher.h"

int main() {
  int nmol;
  if (std::scanf("%d", &nmol) != 1) return 1;
  std::vector<int> out(frag_prog::N_NAMED);
  // header line: the column names, so the comparison cannot silently mis-align
  for (int i = 0; i < frag_prog::N_NAMED; ++i)
    std::printf("%s%s", i ? "\t" : "#", frag_prog::NAMED[i].name);
  std::printf("\tNHOHCount\tHeavyAtomCount\n");
  for (int k = 0; k < nmol; ++k) {
    int n, nb;
    if (std::scanf("%d %d", &n, &nb) != 2) return 1;
    fragmatch::Mol m;
    m.alloc(n, nb);
    for (int i = 0; i < n; ++i)
      if (std::scanf("%d %d %d %d %d %d %d", &m.z[i], &m.deg[i], &m.nH[i], &m.fchg[i],
                     &m.arom[i], &m.nring[i], &m.tval[i]) != 7) return 1;
    for (int e = 0; e < nb; ++e)
      if (std::scanf("%d %d %d %d", &m.bu[e], &m.bv[e], &m.border[e], &m.bring[e]) != 4) return 1;
    m.finish();
    fragmatch::countAll(m, out.data());
    for (int i = 0; i < frag_prog::N_NAMED; ++i)
      std::printf("%s%d", i ? "\t" : "", out[i]);
    std::printf("\t%d\t%d\n", fragmatch::nhohCount(m), fragmatch::heavyAtomCount(m));
  }
  return 0;
}
