// Standalone harness for src/hume_core/frag_matcher.h, in the shape of cpp/estate_typer.cpp:
// read a flat dump of the boundary columns, print one line of counts per molecule, and let
// cpp/verify_frag.py compare against the pinned RDKit.  Nothing here is wired into the
// extension -- see the report for the wiring instructions.
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
// `tval` is RDKit's GetTotalValence().  It is the one column the (n_atoms, 9) boundary does not
// carry and cannot derive -- see the header comment in frag_matcher.h for the 11,238-atom
// counterexample.
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
