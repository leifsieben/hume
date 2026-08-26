#!/bin/sh
# Night queue. Sequential, not concurrent: UMA collapses past ~6 workers, so two
# 6-worker jobs would contend. Total expected ~2h.
set -x
# 1) descriptor targets — ChemTFM venv has mordred + pinned rdkit 2025.9.2 + chemtfm
PYTHONPATH=/Users/lsieben/VSCode/ChemTFM_OLD \
  /Users/lsieben/VSCode/ChemTFM_OLD/.venv/bin/python build_targets.py --workers 10
# 2) UMA embeddings — .venv-uma has fairchem 2.21, where the embedding hook still fires
#    (2.22 torch.compiles the graph and the hook never sees node_embedding).
#    The embed path reads pickled Z/R/charge and never calls RDKit, so that venv's
#    newer RDKit cannot affect canonical-SMILES keys here.
/Users/lsieben/VSCode/ChemTFM_OLD/.venv-uma/bin/python uma_100k.py embed --workers 6
