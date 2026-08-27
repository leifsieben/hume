# High-throughput Universal Molecular Embeddings (HUME)

This is an opinionated attempt at a solution for a dire need in the cheminformatics community: how do you actually represent a molecule well? We've seen time and time again that XGBoost on fingerprints is a nearly unbeatable baseline. In the cases where it seemingly gets beat -- like CheMeleon -- you find ex post facto that you could've replaced CheMeleon with just the descriptors it was pretrained on. We think this comes down to two postulates a good molecular embeddig must fulfil: 

1) Resolution: It must resolve any chemical change, including stereochemistry, ring number, isotopes, scaffold editing. All the DL embeddings we're aware of actually mush together molecules meaning structure isn't actually resolvable, even though activity cliffs for example routinely form on just a stereocenter. 
2) Expressiveness: Embedding distance should reflect chemical similarities across many properties both structural and physico-chemical (arguably even 3D and 4D properties). Descriptors help massively in this and DL models can be helpful in predicting these. Now descriptors can always be calculated from the SMILES and this remains the upper bound for performance for these DL models, although they are often faster than the sometimes tedious computations. 

This suggests two things: a) from all we've seen ECFP (with stereochem on) is close to optimal in terms of structure. It is expressive while being nearly perfectly expressive in terms of structure. What it lacks is PChem and here a good embedding finds some headspace. However not all descriptors are alike: ECFP requires constructing a mol object anyway, so no overhead for that, and quite a number of descriptors can be exactly computed from it (or a fast to compute input) in microseconds -- you get these for free. 

For the rest, the ones that actually drive the myth of "descriptors being slow", these indeed should be predicted. But if this is your problem setup, surely, you'd build a model different than CheMeleon? Here we try and find the fastest and most accurate model to predict the slow predictors we can so that even in a VS of billions of molecules the HUMEs are feasible to compute. Feeding these into XGBoost should give results comparable to ECFP+RDKit+Mordred descriptors, the best embedding we are aware of, at a fraction of the cost. 

## Potential downsides
Is it a problem to mix smooth and discrete/sparse inputs? Not for XGBoost I think. 
Is the length of the input a problem? Could/should we compress it further down to fewer dimensions? But why? XGboost isn't limited by this, is it? 

## Results

Here we need to look at other models: CLIMB, smi-ted, CheMeleoen, Grover, Chemprop, MiniMol vs ECFP and Heather Kulik's new fingerprint as well. 

In terms of actual figures to show in the paper 
- A) Resolution: Show 1000 molecules each with some small chemical change (stereochem flipped, ring substitued, scaffold edit, isotope etc) and show that ECFP resolves nearly all of them perfectly while the other models typically fail. Also show some negative results, i.e. very similar descriptors --> can structure still be resolved. 
- B) Showing redundancy. Either measured as downstream performance on some datasets or maybe we can come up with some other setup: test how redundant DL embeddings are to ECFP, RDkit desc, Mordred desc --> do any of these contain information that is useful + not contained in these 3 classical sources. 

From these two define best CLM and best GNN. 

- C) Benchmarking across many different applications with XGBoost on HUME vs XGBoost on ECFP+RDKit+Mordred vs XGBoost on ECFP vs best CLM vs best GNN
- D) plot of 10k, 100k, 1M, 1B, 10B SMILES to encode and show an extrapolated (measure a couple points) amount of CPU hours it would take to encode them: best CLM, best GNN, ECFP only, ECFP+RDKit+Mordred, HUME. Show same in mirror plot but in GPU hours (only CLM and GNN will change)

Then in the end we need to package everything up, have it be very user friendly with python API etc, set up pip install, set up documentation, write extensive methodology. 
Show in the SI that the descriptors we compute are bit-identical with RDKit or Mordred implementation on 1M (100M?) SMILES 


. Novelty — honest read

Split the claim into three parts. They have very different novelty status.

Claim	Novel?	Closest prior art
"XGBoost on ECFP+descriptors is a near-unbeatable baseline; pretrained encoders don't beat it"	No. Well-trodden.	Jiang 2021, Deng 2023, van Tilborg 2022 (MoleculeACE), Praski 2025 — all already in your notes. Also Sun et al. NeurIPS 2022, "Does GNN pretraining help molecular representation?", which found pretraining gains vanish against properly tuned baselines.
"Descriptors are fast if implemented well in C++"	Weakly. Asserted before, rarely quantified.	alvaDesc (commercial C++, Dragon lineage) is the incumbent speed claim. Dalke's chemfp is the canonical "cheminformatics done properly in C". Descriptastorus (Kelley — who also wrote BCUT.cpp). Mordred's own paper benchmarks against PaDEL. scikit-fingerprints (2024/25) benchmarks parallel FP generation.
"Co-generate ECFP and descriptors from one traversal, sharing intermediates"	Yes, as far as I know.	I'm not aware of a published dependency-DAG-over-shared-intermediates design. This is the actual contribution.
"DL descriptor prediction is slower than exact computation at matched accuracy"	Yes, as a quantified negative result.	The genre exists; this specific measurement doesn't, to my knowledge.