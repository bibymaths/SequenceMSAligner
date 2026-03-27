# References

The following research papers form the theoretical foundation of this MSA tool.

---

## Core Algorithms

**[1] Needleman, S. B. & Wunsch, C. D. (1970).**  
*A general method applicable to the search for similarities in the amino acid sequence of two proteins.*  
Journal of Molecular Biology, 48(3), 443–453.  
https://doi.org/10.1016/0022-2836(70)90057-4
> The original dynamic programming algorithm for global pairwise sequence alignment, foundational to this tool's
`computeGlobalAlignment` function.

---

**[2] Gotoh, O. (1982).**  
*An improved algorithm for matching biological sequences.*  
Journal of Molecular Biology, 162(3), 705–708.  
https://doi.org/10.1016/0022-2836(82)90398-9
> Introduced the affine gap penalty model (gap open + gap extend) that reduces the O(m²n) complexity of gap scoring to
> O(mn). This tool's DP recurrence directly implements Gotoh's three-matrix formulation (S, E, F).

---

**[3] Henikoff, S. & Henikoff, J. G. (1992).**  
*Amino acid substitution matrices from protein blocks.*  
Proceedings of the National Academy of Sciences, 89(22), 10915–10919.  
https://doi.org/10.1073/pnas.89.22.10915
> Describes the BLOSUM family of substitution matrices, including BLOSUM62 used by this tool for protein alignment
> scoring.

---

**[4] Rice, P., Longden, I., & Bleasby, A. (2000).**  
*EMBOSS: The European Molecular Biology Open Software Suite.*  
Trends in Genetics, 16(6), 276–277.  
https://doi.org/10.1016/S0168-9525(00)02024-2
> Source of the EDNAFULL (also called NUC4.4) matrix used for DNA/RNA alignment with IUPAC ambiguity code support, as
> distributed in the EMBOSS package.

---

## Phylogenetics and Guide Trees

**[5] Sokal, R. R. & Michener, C. D. (1958).**  
*A statistical method for evaluating systematic relationships.*  
University of Kansas Science Bulletin, 38, 1409–1438.
> Introduced the UPGMA (Unweighted Pair Group Method with Arithmetic Mean) algorithm for hierarchical clustering, used
> here to construct the guide tree that orders progressive alignment.

---

## Progressive Multiple Sequence Alignment

**[6] Feng, D. F. & Doolittle, R. F. (1987).**  
*Progressive sequence alignment as a prerequisite to correct phylogenetic trees.*  
Journal of Molecular Evolution, 25(4), 351–360.  
https://doi.org/10.1007/BF02603120
> Established the progressive alignment paradigm: build a guide tree, then align sequences from the leaves up. The
> alignment order follows evolutionary relatedness. This paper is the conceptual basis for the guide-tree-driven
`build_profile` function.

---

**[7] Gusfield, D. (1993).**  
*Efficient methods for multiple sequence alignment with guaranteed error bounds.*  
Bulletin of Mathematical Biology, 55(1), 141–154.  
https://doi.org/10.1007/BF02460299
> Proved that center-star alignment achieves a 2(1-1/k) approximation to the optimal SP score, where k is the number of
> sequences. This forms the theoretical basis for the `msa_star` function.

---

## Iterative Refinement

**[8] Berger, M. P. & Munson, P. J. (1991).**  
*A novel randomized iterative strategy for aligning multiple protein sequences.*  
Computer Applications in the Biosciences, 7(4), 479–484.  
https://doi.org/10.1093/bioinformatics/7.4.479
> Proposed the iterative refinement strategy of splitting an MSA into two groups, realigning them, and accepting the new
> alignment if it improves the objective score. This directly corresponds to the `refine_msa` and `refine_msa_worker`
> functions.

---

## Scoring Functions for MSA

**[9] Carrillo, H. & Lipman, D. (1988).**  
*The multiple sequence alignment problem in biology.*  
SIAM Journal on Applied Mathematics, 48(5), 1073–1082.  
https://doi.org/10.1137/0148063
> Formally defined the Sum-of-Pairs (SP) objective function for multiple sequence alignment, which is implemented in
`calculate_sp_score` as the quality metric guiding both gap penalty search and iterative refinement.

---

## SIMD Vectorization

**[10] Rognes, T. (2011).**  
*Faster Smith-Waterman database searches with inter-sequence SIMD parallelisation.*  
BMC Bioinformatics, 12, 221.  
https://doi.org/10.1186/1471-2105-12-221
> Demonstrated the use of SIMD (SSE2/AVX) intrinsics to accelerate sequence alignment DP, providing the methodological
> blueprint for AVX2 vectorization of the inner DP loop in `computeGlobalAlignment`.

---

## Tools and Software Comparisons

**[11] Thompson, J. D., Higgins, D. G., & Gibson, T. J. (1994).**  
*CLUSTAL W: Improving the sensitivity of progressive multiple sequence alignment through sequence weighting,
position-specific gap penalties and weight matrix choice.*  
Nucleic Acids Research, 22(22), 4673–4680.  
https://doi.org/10.1093/nar/22.22.4673
> The landmark MSA tool ClustalW that popularized the UPGMA/progressive-alignment pipeline. This tool follows a similar
> architectural approach: distance matrix → UPGMA tree → progressive alignment → iterative refinement.

---

**[12] Edgar, R. C. (2004).**  
*MUSCLE: Multiple sequence alignment with high accuracy and high throughput.*  
Nucleic Acids Research, 32(5), 1792–1797.  
https://doi.org/10.1093/nar/gkh340
> MUSCLE introduced the profile-profile progressive alignment and iterative refinement paradigm that achieves high
> accuracy. This tool's refinement module is conceptually similar to MUSCLE's refinement stage.
