# Algorithms

This document describes the core algorithmic components of the MSA tool.

---

## 1. Pairwise Global Alignment: Needleman-Wunsch with Affine Gap Penalties

The foundational pairwise alignment engine uses the **Needleman-Wunsch (NW)** dynamic programming algorithm [1],
extended with **affine gap penalties** [2].

### Recurrence

The alignment DP uses three matrices — **S** (best score), **E** (gap in X), **F** (gap in Y):

```
F[i][j] = max(S[i-1][j] + GAP_OPEN, F[i-1][j] + GAP_EXTEND)
E[i][j] = max(S[i][j-1] + GAP_OPEN, E[i][j-1] + GAP_EXTEND)
S[i][j] = max(S[i-1][j-1] + score(x[i], y[j]), E[i][j], F[i][j])
```

Affine gap cost for a gap of length `k` = `GAP_OPEN + k * GAP_EXTEND`.

### AVX2 Vectorization

The inner loop over columns is vectorized with **Intel AVX2 SIMD intrinsics**, processing 8 integers simultaneously
using `__m256i` vectors. This yields a practical ~8x throughput improvement on the DP fill step.

### Traceback

A traceback buffer stores pointer characters (`M`, `E`, `F`) for path reconstruction. The aligned sequences are
recovered by reverse-tracing from cell `(m, n)` to `(0, 0)`.

---

## 2. Scoring Matrices

### DNA — EDNAFULL

For DNA/RNA sequences, the **EDNAFULL** matrix handles all IUPAC ambiguity codes (`R`, `Y`, `S`, `W`, `K`, `M`, `B`,
`D`, `H`, `V`, `N`) in addition to standard bases `A`, `C`, `G`, `T/U` [3].

### Protein — BLOSUM62

For protein sequences, the **BLOSUM62** substitution matrix is used [4]. It was derived from the BLOCKS database of
conserved protein regions by counting substitutions in sequence pairs sharing ≥62% identity, making it well-suited for
moderately divergent homologs.

Unknown residues are mapped to `X` (BLOSUM62) or `N` (EDNAFULL).

---

## 3. Distance Matrix and UPGMA Guide Tree

### Pairwise Distance

For each sequence pair `(i, j)`, pairwise global alignment is computed and sequence **identity** is calculated as:

```
identity = matches / alignment_length
distance = 1.0 - identity
```

The distance matrix computation is **parallelized with OpenMP** using a `schedule(dynamic)` clause for load balancing.

### UPGMA

The **Unweighted Pair Group Method with Arithmetic Mean (UPGMA)** [5] builds a rooted ultrametric tree from the distance
matrix. The implementation uses a **min-heap priority queue** for O(n² log n) complexity.

At each step, the two closest clusters `a` and `b` are merged. New distances are updated as:

```
D(merged, k) = (D(a,k)*|a| + D(b,k)*|b|) / (|a| + |b|)
```

The resulting tree is serialized to **Newick format** and written to `guide_tree.nwk`.

---

## 4. Progressive MSA via Guide Tree

After parsing the Newick guide tree into a binary tree structure, **progressive alignment** is performed bottom-up [6]:

1. **Leaf nodes** receive their raw sequences as profiles.
2. **Internal nodes** align two child profiles using their **consensus sequences** as representatives.
3. **Gap projection** (`projectGaps`) propagates newly introduced gaps back into all sequences of a profile.
4. The two aligned profiles are **merged** to form the parent profile.

The consensus-based profile alignment avoids full profile DP while still propagating structural information across the
alignment.

---

## 5. Star Alignment (Fallback Method)

A simpler `msa_star` function implements **center-star alignment** [7]:

1. The first sequence is chosen as the center.
2. Each subsequent sequence is pairwise-aligned to the (growing) center.
3. Gaps introduced in the center are projected into all previously aligned sequences.

This runs in O(n × L²) and serves as a simpler alternative to guide-tree progressive alignment.

---

## 6. Iterative Refinement

After the initial progressive MSA, **iterative refinement** improves the alignment using random profile splitting [8]:

1. Sequences are **randomly partitioned** into two groups A and B.
2. Consensus sequences for A and B are computed.
3. A and B are **re-aligned** via pairwise NW on their consensus sequences.
4. Gaps are projected back into full profiles.
5. If the new **SP score** is better, it replaces the current best.

Multiple refinement **workers run in parallel** (OpenMP), each seeded with an independent Mersenne Twister RNG. The best
result across all threads is adopted after each round.

---

## 7. Sum-of-Pairs (SP) Score

Alignment quality is evaluated using the **Sum-of-Pairs (SP) score** [9]:

```
SP = Σ_{i<k} score(seq_i, seq_k, gap_penalties)
```

For each pair of sequences, the affine gap penalty model is applied correctly per pair (gap state tracked independently
per pair across columns). This is the standard objective function for MSA quality evaluation.

---

## 8. Optimal Gap Penalty Search

A **grid search** over gap penalty parameters is performed before the main alignment:

- `GAP_OPEN` ∈ {-25, -22.5, ..., -10} (step 2.5)
- `GAP_EXTEND` ∈ {-5, -4, ..., -1} (step 1.0)

For each combination, a full MSA is built and the SP score is computed. The combination with the best SP score is
selected. The grid search is **parallelized with OpenMP collapse(2)**.

---

## References

See [REFERENCES.md](REFERENCES.md).
