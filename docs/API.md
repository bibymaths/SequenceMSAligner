# API Reference

This document describes all major functions in `main.cpp`.

---

## Scoring Functions

### `int edna_score(char x, char y)`

Returns the EDNAFULL score for a pair of DNA/RNA characters (IUPAC-aware).  
Unknown characters map to `N`.

### `int blosum62_score(char x, char y)`

Returns the BLOSUM62 score for a pair of amino acid characters.  
Unknown characters map to `X`.

### `int score(char x, char y, ScoreMode mode)`

Dispatcher: calls `edna_score` for `MODE_DNA`, `blosum62_score` for `MODE_PROTEIN`.

---

## Pairwise Alignment

### `int computeGlobalAlignment(x, y, mode, score_fn, aligned_x, aligned_y)`

Computes global alignment of strings `x` and `y` using Needleman-Wunsch with affine gap penalties and AVX2 SIMD
vectorization.

| Parameter   | Type            | Description                          |
|-------------|-----------------|--------------------------------------|
| `x`         | `const string&` | First sequence                       |
| `y`         | `const string&` | Second sequence                      |
| `mode`      | `ScoreMode`     | `MODE_DNA` or `MODE_PROTEIN`         |
| `score_fn`  | `ScoreFn`       | Function pointer to scoring function |
| `aligned_x` | `string&`       | Output: aligned first sequence       |
| `aligned_y` | `string&`       | Output: aligned second sequence      |
| **returns** | `int`           | Alignment score                      |

---

## Distance and Tree

### `vector<vector<double>> computeDistanceMatrix(seqs, mode, fn)`

Computes a symmetric pairwise distance matrix from identity scores.  
Parallelized with `#pragma omp parallel for schedule(dynamic)`.

### `string buildUPGMATree(D, names)`

Constructs a UPGMA phylogenetic tree from distance matrix `D`.  
Returns a **Newick-format** string.

### `Node* parseNewick(nwk, names)`

Parses a Newick string into a binary tree of `Node` structs.  
Leaf nodes carry `seq_index` pointing into the original sequence array.

### `string formatNewickString(nwk)`

Formats a compact Newick string into a human-readable, indented multi-line form.

---

## MSA Construction

### `vector<string> msa_star(hdrs, seqs, mode, fn)`

Builds an MSA using the center-star method. Simple O(n × L²) approach.

### `vector<string> build_profile(Node* n, mode, fn)`

Recursively builds an MSA profile for a guide-tree node.  
Leaf nodes return single-sequence profiles.  
Internal nodes align child consensus sequences and project gaps.

### `void projectGaps(oldc, newc, seqs)`

Projects newly introduced gaps from `newc` (aligned representative) back into all sequences in `seqs`. Compares old and
new representative character-by-character to detect inserted gap columns.

---

## Refinement

### `vector<string> refine_msa(initial_msa, rounds, iterations_per_round, mode, fn)`

Orchestrates multi-round parallel iterative refinement.  
Spawns OpenMP threads, each running `refine_msa_worker`.  
After each round, adopts the best result across all threads.

### `vector<string> refine_msa_worker(initial_msa, iterations, mode, fn, g)`

Single-threaded refinement worker. Performs random profile splitting and re-alignment for `iterations` steps. Returns
the best MSA found.

---

## Scoring and Analysis

### `long long calculate_sp_score(msa, mode, fn)`

Computes the Sum-of-Pairs (SP) score of an MSA with true affine gap penalties.  
Time complexity: O(N² × L) where N = number of sequences, L = alignment length.

### `string generate_consensus(profile)`

Generates a consensus sequence by taking the most frequent non-gap character at each alignment column.

### `void analyze_and_save_consensus(msa, hdrs, consensus_seq, outdir)`

Writes a detailed report (`consensus_details.txt`) mapping each consensus column to matching positions in individual
sequences.

---

## I/O Functions

### `void processFasta(fn, hdr, seq)`

Reads a FASTA file. Extracts the first header line and concatenates all sequence lines into a single uppercase string.
Strips non-alphabetic characters.

### `void sanitize_header(header)`

Replaces whitespace characters with underscores in a FASTA header string.

### `void saveIdentityMatrix(D, hdrs, outdir)`

Saves a formatted pairwise identity matrix to `identity_matrix.txt`.

### `void saveMSA_to_HTML(aln, hdrs, outdir)`

Writes a color-coded HTML MSA visualization to `msa_visualization.html`.  
Color scheme:

- 🟢 **Green** — fully conserved columns
- ⬜ **Gray** — variable residues
- 🔴 **Red** — gaps

### `GapSearchResult find_optimal_gap_penalties(seqs, hdrs, mode, fn)`

Grid-searches gap penalty combinations in parallel using OpenMP `collapse(2)`.  
Returns a `GapSearchResult` with `score`, `gap_open`, `gap_extend`.

---

## Data Structures

### `enum ScoreMode`

```cpp
enum ScoreMode { MODE_DNA, MODE_PROTEIN };
```

### `struct Node`

```cpp
struct Node {
    bool leaf;
    int seq_index;
    vector<string> profile;
    Node *left, *right;
};
```

### `struct GapSearchResult`

```cpp
struct GapSearchResult {
    long long score;
    double gap_open;
    double gap_extend;
};
```

---

## Global Parameters

| Variable     | Type     | Default | Description                |
|--------------|----------|---------|----------------------------|
| `GAP_OPEN`   | `double` | 0       | Gap opening penalty        |
| `GAP_EXTEND` | `double` | 0       | Gap extension penalty      |
| `LINE_WIDTH` | `int`    | 80      | Columns per line in output |
