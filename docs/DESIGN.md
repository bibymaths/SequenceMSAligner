# Design and Architecture

## Pipeline Overview

```
Input FASTA files
       │
       ▼
┌─────────────────────┐
│  processFasta()     │  Parse headers + sequences
└─────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  find_optimal_gap_penalties()   │  Grid search (OpenMP parallel)
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────┐
│ computeDistanceMatrix│  Pairwise NW + identity (OpenMP)
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  buildUPGMATree()   │  UPGMA clustering → Newick
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  parseNewick()      │  Build binary tree of Node structs
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  build_profile()    │  Progressive alignment bottom-up
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  refine_msa()       │  Iterative refinement (OpenMP workers)
└─────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Output: msa.fasta, .html, .nwk, consensus, ...  │
└─────────────────────────────────────────────────┘
```

---

## Parallelism Strategy

The tool employs **two levels of parallelism**:

### 1. Data Parallelism — Distance Matrix

```cpp
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < n; ++i)
    for (int j = i+1; j < n; ++j)
        // pairwise NW
```

Each `(i, j)` alignment is independent. Dynamic scheduling balances variable-length sequence pairs.

### 2. Task Parallelism — Iterative Refinement

```cpp
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    std::mt19937 g(std::random_device{}() + tid);
    thread_results[tid] = refine_msa_worker(global_best_msa, iterations, mode, fn, g);
}
```

Each thread independently searches for improvements using its own RNG seed, starting from the current global best.
Results are merged serially after each round.

### 3. Instruction-Level Parallelism — AVX2

The DP inner loop in `computeGlobalAlignment` processes 8 cells simultaneously using `__m256i` SIMD vectors, reducing
the effective loop count by 8×.

---

## Memory Layout

The DP uses two rolling rows (`S_prev`, `S_cur`) of width `n8+1` (n rounded up to multiple of 8 for AVX2 alignment). A
flat `trace_buf` of size `(m+1)×(n+1)` stores traceback characters.

```
S_prev[n8+1]   ← previous row
S_cur [n8+1]   ← current row
E_cur [n8+1]   ← E matrix (gap in x)
F_prev[n8+1]   ← F matrix (gap in y)
trace_buf[(m+1)×(n+1)]  ← path reconstruction
```

Rows are swapped via pointer swap (`std::swap(Sp, Sc)`) after each row fill — zero copy.

---

## Profile Representation

Profiles are represented as `std::vector<std::string>` where all strings share the same length (the alignment length).
Gaps are represented as the `-` character.

Gap projection (`projectGaps`) inserts gap columns by comparing the old and new consensus sequences and inserting `-` at
positions where the new sequence has an inserted gap relative to the old one.

---

## Scoring Mode Dispatch

Scoring is dispatched via a **function pointer** (`ScoreFn`):

```cpp
using ScoreFn = int (*)(char, char);
ScoreFn fn = (mode == MODE_DNA) ? edna_score : blosum62_score;
```

This allows all downstream functions to call `fn(a, b)` without branching on mode at every call site.

---

## Header Normalization

FASTA headers are normalized differently per mode:

- **DNA mode**: header truncated at the first whitespace/underscore → short locus ID
- **Protein mode**: UniProt pipe-delimited format `sp|ID|NAME` → extracts the accession between first and second `|`

All whitespace in headers is replaced with `_` to prevent tokenization issues in Newick strings and output files.

---

## Guide Tree Parsing

The Newick parser uses a **character-by-character state machine** with an explicit `std::stack<Node*>`:

- `(` → push new internal node
- `)` → pop node from stack
- `,` → flush current text as leaf, stay at same level
- `:` → skip branch length digits
- Other chars → accumulate into current text buffer

Leaf node names are looked up in the `names` vector to retrieve their `seq_index`.

---

## Error Handling

| Location          | Failure Mode                   | Response                               |
|-------------------|--------------------------------|----------------------------------------|
| `processFasta`    | File not found                 | `throw std::runtime_error`             |
| `parseNewick`     | Leaf name not in input headers | Print fatal error to stderr, return -1 |
| `build_profile`   | Invalid seq_index on leaf      | Print fatal error, `exit(1)`           |
| `saveMSA_to_HTML` | Cannot open output file        | Print warning to stderr, continue      |
| `main()`          | Insufficient arguments         | Print usage string, `return 1`         |
