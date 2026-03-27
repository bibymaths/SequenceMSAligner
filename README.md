# MSA Tool — Multiple Sequence Alignment using AVX2 + affine gap combination

**Author:** Abhinav Mishra  
**Language:** C++17  
**Parallelism:** OpenMP + AVX2 SIMD

## Overview

This tool performs **Multiple Sequence Alignment (MSA)** for both DNA and protein sequences. It implements a full
pipeline:

1. FASTA input parsing
2. Pairwise distance matrix computation
3. Guide tree construction (UPGMA)
4. Progressive MSA via guide-tree traversal
5. Iterative refinement using random profile splitting
6. Consensus generation and output

## Features

- Supports **DNA** (EDNAFULL matrix) and **Protein** (BLOSUM62 matrix) modes
- AVX2-vectorized Needleman-Wunsch global alignment (affine gap penalties)
- UPGMA guide tree with Newick output
- Parallel iterative refinement (OpenMP multi-threaded)
- Automatic grid search for optimal gap penalties
- HTML visualization output with color-coded conservation
- Sum-of-Pairs (SP) scoring for alignment quality

## Output Files

| File                     | Description                       |
|--------------------------|-----------------------------------|
| `msa.fasta`              | Final MSA in FASTA format         |
| `msa_visualization.html` | Color-coded HTML alignment viewer |
| `guide_tree.nwk`         | UPGMA guide tree in Newick format |
| `consensus.fasta`        | Consensus sequence                |
| `consensus_details.txt`  | Per-column consensus match report |
| `identity_matrix.txt`    | Pairwise sequence identity matrix |

## Quick Start

```bash
# Compile
g++ -O3 -mavx2 -fopenmp -std=c++17 -o msa main.cpp

# DNA alignment
./msa --mode dna out_dir seq1.fasta seq2.fasta seq3.fasta

# Protein alignment with custom gap penalties
./msa --mode protein --gap_open -12 --gap_extend -2 out_dir p1.fasta p2.fasta
```

## Dependencies

- `EDNAFULL.h` — DNA substitution matrix (IUPAC)
- `EBLOSUM62.h` — Protein BLOSUM62 substitution matrix
- C++17 standard library
- Intel AVX2 intrinsics (`immintrin.h`)
- OpenMP (`omp.h`)

## References

See [REFERENCES.md](docs/REFERENCES.md) for full citations.
