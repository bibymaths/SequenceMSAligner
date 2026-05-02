<img src="docs/assets/logo.png" alt="SequenceMSAligner" width="300">

**Multiple Sequence Alignment using AVX2 + affine gap combination**

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg) ![CMake](https://img.shields.io/badge/CMake-≥3.10-blue.svg)
[![Doxygen](https://img.shields.io/badge/docs-Doxygen-blue)](https://bibymaths.github.io/SequenceMSAligner/api/index.html)

--- 

## [Documentation](https://bibymaths.github.io/SequenceMSAligner/)

--- 

## Features

- Supports **DNA** (EDNAFULL matrix) and **Protein** (BLOSUM62 matrix) modes
- AVX2-vectorized Needleman-Wunsch global alignment (affine gap penalties)
- UPGMA guide tree with Newick output
- Parallel iterative refinement (OpenMP multi-threaded)
- Automatic grid search for optimal gap penalties
- HTML visualization output with color-coded conservation
- Sum-of-Pairs (SP) scoring for alignment quality

---

## Output Files

| File                     | Description                       |
|--------------------------|-----------------------------------|
| `msa.fasta`              | Final MSA in FASTA format         |
| `msa_visualization.html` | Color-coded HTML alignment viewer |
| `guide_tree.nwk`         | UPGMA guide tree in Newick format |
| `consensus.fasta`        | Consensus sequence                |
| `consensus_details.txt`  | Per-column consensus match report |
| `identity_matrix.txt`    | Pairwise sequence identity matrix |

--- 

## Quick Start

```bash
# Compile
g++ -O3 -mavx2 -fopenmp -std=c++17 -o msa main.cpp

# DNA alignment
./msa --mode dna out_dir seq1.fasta seq2.fasta seq3.fasta

# Protein alignment with custom gap penalties
./msa --mode protein --gap_open -12 --gap_extend -2 out_dir p1.fasta p2.fasta
```

---

## References

See [REFERENCES](https://bibymaths.github.io/SequenceMSAligner/REFERENCES.html) for full citations.

---