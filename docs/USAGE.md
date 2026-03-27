# Usage Guide

## Installation and Compilation

### Prerequisites

- GCC ≥ 9.0 or Clang ≥ 10.0 (C++17 support)
- Intel CPU with AVX2 support (Haswell, 2013+)
- OpenMP (typically bundled with GCC)

### Compile

```bash
g++ -O3 -mavx2 -fopenmp -std=c++17 -o msa main.cpp
```

For debugging (disable optimizations, enable assertions):

```bash
g++ -O0 -g -mavx2 -fopenmp -std=c++17 -o msa_debug main.cpp
```

---

## Command-Line Interface

```
./msa [OPTIONS] <outdir> <file1.fasta> <file2.fasta> [...]
```

### Options

| Flag           | Values             | Default | Description                      |
|----------------|--------------------|---------|----------------------------------|
| `--mode`       | `dna` or `protein` | `dna`   | Sequence type and scoring matrix |
| `--gap_open`   | float (negative)   | auto    | Gap opening penalty              |
| `--gap_extend` | float (negative)   | auto    | Gap extension penalty            |

> **Note:** If `--gap_open` and `--gap_extend` are not supplied, the tool automatically performs a grid search to find
> the optimal gap penalties for the input dataset.

---

## Input Format

Each input sequence must be in standard **FASTA format** in its own file:

```
>Sequence_Name Optional description here
ATCGATCGATCGATCG
ATCGATCG
```

- Only the **first header** per file is used.
- All sequence characters are converted to **uppercase**.
- Non-alphabetic characters are ignored.
- For DNA: standard bases `A`, `C`, `G`, `T` and IUPAC ambiguity codes are accepted.
- For Protein: standard 20 amino acids plus `B`, `Z`, `X`, `*`.

---

## Example Workflows

### DNA Alignment (automatic gap penalties)

```bash
./msa --mode dna results/ human.fasta chimp.fasta gorilla.fasta orangutan.fasta
```

### Protein Alignment with manual gap penalties

```bash
./msa --mode protein --gap_open -12 --gap_extend -2 output/ p1.fasta p2.fasta p3.fasta
```

### View output

```bash
# Check the guide tree
cat results/guide_tree.nwk

# View identity matrix
cat results/identity_matrix.txt

# Open HTML visualization in browser
firefox results/msa_visualization.html
```

---

## Output Files Explained

### `msa.fasta`

Final aligned sequences in FASTA format. All sequences have the same length (padded with `-`).

```
>SEQ1
ATCG--ATCG
>SEQ2
ATCGATCG--
```

### `msa_visualization.html`

Interactive HTML file showing the alignment in colored blocks of 80 columns.

- **Green background**: fully conserved column
- **Light gray**: variable residue
- **Red**: gap character (`-`)

Each line shows `[Header] [start position] [sequence block] [end position]`.

### `guide_tree.nwk`

Newick-format UPGMA tree used for progressive alignment ordering:

```
(
    (
        SEQ1:0.05,
        SEQ2:0.05
    ):0.12,
    SEQ3:0.17
);
```

### `identity_matrix.txt`

Tab-aligned pairwise identity percentages:

```
    1: SEQ1      100.00   87.30   72.15
    2: SEQ2       87.30  100.00   69.40
    3: SEQ3       72.15   69.40  100.00
```

### `consensus.fasta`

The per-column majority-vote consensus sequence.

### `consensus_details.txt`

Detailed report of which sequence residues match the consensus at each alignment column, with original (ungapped)
position numbers.

---

## Performance Tips

- **Thread count**: controlled by `OMP_NUM_THREADS` environment variable.
  ```bash
  OMP_NUM_THREADS=8 ./msa --mode dna out/ *.fasta
  ```
- AVX2 processes 8 integers per cycle; ensure the compiler flag `-mavx2` is set.
- For large datasets (>50 sequences), automatic gap penalty search can be slow — provide manual `--gap_open` and
  `--gap_extend` values.
- The refinement phase runs `3 rounds × 10 iterations × num_threads` alignments. Reduce hardcoded values in `main()` if
  needed for large N.

---

## Troubleshooting

| Error                                       | Likely Cause                                 | Fix                                                               |
|---------------------------------------------|----------------------------------------------|-------------------------------------------------------------------|
| `Cannot open <file>`                        | FASTA file path incorrect                    | Check path and permissions                                        |
| `FATAL PARSING ERROR: name not found`       | Header mismatch between FASTA and guide tree | Ensure FASTA headers are unique and contain no special characters |
| `FATAL LOGIC ERROR: Invalid sequence index` | Newick parse failure                         | Check that all sequences have unique, simple headers              |
| Segfault / crash                            | Non-AVX2 CPU                                 | Remove `-mavx2`, recompile with scalar fallback                   |
