<img src="docs/assets/logo.png" alt="SequenceMSAligner" width="300">

**Multiple Sequence Alignment using AVX2 + affine gap combination**

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg) ![CMake](https://img.shields.io/badge/CMake-≥3.10-blue.svg)
[![Doxygen](https://img.shields.io/badge/docs-Doxygen-blue)](https://bibymaths.github.io/SequenceMSAligner/api/index.html)

--- 

## [Documentation](https://bibymaths.github.io/SequenceMSAligner/)

---

## Quick Start

Download the prebuilt Linux x86_64 binary:

```bash
wget https://github.com/bibymaths/SequenceMSAligner/releases/download/v0.9.0/msalign-linux-x86_64
chmod +x msalign-linux-x86_64
````

Run DNA alignment:

```bash
./msalign-linux-x86_64 \
  --mode=dna \
  --gap-open=-10 \
  --gap-extend=-0.5 \
  results \
  seq1.fasta seq2.fasta seq3.fasta
```

Run protein alignment:

```bash
./msalign-linux-x86_64 \
  --mode=protein \
  --gap-open=-12 \
  --gap-extend=-2 \
  results \
  p1.fasta p2.fasta p3.fasta
```

Build from source with CMake:

```bash
git clone https://github.com/bibymaths/SequenceMSAligner.git
cd SequenceMSAligner
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/msalign --mode=dna --gap-open=-10 --gap-extend=-0.5 results seq1.fasta seq2.fasta seq3.fasta
```

--- 

## References

See [REFERENCES](https://bibymaths.github.io/SequenceMSAligner/REFERENCES.html) for full citations.

---