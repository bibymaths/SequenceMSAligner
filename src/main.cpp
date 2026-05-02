/**
 * @file main.cpp
 * @brief Multiple sequence alignment (MSA) tool for DNA and protein sequences.
 *
 * Implements a full MSA pipeline including:
 *  - Pairwise global alignment via affine-gap Needleman-Wunsch with AVX2
 *    vectorization.
 *  - UPGMA guide-tree construction from a pairwise identity distance matrix.
 *  - Three independent progressive alignment strategies run in parallel:
 *    (1) Standard progressive NW, (2) FFT-seeded banded alignment, and
 *    (3) COFFEE Log-Expectation (LE) guided alignment.
 *  - Iterative MSA refinement using random profile splitting and
 *    Sum-of-Pairs (SP) score optimization across OpenMP threads.
 *  - HTML and FASTA output with color-coded conservation visualization.
 *
 * @author Abhinav Mishra <mishraabhinav36@gmail.com>
 * @copyright Copyright (C) 2025-2026 Abhinav Mishra
 * @license BSD 3-Clause
 */

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <climits>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <valarray>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <chrono>
#include <random>
#include <ctime>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include <omp.h>
#include <cstdlib>

#include "EBLOSUM62.h"
#include "EDNAFULL.h"

// ANSI color codes
#define RESET "\033[0m"
#define GREEN "\033[32m"
#define RED "\033[31m"
#define CYAN "\033[36m"

/**
 * @brief Holds affine gap penalty parameters for sequence alignment.
 *
 * Gap cost for a run of k gaps is computed as:
 *   cost = gap_open + k * gap_extend
 *
 * Both values should be negative. @c gap_open penalizes the initiation
 * of a new gap and @c gap_extend penalizes each additional position within
 * an existing gap.
 */
struct AlignParams {
  double gap_open = -10.0;  ///< Penalty for opening a new gap.
  double gap_extend =
      -0.5;  ///< Penalty for extending an existing gap by one position.
};

/**
 * @brief A node in the UPGMA guide tree used for progressive alignment.
 *
 * Leaf nodes correspond to individual input sequences. Internal nodes
 * represent merged alignment profiles. The @c profile field is populated
 * bottom-up during the progressive alignment traversal.
 */
struct Node {
  bool leaf;       ///< True if this is a leaf (single sequence) node.
  int  seq_index;  ///< Index into the original sequence array; valid only if @c
                   ///< leaf is true.
  std::vector<std::string> profile;  ///< Aligned sequences accumulated at this
                                     ///< node during tree traversal.
  Node* left  = nullptr;  ///< Left child node; nullptr for leaf nodes.
  Node* right = nullptr;  ///< Right child node; nullptr for leaf nodes.
};

/**
 * @brief Recursively deallocates all nodes of a guide tree.
 *
 * Performs a post-order traversal, freeing children before the parent,
 * to prevent use-after-free. Safe to call with a null pointer.
 *
 * @param n Pointer to the root node of the tree to free. May be @c nullptr.
 */
void free_tree(Node* n) {
  if (!n) return;
  free_tree(n->left);
  free_tree(n->right);
  delete n;
}

static const int LINE_WIDTH = 80;

inline int checked_size_to_int(std::size_t value, const char* name) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string(name) + " exceeds int range");
  }
  return static_cast<int>(value);
}

// Scoring modes
enum ScoreMode { MODE_DNA, MODE_PROTEIN };
using ScoreFn = int (*)(char, char);

using CArray = std::valarray<std::complex<double>>;

// DNA / EDNAFULL lookup
static const std::array<uint8_t, 256> char2idx = []() {
  std::array<uint8_t, 256> m{};
  m.fill(255);
  m['A'] = 0;
  m['C'] = 1;
  m['G'] = 2;
  m['T'] = 3;
  m['U'] = 3;
  m['R'] = 4;
  m['Y'] = 5;
  m['S'] = 6;
  m['W'] = 7;
  m['K'] = 8;
  m['M'] = 9;
  m['B'] = 10;
  m['D'] = 11;
  m['H'] = 12;
  m['V'] = 13;
  m['N'] = 14;
  m['X'] = 14;
  return m;
}();

// Protein / BLOSUM62 lookup
static const std::array<uint8_t, 256> prot_idx = []() {
  std::array<uint8_t, 256> m{};
  m.fill(255);
  const char* AA = "ARNDCQEGHILKMFPSTWYVBZX*";
  for (int i = 0; AA[i]; ++i) m[(uint8_t)AA[i]] = i;
  return m;
}();

/**
 * @brief Returns the BLOSUM62 substitution score for a pair of amino acids.
 *
 * Characters are converted to uppercase before lookup. Unknown or
 * non-standard amino acids are mapped to @c 'X' before querying the matrix.
 *
 * @param x First amino acid character (case-insensitive).
 * @param y Second amino acid character (case-insensitive).
 * @return  Integer substitution score from the BLOSUM62 matrix.
 */
inline int blosum62_score(char x, char y) {
  // map to uppercase in case it slipped through
  x = static_cast<char>(std::toupper(static_cast<unsigned char>(x)));
  y = static_cast<char>(std::toupper(static_cast<unsigned char>(y)));

  uint8_t ix = prot_idx[static_cast<uint8_t>(x)];
  if (ix == 255) ix = prot_idx['X'];  // unknown ? X
  uint8_t iy = prot_idx[static_cast<uint8_t>(y)];
  if (iy == 255) iy = prot_idx['X'];

  return static_cast<int>(std::round(EBLOSUM62_matrix[ix][iy]));
}

/**
 * @brief Returns the EDNAFULL substitution score for a pair of nucleotides.
 *
 * Characters are converted to uppercase before lookup. Handles all IUPAC
 * ambiguity codes (R, Y, S, W, K, M, B, D, H, V, N). Unknown characters
 * are mapped to @c 'N' before querying the matrix.
 *
 * @param x First nucleotide character (case-insensitive, IUPAC).
 * @param y Second nucleotide character (case-insensitive, IUPAC).
 * @return  Integer substitution score from the EDNAFULL matrix.
 */
inline int edna_score(char x, char y) {
  x = static_cast<char>(std::toupper(static_cast<unsigned char>(x)));
  y = static_cast<char>(std::toupper(static_cast<unsigned char>(y)));

  uint8_t ix = char2idx[static_cast<uint8_t>(x)];
  if (ix == 255) ix = char2idx['N'];  // unknown ? N
  uint8_t iy = char2idx[static_cast<uint8_t>(y)];
  if (iy == 255) iy = char2idx['N'];

  return static_cast<int>(std::round(EDNAFULL_matrix[ix][iy]));
}

/**
 * @brief Dispatches to the correct substitution scoring function based on mode.
 *
 * @param x    First sequence character.
 * @param y    Second sequence character.
 * @param mode Scoring mode: @c MODE_DNA uses EDNAFULL, @c MODE_PROTEIN uses
 * BLOSUM62.
 * @return     Integer substitution score.
 */
inline int score(char x, char y, ScoreMode mode) {
  return mode == MODE_DNA ? edna_score(x, y) : blosum62_score(x, y);
}

/**
 * @brief Performs an in-place Cooley-Tukey radix-2 decimation-in-time FFT.
 *
 * Recursively splits the input into even- and odd-indexed elements,
 * transforms each half, then combines them using the butterfly operation.
 * The array length must be a power of 2; no length validation is performed.
 *
 * Time complexity: O(N log N).
 *
 * @param[in,out] x Complex-valued array of length 2^k to transform in place.
 *                  On return, contains the discrete Fourier transform of the
 * input.
 */
void fft(CArray& x) {
  const size_t N = x.size();
  if (N <= 1) return;

  CArray even = x[std::slice(0, N / 2, 2)];
  CArray odd  = x[std::slice(1, N / 2, 2)];

  fft(even);
  fft(odd);

  for (size_t k = 0; k < N / 2; ++k) {
    auto t       = std::polar(1.0, -2.0 * M_PI * k / N) * odd[k];
    x[k]         = even[k] + t;
    x[k + N / 2] = even[k] - t;
  }
}

/**
 * @brief Encodes a biological sequence as a real-valued numeric vector for FFT.
 *
 * Two encoding schemes are used depending on the scoring mode:
 *  - **Protein**: Kyte-Doolittle hydrophobicity scale. Hydrophobic residues
 *    receive positive values; hydrophilic residues receive negative values.
 *    Unrecognized amino acids are encoded as 0.0.
 *  - **DNA**: Binary purine/pyrimidine indicator. Purines (A, G) ? +1.0;
 *    pyrimidines (C, T, U) ? ?1.0. Ambiguous bases encode as 0.0.
 *
 * The resulting vector is suitable as input to @c fftCrossCorrelation to
 * estimate the relative offset between two sequences without full DP alignment.
 *
 * @param seq  Input sequence string (uppercase recommended).
 * @param mode Encoding mode: @c MODE_PROTEIN or @c MODE_DNA.
 * @return     Vector of doubles of length @c seq.size() containing the
 * encoding.
 */
std::vector<double> encodeSequence(const std::string& seq, ScoreMode mode) {
  // Hydrophobicity (Kyte-Doolittle) for protein
  static const std::array<double, 256> hydro = []() {
    std::array<double, 256> h{};
    h['A'] = 1.8;
    h['R'] = -4.5;
    h['N'] = -3.5;
    h['D'] = -3.5;
    h['C'] = 2.5;
    h['Q'] = -3.5;
    h['E'] = -3.5;
    h['G'] = -0.4;
    h['H'] = -3.2;
    h['I'] = 4.5;
    h['L'] = 3.8;
    h['K'] = -3.9;
    h['M'] = 1.9;
    h['F'] = 2.8;
    h['P'] = -1.6;
    h['S'] = -0.8;
    h['T'] = -0.7;
    h['W'] = -0.9;
    h['Y'] = -1.3;
    h['V'] = 4.2;
    return h;
  }();
  // Binary purine/pyrimidine for DNA
  static const std::array<double, 256> purine = []() {
    std::array<double, 256> p{};
    p['A'] = 1.0;
    p['G'] = 1.0;
    p['C'] = -1.0;
    p['T'] = -1.0;
    p['U'] = -1.0;
    return p;
  }();

  std::vector<double> enc(seq.size());
  for (size_t i = 0; i < seq.size(); ++i)
    enc[i] = (mode == MODE_PROTEIN)
                 ? hydro[static_cast<unsigned char>(seq[i])]
                 : purine[static_cast<unsigned char>(seq[i])];
  return enc;
}

/**
 * @brief Estimates the best alignment offset between two sequences via FFT
 * cross-correlation.
 *
 * Encodes both sequences as real-valued vectors (see @c encodeSequence),
 * zero-pads them to the next power-of-2 length, computes their
 * cross-correlation in the frequency domain as @f$ F_a \cdot \overline{F_b}
 * @f$, and locates the peak of the inverse transform. The peak position is the
 * shift of @p b relative to @p a that maximises their physicochemical
 * similarity.
 *
 * Negative offsets (b shifted left of a) are recovered by wrapping indices in
 * @f$ [N/2, N) @f$ back to @f$ [-(N/2), 0) @f$.
 *
 * Time complexity: O((|a| + |b|) log(|a| + |b|)).
 *
 * @param a    First sequence.
 * @param b    Second sequence.
 * @param mode Encoding mode passed to @c encodeSequence.
 * @return     A pair @c {best_offset, best_score} where @c best_offset is the
 *             integer shift of @p b relative to @p a with the highest
 * correlation and @c best_score is the corresponding correlation value.
 */
std::pair<int, double> fftCrossCorrelation(const std::string& a,
                                           const std::string& b,
                                           ScoreMode          mode) {
  size_t N   = 1;
  size_t len = a.size() + b.size();
  while (N < len) N <<= 1;

  auto ea = encodeSequence(a, mode);
  auto eb = encodeSequence(b, mode);

  CArray fa(N), fb(N);
  for (size_t i = 0; i < ea.size(); ++i) fa[i] = ea[i];
  for (size_t i = 0; i < eb.size(); ++i) fb[i] = eb[i];

  fft(fa);
  fft(fb);

  CArray fc(N);
  for (size_t i = 0; i < N; ++i) fc[i] = fa[i] * std::conj(fb[i]);

  fc = fc.apply(std::conj);
  fft(fc);
  fc = fc.apply(std::conj);
  for (auto& v : fc) v /= static_cast<double>(N);

  int    best_offset = 0;
  double best_score  = -1e18;
  for (size_t i = 0; i < N; ++i) {
    if (fc[i].real() > best_score) {
      best_score  = fc[i].real();
      best_offset = static_cast<int>(i);
    }
  }
  // FFT offsets in [N/2, N) represent negative shifts
  if (best_offset > static_cast<int>(N / 2)) best_offset -= static_cast<int>(N);

  return {best_offset, best_score};
}

/**
 * @brief Replaces all whitespace characters in a FASTA header with underscores.
 *
 * This function is retained for backwards compatibility but is no longer used
 * as the primary header normalisation mechanism. It simply replaces any
 * whitespace with an underscore. New code should prefer
 * canonicalize_fasta_header() which performs more robust parsing.
 *
 * @param[in,out] header FASTA header string to sanitize in place.
 */
void sanitize_header(std::string& header) {
  for (char& c : header) {
    if (isspace(static_cast<unsigned char>(c))) {
      c = '_';
    }
  }
}

/**
 * @brief Produce a canonical identifier for a FASTA header.
 *
 * The canonical name is used internally to identify sequences and to label
 * output files. It is derived from the first whitespace‐delimited token of
 * the header (or the UniProt accession for protein sequences) and then
 * cleansed of characters that could confuse Newick parsers or file systems.
 *
 * Behaviour by mode:
 *  - **Protein mode**: If the header contains pipe characters (e.g.
 *    "sp|P12345|NAME"), extract the accession between the first two pipes.
 *    Otherwise take the first token before any whitespace. The accession
 *    characters are then mapped: letters, digits, underscores and hyphens
 *    are kept as–is; periods (`.`) and other punctuation are replaced
 *    with underscores.
 *  - **DNA mode**: Take the first token before any whitespace. Do **not**
 *    split on underscores or other punctuation. All characters other than
 *    alphanumerics, underscores or hyphens are replaced with underscores.
 *    Dots are replaced with underscores to avoid confusion with branch
 *    lengths in Newick strings.
 *
 * In both modes, leading '>' is removed if present and leading/trailing
 * whitespace is trimmed. Internal whitespace is not expected in the token
 * because splitting stops at the first whitespace character. Characters
 * ":", ",", "(", ")" and ";" are converted to underscores.
 *
 * @param header Raw FASTA header line without the leading '>'
 * @param mode Scoring mode specifying DNA or protein behaviour
 * @return Canonicalised header string safe for internal use
 */
std::string canonicalize_fasta_header(std::string header, ScoreMode mode) {
  // Trim leading '>' if provided inadvertently
  if (!header.empty() && header[0] == '>') {
    header.erase(header.begin());
  }
  // Trim leading whitespace
  auto lpos = header.find_first_not_of(" \t\r\n");
  auto rpos = header.find_last_not_of(" \t\r\n");
  if (lpos == std::string::npos) {
    header.clear();
  } else {
    header = header.substr(lpos, rpos - lpos + 1);
  }
  std::string token;
  if (mode == MODE_PROTEIN) {
    // UniProt style: sp|P12345|NAME
    size_t p1 = header.find('|');
    size_t p2 = (p1 == std::string::npos ? std::string::npos : header.find('|', p1 + 1));
    if (p1 != std::string::npos && p2 != std::string::npos && p2 > p1 + 1) {
      token = header.substr(p1 + 1, p2 - p1 - 1);
    } else {
      // Otherwise take the first token before whitespace
      size_t ws = header.find_first_of(" \t\r\n");
      token = (ws != std::string::npos) ? header.substr(0, ws) : header;
    }
  } else {
    // DNA mode: first token before whitespace
    size_t ws = header.find_first_of(" \t\r\n");
    token = (ws != std::string::npos) ? header.substr(0, ws) : header;
  }
  // Build canonical name by replacing unsafe characters
  std::string canonical;
  canonical.reserve(token.size());
  for (char c : token) {
    // Accept letters, digits, underscore and hyphen directly
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') {
      canonical.push_back(c);
    } else if (c == '.') {
      // Replace dot with underscore to disambiguate accession dot from branch length
      canonical.push_back('_');
    } else {
      // Replace all other characters (including punctuation and whitespace) with underscore
      canonical.push_back('_');
    }
  }
  return canonical;
}

/**
 * @brief Generate a list of stable internal labels for tree construction.
 *
 * Internal labels take the form "SEQ_0", "SEQ_1", ... up to n-1. These
 * labels are used inside the guide tree to avoid ambiguities caused by
 * biological headers containing punctuation, whitespace or duplicate
 * substrings. The mapping from internal label to original sequence index
 * is implicit via the suffix number.
 *
 * @param n Number of sequences
 * @return Vector of size @p n containing labels SEQ_0 .. SEQ_{n-1}
 */
std::vector<std::string> make_internal_tree_labels(std::size_t n) {
  std::vector<std::string> labels;
  labels.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    labels.emplace_back("SEQ_" + std::to_string(i));
  }
  return labels;
}

/**
 * @brief Verify that a set of headers are unique.
 *
 * Throws a runtime_error if any duplicate header is found. This ensures the
 * downstream mapping from header to sequence index is one-to-one.
 *
 * @param hdrs Vector of header strings
 */
void validate_headers_unique(const std::vector<std::string>& hdrs) {
  std::unordered_set<std::string> seen;
  for (const auto& h : hdrs) {
    if (!seen.insert(h).second) {
      throw std::runtime_error("Duplicate header detected: " + h);
    }
  }
}

/**
 * @brief Validate that the final MSA has the expected number of rows.
 *
 * If the number of rows differs from @p expected_rows an exception is
 * thrown. This prevents silent production of incomplete alignments when
 * guide tree parsing fails.
 *
 * @param msa Vector of aligned sequences
 * @param expected_rows Expected number of rows corresponding to input sequences
 */
void validate_msa_shape(const std::vector<std::string>& msa,
                        std::size_t                    expected_rows) {
  if (msa.size() != expected_rows) {
    std::ostringstream oss;
    oss << "MSA row count (" << msa.size()
        << ") != expected header count (" << expected_rows << ")";
    throw std::runtime_error(oss.str());
  }
}

/**
 * @brief Escape special characters for safe inclusion in HTML.
 *
 * Converts &, <, >, " and ' into their corresponding HTML entities. This
 * prevents malformed HTML when sequence headers contain special characters.
 *
 * @param s Input string
 * @return Escaped string safe to embed in HTML
 */
static std::string htmlEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '&':
        out += "&amp;";
        break;
      case '<':
        out += "&lt;";
        break;
      case '>':
        out += "&gt;";
        break;
      case '"':
        out += "&quot;";
        break;
      case '\'':
        out += "&#39;";
        break;
      default:
        out.push_back(c);
    }
  }
  return out;
}

/**
 * @brief Reads a FASTA file and extracts the first sequence record.
 *
 * Only the first header line (the line beginning with @c '>') is captured.
 * Sequence lines following that header are concatenated into a single
 * uppercase string with non-alphabetic characters silently discarded.
 * If a second header line is encountered, the function stops reading
 * further records. This enforces the one-sequence-per-file convention
 * expected by the CLI.
 *
 * @param fn   Path to the FASTA file to read.
 * @param[out] hdr Header string without the leading @c '>' character.
 * @param[out] seq Uppercase sequence with all non-alpha characters removed.
 * @throws std::runtime_error if the file cannot be opened, if no header
 *         is found or if the first record contains no sequence.
 */
void processFasta(const std::string& fn, std::string& hdr, std::string& seq) {
  std::ifstream f(fn);
  if (!f) throw std::runtime_error("Cannot open " + fn);
  hdr.clear();
  seq.clear();
  std::string line;
  bool        gotHdr = false;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    if (line[0] == '>') {
      if (!gotHdr) {
        // Capture the first header without the leading '>'
        hdr    = line.substr(1);
        gotHdr = true;
      } else {
        // Second header encountered – stop reading further records
        break;
      }
      continue;
    }
    if (!gotHdr) {
      // Skip any leading sequence lines before the first header
      continue;
    }
    // Append only alphabetic characters (converted to uppercase) to the sequence
    for (char c : line) {
      if (std::isalpha(static_cast<unsigned char>(c))) {
        seq.push_back(std::toupper(static_cast<unsigned char>(c)));
      }
    }
  }
  // Validate that we found both a header and a sequence
  if (!gotHdr) {
    throw std::runtime_error("No FASTA header found in " + fn);
  }
  if (seq.empty()) {
    throw std::runtime_error("No sequence found in the first record of " + fn);
  }
}

/**
 * @brief Computes the final score row of a global NW alignment using an
 *        affine gap penalty model (score-only, no traceback).
 *
 * Implements the standard three-matrix (S, E, F) recurrence using two
 * rolling rows, consuming O(n) space. Used as a lightweight score probe
 * when only the last row of the DP table is needed (e.g. as a fallback
 * diagnostic). For production alignment use @c hirsch_forward or
 * @c computeGlobalAlignment.
 *
 * @note The F-matrix recurrence in this version has a known redundancy
 *       (@c fj is always @c INT_MIN/2) ? it is preserved for compatibility
 *       and replaced by @c hirsch_forward in the hot path.
 *
 * @param x    First sequence (ungapped, uppercase).
 * @param y    Second sequence (ungapped, uppercase).
 * @param mode Scoring mode ? passed for API consistency, dispatched via @p fn.
 * @param fn   Substitution scoring function pointer.
 * @param p    Affine gap penalty parameters.
 * @return     Vector of length @c n+1 containing @c S[m][0..n], the final
 *             row of the NW score matrix. @c S[m][n] is the optimal score.
 */
std::vector<int> nw_score_row(const std::string& x, const std::string& y,
                              ScoreMode mode, ScoreFn fn,
                              const AlignParams& p) {
  const int m     = checked_size_to_int(x.size(), "x.size()");
  const int n     = checked_size_to_int(y.size(), "y.size()");
  const int iOpen = static_cast<int>(std::round(p.gap_open));
  const int iExt  = static_cast<int>(std::round(p.gap_extend));

  // ?? Opt 5: reserve exact capacity, no realloc ???????????????????????????
  std::vector<int> S(n + 1), E(n + 1);
  S.reserve(n + 1);
  E.reserve(n + 1);

  S[0] = 0;
  for (int j = 1; j <= n; ++j) {
    E[j] = (j == 1) ? iOpen + iExt : E[j - 1] + iExt;
    S[j] = E[j];
  }

  // ?? Opt 3: build per-row score table once, avoid fn-ptr call in j-loop ??
  alignas(32) int row_scores[256];

  for (int i = 1; i <= m; ++i) {
    // Opt 3: populate lookup for x[i-1] vs all y chars
    for (int c = 0; c < 256; ++c)
      row_scores[c] = fn(x[i - 1], static_cast<char>(c));

    std::vector<int> Snew(n + 1), Enew(n + 1);
    Snew[0]  = S[0] + iOpen + iExt;
    Enew[0]  = INT_MIN / 2;
    int Fval = Snew[0];

    for (int j = 1; j <= n; ++j) {
      Fval    = std::max(Snew[j - 1] + iOpen, Fval + iExt);
      int ej  = std::max(Snew[j - 1] + iOpen, Enew[j - 1] + iExt);
      Enew[j] = ej;
      // Opt 3: table lookup replaces fn() call
      int diag = S[j - 1] + row_scores[static_cast<unsigned char>(y[j - 1])];
      Snew[j]  = std::max({diag, ej, Fval});
    }
    S = std::move(Snew);
    E = std::move(Enew);
  }
  return S;
}

/**
 * @brief Computes the forward NW score row (top half) for Hirschberg's
 *        algorithm using AVX2 SIMD and a gather-based score lookup.
 *
 * Fills rows @f$ 0 \ldots |x| @f$ of the DP table using two rolling rows
 * (O(n) space). The match score inner loop batches 8 lookups per AVX2
 * gather, while E and F are computed scalarly due to sequential dependency.
 * A per-row score table is built once per outer iteration to eliminate
 * repeated function-pointer calls.
 *
 * @param x     First (top) sequence ? rows 0 to |x|.
 * @param y     Second (column) sequence ? columns 0 to |y|.
 * @param fn    Substitution scoring function pointer.
 * @param iOpen Gap-open penalty (integer, negative).
 * @param iExt  Gap-extend penalty (integer, negative).
 * @return      Score vector of length @c |y|+1 representing the last filled
 * row.
 */
static std::vector<int> hirsch_forward(const std::string& x,
                                       const std::string& y, ScoreFn fn,
                                       int iOpen, int iExt) {
  const int m  = checked_size_to_int(x.size(), "x.size()");
  const int n  = checked_size_to_int(y.size(), "y.size()");
  const int n8 = (n + 7) & ~7;

  // ?? Opt 5: reserve with alignment padding ???????????????????????????????
  std::vector<int> cur(n8 + 2, INT_MIN / 2);
  std::vector<int> prev(n8 + 2, INT_MIN / 2);
  std::vector<int> E(n8 + 2, INT_MIN / 2);
  cur.reserve(n8 + 2);
  prev.reserve(n8 + 2);

  const __m256i vOpen = _mm256_set1_epi32(iOpen);

  // ?? Opt 3: per-row score lookup table (256 entries, built once per i) ???
  alignas(32) int row_scores[256];

  prev[0] = 0;
  E[0]    = INT_MIN / 2;
  for (int j = 1; j <= n; ++j) {
    E[j]    = (j == 1) ? iOpen + iExt : E[j - 1] + iExt;
    prev[j] = E[j];
  }

  for (int i = 1; i <= m; ++i) {
    // Opt 3: build score row for x[i-1] vs every possible character
    for (int c = 0; c < 256; ++c)
      row_scores[c] = fn(x[i - 1], static_cast<char>(c));

    cur[0]   = (i == 1) ? iOpen + iExt : prev[0] + iExt;
    int Fval = cur[0];
    E[0]     = INT_MIN / 2;

    for (int j = 1; j <= n8; j += 8) {
      // ?? Opt 2 (gather): load y characters and gather scores from table ??
      alignas(32) int y_chars[8] = {};
      for (int k = 0; k < 8; ++k) {
        int jk = j + k;
        if (jk <= n) y_chars[k] = static_cast<unsigned char>(y[jk - 1]);
      }
      __m256i vIdx = _mm256_loadu_si256((__m256i*)y_chars);
      // Opt 2: _mm256_i32gather_epi32 replaces 8 fn() calls
      __m256i vScores = _mm256_i32gather_epi32(row_scores, vIdx, 4);

      // ?? Opt AVX2: M = prev[j-1..j+6] + gathered scores ?????????????????
      __m256i vPrevDiag = _mm256_loadu_si256((__m256i*)&prev[j - 1]);
      __m256i vM        = _mm256_add_epi32(vPrevDiag, vScores);

      // ?? Opt AVX2: F-open candidates = prev[j..j+7] + iOpen ??????????????
      __m256i vPrevJ = _mm256_loadu_si256((__m256i*)&prev[j]);
      __m256i vFopen = _mm256_add_epi32(vPrevJ, vOpen);

      alignas(32) int Mcand[8], Fopen_cand[8];
      _mm256_storeu_si256((__m256i*)Mcand, vM);
      _mm256_storeu_si256((__m256i*)Fopen_cand, vFopen);

      // ?? Scalar E/F merge (sequential dependency along j) ?????????????????
      for (int k = 0; k < 8; ++k) {
        int jk = j + k;
        if (__builtin_expect(jk > n, 0)) break;  // Opt 7: branch hint

        Fval    = std::max(Fopen_cand[k], Fval + iExt);
        int Ej  = std::max(cur[jk - 1] + iOpen, E[jk - 1] + iExt);
        E[jk]   = Ej;
        cur[jk] = std::max({Mcand[k], Ej, Fval});
      }
    }
    std::swap(cur, prev);
    std::fill(cur.begin(), cur.end(), INT_MIN / 2);
  }
  return std::vector<int>(prev.begin(), prev.begin() + n + 1);
}

/**
 * @brief Computes the reverse NW score row (bottom half) for Hirschberg's
 *        algorithm using AVX2 SIMD and a gather-based score lookup.
 *
 * Mirrors @c hirsch_forward but traverses rows from @f$ |x|-1 \ldots 0 @f$
 * and columns right-to-left. The resulting score vector, when added
 * element-wise to the forward score vector, identifies the optimal split
 * column for the Hirschberg divide step.
 *
 * @param x     Second (bottom) sequence ? rows |x|-1 down to 0.
 * @param y     Second (column) sequence ? columns |y| down to 0.
 * @param fn    Substitution scoring function pointer.
 * @param iOpen Gap-open penalty (integer, negative).
 * @param iExt  Gap-extend penalty (integer, negative).
 * @return      Score vector of length @c |y|+1 representing the last filled
 *              row in the reverse direction.
 */
static std::vector<int> hirsch_reverse(const std::string& x,
                                       const std::string& y, ScoreFn fn,
                                       int iOpen, int iExt) {
  const int m  = checked_size_to_int(x.size(), "x.size()");
  const int n  = checked_size_to_int(y.size(), "y.size()");
  const int n8 = (n + 7) & ~7;

  // ?? Opt 5: pre-sized with alignment padding ??????????????????????????????
  std::vector<int> cur(n8 + 2, INT_MIN / 2);
  std::vector<int> prev(n8 + 2, INT_MIN / 2);
  std::vector<int> E(n8 + 2, INT_MIN / 2);

  // ?? Opt 3: per-row score table ???????????????????????????????????????????
  alignas(32) int row_scores[256];

  prev[n] = 0;
  for (int j = n - 1; j >= 0; --j) {
    E[j]    = (j == n - 1) ? iOpen + iExt : E[j + 1] + iExt;
    prev[j] = E[j];
  }

  for (int i = m - 1; i >= 0; --i) {
    // Opt 3: build score row for x[i] vs every possible character
    for (int c = 0; c < 256; ++c)
      row_scores[c] = fn(x[i], static_cast<char>(c));

    cur[n]   = (i == m - 1) ? iOpen + iExt : prev[n] + iExt;
    int Fval = cur[n];
    E[n]     = INT_MIN / 2;

    for (int j = n - 1; j >= 0; j -= 8) {
      const int jstart = std::max(0, j - 7);
      const int len    = j - jstart + 1;

      // ?? Opt 2 (gather): load y chars right-to-left, gather scores ????????
      alignas(32) int y_chars[8] = {};
      for (int k = 0; k < len; ++k) {
        int jk     = j - k;
        y_chars[k] = static_cast<unsigned char>(y[jk]);
      }
      __m256i vIdx    = _mm256_loadu_si256((__m256i*)y_chars);
      __m256i vScores = _mm256_i32gather_epi32(row_scores, vIdx, 4);

      // ?? Opt AVX2: M = prev[jstart+1..j+1] + scores ???????????????????????
      __m256i vPrevDiag = _mm256_loadu_si256((__m256i*)&prev[jstart + 1]);
      __m256i vM        = _mm256_add_epi32(vPrevDiag, vScores);

      alignas(32) int Mcand[8];
      _mm256_storeu_si256((__m256i*)Mcand, vM);

      // ?? Scalar E/F merge (sequential right-to-left dependency) ???????????
      for (int k = 0; k < len; ++k) {
        int jk = j - k;
        Fval   = std::max(prev[jk] + iOpen, Fval + iExt);
        int Ej = std::max(cur[jk + 1] + iOpen, E[jk + 1] + iExt);
        E[jk]  = Ej;
        // Mcand is computed left-to-right but j-loop is right-to-left
        cur[jk] = std::max({Mcand[len - 1 - k], Ej, Fval});
      }
    }
    std::swap(cur, prev);
    std::fill(cur.begin(), cur.end(), INT_MIN / 2);
  }
  return std::vector<int>(prev.begin(), prev.begin() + n + 1);
}

/**
 * @brief Computes global pairwise alignment via Hirschberg's divide-and-conquer
 *        algorithm with AVX2 split-point search and OpenMP task parallelism.
 *
 * Achieves O(mn) time and O(n) space by recursively splitting the first
 * sequence at its midpoint, computing forward and reverse score rows for
 * the two halves, and identifying the optimal column split point via a
 * vectorized argmax. The two resulting subproblems are dispatched as
 * independent OpenMP tasks up to @c MAX_TASK_DEPTH levels of recursion.
 *
 * Base cases:
 *  - @c m==0 : fill @p ax with n gaps, copy @p y to @p ay.
 *  - @c n==0 : copy @p x to @p ax, fill @p ay with m gaps.
 *  - @c m==1 : place @c x[0] at the column maximising match score minus
 *              flanking gap penalties; surround with gap characters.
 *
 * @param x      First sequence (ungapped, uppercase).
 * @param y      Second sequence (ungapped, uppercase).
 * @param fn     Substitution scoring function pointer.
 * @param iOpen  Gap-open penalty (integer, negative).
 * @param iExt   Gap-extend penalty (integer, negative).
 * @param[out] ax  Aligned version of @p x, appended to in place.
 * @param[out] ay  Aligned version of @p y, appended to in place.
 * @param depth  Recursion depth used to cap OpenMP task spawning.
 *               Pass 0 at the top-level call (default).
 */
void hirschberg(const std::string& x, const std::string& y, ScoreFn fn,
                int iOpen, int iExt, std::string& ax, std::string& ay,
                int depth = 0) {
  const int m = checked_size_to_int(x.size(), "x.size()");
  const int n = checked_size_to_int(y.size(), "y.size()");

  // ?? Base cases ????????????????????????????????????????????????????????????
  if (m == 0) {
    ax += std::string(n, '-');
    ay += y;
    return;
  }
  if (n == 0) {
    ax += x;
    ay += std::string(m, '-');
    return;
  }
  if (m == 1) {
    // Opt 7: branch hint ? best_j == 0 is common for very short y
    int best_j = 0, best_score = INT_MIN;
    for (int j = 0; j < n; ++j) {
      int left  = (j == 0) ? 0 : iOpen + j * iExt;
      int right = (j == n - 1) ? 0 : iOpen + (n - j - 1) * iExt;
      int s     = left + fn(x[0], y[j]) + right;
      if (__builtin_expect(s > best_score, 0)) {
        best_score = s;
        best_j     = j;
      }
    }
    // Opt 5: reserve before appending to avoid realloc in recursion
    ax.reserve(ax.size() + n);
    ay.reserve(ay.size() + n);
    if (best_j > 0) ax += std::string(best_j, '-');
    ax += x[0];
    if (best_j < n - 1) ax += std::string(n - best_j - 1, '-');
    ay += y;
    return;
  }

  const int mid = m / 2;

  // ?? Forward and reverse score rows ???????????????????????????????????????
  auto fwd = hirsch_forward(x.substr(0, mid), y, fn, iOpen, iExt);
  auto bwd = hirsch_reverse(x.substr(mid), y, fn, iOpen, iExt);

  // ?? Opt 2: AVX2 vectorized argmax for split-point search ?????????????????
  int split_j = 0, best = INT_MIN;
  {
    const int n8 = (n + 1 + 7) & ~7;

    // Opt 5: reserve padded vectors once
    std::vector<int> fwd8(n8, INT_MIN / 2);
    std::vector<int> bwd8(n8, INT_MIN / 2);
    for (int j = 0; j <= n; ++j) {
      fwd8[j] = fwd[j];
      bwd8[j] = bwd[j];
    }

    __m256i       vBest  = _mm256_set1_epi32(INT_MIN);
    __m256i       vBestJ = _mm256_set1_epi32(-1);
    __m256i       vIdx   = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i v8     = _mm256_set1_epi32(8);

    for (int j = 0; j < n8; j += 8) {
      __m256i vF   = _mm256_loadu_si256((__m256i*)&fwd8[j]);
      __m256i vB   = _mm256_loadu_si256((__m256i*)&bwd8[j]);
      __m256i vS   = _mm256_add_epi32(vF, vB);
      __m256i mask = _mm256_cmpgt_epi32(vS, vBest);
      vBest        = _mm256_blendv_epi8(vBest, vS, mask);
      vBestJ       = _mm256_blendv_epi8(vBestJ, vIdx, mask);
      vIdx         = _mm256_add_epi32(vIdx, v8);
    }

    alignas(32) int best_arr[8], bestj_arr[8];
    _mm256_storeu_si256((__m256i*)best_arr, vBest);
    _mm256_storeu_si256((__m256i*)bestj_arr, vBestJ);
    for (int k = 0; k < 8; ++k) {
      if (best_arr[k] > best && bestj_arr[k] <= n) {
        best    = best_arr[k];
        split_j = bestj_arr[k];
      }
    }
  }

  // ?? Opt 5: reserve output strings before recursive append ?????????????????
  std::string ax1, ay1, ax2, ay2;
  ax1.reserve(mid + split_j);
  ay1.reserve(mid + split_j);
  ax2.reserve((m - mid) + (n - split_j));
  ay2.reserve((m - mid) + (n - split_j));

  // ?? Opt 6 + OpenMP tasks: cap depth to avoid excessive task overhead ??????
  constexpr int MAX_TASK_DEPTH = 4;  // 2^4 = 16 concurrent tasks max

#pragma omp task shared(ax1, ay1) if (depth < MAX_TASK_DEPTH)
  hirschberg(x.substr(0, mid), y.substr(0, split_j), fn, iOpen, iExt, ax1, ay1,
             depth + 1);

#pragma omp task shared(ax2, ay2) if (depth < MAX_TASK_DEPTH)
  hirschberg(x.substr(mid), y.substr(split_j), fn, iOpen, iExt, ax2, ay2,
             depth + 1);

#pragma omp taskwait

  // Opt 5: reserve final concatenation
  ax.reserve(ax.size() + ax1.size() + ax2.size());
  ay.reserve(ay.size() + ay1.size() + ay2.size());
  ax += ax1;
  ax += ax2;
  ay += ay1;
  ay += ay2;
}

/**
 * @brief Computes a global pairwise alignment using affine-gap Needleman-Wunsch
 *        with AVX2 SIMD vectorization.
 *
 * Implements the three-matrix recurrence (S, E, F) for affine gap costs:
 * @f[
 *   E_{i,j} = \max(S_{i,j-1} + g_o,\; E_{i,j-1} + g_e)
 * @f]
 * @f[
 *   F_{i,j} = \max(S_{i-1,j} + g_o,\; F_{i-1,j} + g_e)
 * @f]
 * @f[
 *   S_{i,j} = \max(S_{i-1,j-1} + \sigma(x_i, y_j),\; E_{i,j},\; F_{i,j})
 * @f]
 * where @f$ g_o @f$ = @c p.gap_open and @f$ g_e @f$ = @c p.gap_extend.
 *
 * The inner loop processes 8 columns per iteration using 256-bit AVX2
 * integer intrinsics (@c _mm256_max_epi32, @c _mm256_add_epi32). Gap
 * penalties are rounded to the nearest integer before use in the DP.
 * A full traceback matrix is stored and walked after the fill phase.
 *
 * @param x         First sequence (ungapped, uppercase).
 * @param y         Second sequence (ungapped, uppercase).
 * @param mode      Scoring mode (@c MODE_DNA or @c MODE_PROTEIN), forwarded
 *                  to @p score_fn for documentation but not used directly.
 * @param score_fn  Substitution scoring function pointer (e.g. @c edna_score
 *                  or @c blosum62_score).
 * @param[out] aligned_x  Aligned version of @p x with gap characters inserted.
 * @param[out] aligned_y  Aligned version of @p y with gap characters inserted.
 * @param p         Affine gap penalty parameters.
 * @return          Integer alignment score of the optimal global alignment.
 */
int computeGlobalAlignment(const std::string& x, const std::string& y,
                           ScoreMode mode, ScoreFn score_fn,
                           std::string& aligned_x, std::string& aligned_y,
                           const AlignParams& p) {
  const int m          = checked_size_to_int(x.size(), "x.size()");
  const int n          = checked_size_to_int(y.size(), "y.size()");
  const int iGapOpen   = static_cast<int>(std::round(p.gap_open));
  const int iGapExtend = static_cast<int>(std::round(p.gap_extend));

  constexpr int HIRSCHBERG_THRESHOLD = 10000;

  if (m > HIRSCHBERG_THRESHOLD || n > HIRSCHBERG_THRESHOLD) {
    aligned_x.clear();
    aligned_y.clear();
    aligned_x.reserve(m + n);  // worst-case alignment length
    aligned_y.reserve(m + n);

#pragma omp parallel if (!omp_in_parallel()) default(none) \
    shared(x, y, score_fn, iGapOpen, iGapExtend, aligned_x, aligned_y)
#pragma omp single
    hirschberg(x, y, score_fn, iGapOpen, iGapExtend, aligned_x, aligned_y);

    int  score_val = 0;
    bool in_gap_x = false, in_gap_y = false;
    for (size_t i = 0; i < aligned_x.size(); ++i) {
      bool gx = (aligned_x[i] == '-');
      bool gy = (aligned_y[i] == '-');
      if (!gx && !gy) {
        score_val += score_fn(aligned_x[i], aligned_y[i]);
        in_gap_x = in_gap_y = false;
      } else if (gx && !gy) {
        score_val += in_gap_x ? iGapExtend : iGapOpen + iGapExtend;
        in_gap_x = true;
        in_gap_y = false;
      } else if (!gx && gy) {
        score_val += in_gap_y ? iGapExtend : iGapOpen + iGapExtend;
        in_gap_y = true;
        in_gap_x = false;
      }
      // double-gap columns skipped (invalid in pairwise alignment)
    }
    return score_val;
  }

  // ?? Original AVX2 path (short sequences) ????????????????????????????????
  int n8 = (n + 7) & ~7;

  std::vector<int>  S_prev(n8 + 1), S_cur(n8 + 1);
  std::vector<int>  E_cur(n8 + 1), F_prev(n8 + 1);
  std::vector<char> trace_buf((m + 1) * (n + 1));
  trace_buf[0] = 'M';

  const __m256i vGapOpen   = _mm256_set1_epi32(iGapOpen);
  const __m256i vGapExtend = _mm256_set1_epi32(iGapExtend);

  S_prev[0] = 0;
  for (int j = 1; j <= n; ++j) {
    int e        = (j == 1 ? S_prev[j - 1] + iGapOpen
                           : E_cur[j - 1] + iGapExtend);  // ? BUG 1 fixed
    S_prev[j]    = e;
    E_cur[j]     = e;
    F_prev[j]    = INT_MIN / 2;
    trace_buf[j] = (j == 1 ? 'E' : 'e');
  }
  for (int j = n + 1; j <= n8; ++j) {
    S_prev[j] = INT_MIN / 2;
    E_cur[j]  = INT_MIN / 2;
    F_prev[j] = INT_MIN / 2;
  }

  // temp arrays for vector loads/stores
  int* Sp = S_prev.data();
  int* Sc = S_cur.data();
  int* Ec = E_cur.data();
  int* Fp = F_prev.data();

  for (int i = 1; i <= m; ++i) {
    // scalar first column
    int openF                  = Sp[0] + iGapOpen;
    int extF                   = Fp[0] + iGapExtend;
    Sc[0]                      = std::max(openF, extF);
    Ec[0]                      = INT_MIN / 2;
    Fp[0]                      = Sc[0];
    trace_buf[i * (n + 1) + 0] = (Sc[0] == openF ? 'F' : 'f');

    // broadcast S_prev[i-1][j-1] lanewise
    __m256i vSdiag = _mm256_set1_epi32(Sp[0]);

    // vectorized inner loop: j=1..n8 step 8
    for (int j = 1; j <= n8; j += 8) {
      // load previous vectors
      __m256i vSp = _mm256_loadu_si256((__m256i*)&Sp[j]);
      __m256i vFp = _mm256_loadu_si256((__m256i*)&Fp[j]);
      __m256i vEc =
          _mm256_loadu_si256((__m256i*)&Ec[j - 1]);  // note shift for E
      __m256i vScm1 =
          _mm256_loadu_si256((__m256i*)&Sc[j - 1]);  // current S[j-1]

      // compute F = max(S_prev + gapOpen, F_prev + gapExtend)
      __m256i vF = _mm256_max_epi32(_mm256_add_epi32(vSp, vGapOpen),
                                    _mm256_add_epi32(vFp, vGapExtend));

      // compute E = max(S_cur[j-1] + gapOpen, E_cur[j-1] + gapExtend)
      __m256i vE = _mm256_max_epi32(_mm256_add_epi32(vScm1, vGapOpen),
                                    _mm256_add_epi32(vEc, vGapExtend));

      // build vSdiag: prev_S[j-1], prev_F[j-1], prev_E[j-1]
      __m256i vFp_shift = _mm256_loadu_si256((__m256i*)&Fp[j - 1]);
      __m256i vEp_shift = _mm256_loadu_si256((__m256i*)&Ec[j - 1]);
      __m256i vBestPrev =
          _mm256_max_epi32(_mm256_max_epi32(vSdiag, vFp_shift), vEp_shift);

      // compute match/mismatch scores vector
      int scores[8];
      for (int k = 0; k < 8; ++k) {
        int idx = j + k - 1;
        int sc  = 0;
        if (idx < n) sc = score_fn(x[i - 1], y[idx]);
        scores[k] = sc;
      }
      __m256i vMatch = _mm256_loadu_si256((__m256i*)scores);

      // M = BestPrev + match
      __m256i vM = _mm256_add_epi32(vBestPrev, vMatch);

      // S = max(M, E, F)
      __m256i vS = _mm256_max_epi32(_mm256_max_epi32(vM, vE), vF);

      // store results
      _mm256_storeu_si256((__m256i*)&Sc[j], vS);
      _mm256_storeu_si256((__m256i*)&Fp[j], vF);
      _mm256_storeu_si256((__m256i*)&Ec[j], vE);

      // prepare next vSdiag = prev_S[j..j+7]
      vSdiag = vSp;

      // scalar traceback for this block
      int Sblock[8], Eblock[8], Fblock[8];
      _mm256_storeu_si256((__m256i*)Sblock, vS);
      _mm256_storeu_si256((__m256i*)Eblock, vE);
      _mm256_storeu_si256((__m256i*)Fblock, vF);
      for (int k = 0; k < 8; ++k) {
        if (j + k <= n) {
          char ptr = 'M';  // Default to Match
          // The highest score determines the path.
          // avoids the out-of-bounds read.
          if (Sblock[k] == Eblock[k]) {
            ptr = 'E';  // Gap in sequence X (Insertion)
          } else if (Sblock[k] == Fblock[k]) {
            ptr = 'F';  // Gap in sequence Y (Deletion)
          }
          trace_buf[i * (n + 1) + (j + k)] = ptr;
        }
      }
    }

    // swap rows
    std::swap(Sp, Sc);
    std::swap(Fp, Ec);
  }

  // final score
  int finalScore = Sp[n];

  // traceback
  aligned_x.clear();
  aligned_y.clear();
  int i = m, j = n;
  while (i > 0 || j > 0) {
    if (char path = trace_buf[i * (n + 1) + j]; path == 'M') {
      aligned_x.push_back(x[i - 1]);
      aligned_y.push_back(y[j - 1]);
      --i;
      --j;
    } else if (path == 'F' || path == 'f') {
      aligned_x.push_back(x[i - 1]);
      aligned_y.push_back('-');
      --i;
    } else if (path == 'E' || path == 'e') {
      aligned_x.push_back('-');
      aligned_y.push_back(y[j - 1]);
      --j;
    } else {
      if (i > 0) {
        aligned_x.push_back(x[i - 1]);
        aligned_y.push_back('-');
        --i;
      } else {
        aligned_x.push_back('-');
        aligned_y.push_back(y[j - 1]);
        --j;
      }
    }
  }
  std::reverse(aligned_x.begin(), aligned_x.end());
  std::reverse(aligned_y.begin(), aligned_y.end());
  return finalScore;
}

/**
 * @brief Derives a consensus sequence from a multiple sequence alignment
 * profile.
 *
 * For each alignment column, counts the frequency of each non-gap character
 * across all sequences and selects the plurality character. Columns where all
 * sequences contain a gap yield @c '-' in the consensus.
 *
 * @param profile  Vector of equally-length aligned sequence strings.
 * For each alignment column, counts the frequency of each non-gap character
 * across all sequences and selects the plurality character. Columns where all
 * sequences contain a gap yield @c '-' in the consensus.
 * @return         Consensus string of the same length as the alignment columns.
 *                 Returns an empty string if @p profile is empty.
 */
std::string generate_consensus(const std::vector<std::string>& profile) {
  if (profile.empty() || profile[0].empty()) {
    return "";
  }
  std::string consensus;
  const int   align_len =
      checked_size_to_int(profile[0].size(), "profile[0].size()");
  const int num_seqs = checked_size_to_int(profile.size(), "profile.size()");

  for (int j = 0; j < align_len; ++j) {
    std::array<int, 256> counts{};  // Initialize all counts to 0
    int                  max_count = 0;
    char                 best_char = '-';

    for (int i = 0; i < num_seqs; ++i) {
      char c = profile[i][j];
      if (c != '-') {
        counts[static_cast<unsigned char>(c)]++;
        if (counts[static_cast<unsigned char>(c)] > max_count) {
          max_count = counts[static_cast<unsigned char>(c)];
          best_char = c;
        }
      }
    }
    consensus += best_char;
  }
  return consensus;
}

/**
 * @brief Computes the Sum-of-Pairs (SP) score of a multiple sequence alignment
 *        under an affine gap penalty model.
 *
 * For every unique ordered pair of sequences @f$(i, k)@f$, the score is:
 * @f[
 *   \text{SP} = \sum_{i < k} \sum_{j} \text{col\_score}(i, k, j)
 * @f]
 * where each column contributes one of three cases:
 *  - **Residue vs residue**: @f$ \sigma(c_i, c_k) @f$ from the substitution
 * matrix.
 *  - **Residue vs gap (gap opening)**: @f$ g_o + g_e @f$
 *  - **Residue vs gap (gap extension)**: @f$ g_e @f$
 *  - **Gap vs gap**: 0 (no penalty for double-gap columns).
 *
 * Gap state is tracked independently for each pair so that parallel gaps
 * in different pairs do not interfere.
 *
 * @param msa   Vector of equally-length aligned sequences.
 * @param mode  Scoring mode passed to @c score().
 * @param fn    Substitution scoring function pointer.
 * @param p     Affine gap penalty parameters.
 * @return      Total SP score as a @c long long. Higher is better.
 */
long long calculate_sp_score(const std::vector<std::string>& msa,
                             ScoreMode mode, ScoreFn fn, const AlignParams& p) {
  if (msa.empty() || msa[0].empty()) {
    return 0;
  }

  const auto iGapOpen   = static_cast<long long>(std::round(p.gap_open));
  const auto iGapExtend = static_cast<long long>(std::round(p.gap_extend));

  long long total_score = 0;
  const int num_seqs    = checked_size_to_int(msa.size(), "msa.size()");
  const int align_len   = checked_size_to_int(msa[0].size(), "msa[0].size()");

  // Mode is not used directly here because the scoring behaviour is
  // encapsulated in the function pointer fn. Avoid unused parameter warning.
  (void)mode;

  // Iterate over every unique pair of sequences in the alignment (i and k)
  for (int i = 0; i < num_seqs; ++i) {
    for (int k = i + 1; k < num_seqs; ++k) {
      // For each pair, we must track the gap state independently.
      // A gap between seqs i and k is independent of a gap between i and j.
      bool in_gap_for_this_pair = false;

      // Now, score this specific pair across all columns of the alignment.
      for (int j = 0; j < align_len; ++j) {
        char char_i = msa[i][j];
        char char_k = msa[k][j];

        // Check the type of alignment in this column for this pair
        if (char_i != '-' && char_k != '-') {
          // --- Case 1: Residue vs Residue ---
          // This is a standard match/mismatch.
          // Use the provided scoring function pointer for substitution
          total_score += fn(char_i, char_k);
          in_gap_for_this_pair = false;  // The gap (if any) has ended.

        } else if (char_i !=
                   char_k) {  // This condition is true only for Residue vs Gap
          // --- Case 2: Residue vs Gap ---
          if (in_gap_for_this_pair) {
            // We are already in a gap, so this is an extension.
            total_score += iGapExtend;
          } else {
            // This is the first column of a new gap for this pair.
            // Apply both the open and the first extend penalty.
            total_score += iGapOpen + iGapExtend;
          }
          in_gap_for_this_pair = true;  // We are now in a gap state.

        } else {
          // --- Case 3: Gap vs Gap ---
          // No score is added or subtracted.
          // The 'in_gap' state simply continues.
          in_gap_for_this_pair = true;
        }
      }
    }
  }
  return total_score;
}

/**
 * @brief Normalizes the lengths of all sequences in a multiple sequence alignment
 *        by padding shorter sequences with trailing gap characters.
 *
 * This function ensures that all sequences in the input vector @p msa have the
 * same length by appending @c '-' characters to any sequence that is shorter
 * than the longest sequence. The longest sequence length is determined by
 * scanning through all sequences and finding the maximum length. Sequences
 * that are already of maximum length are left unchanged.
 *
 * @param[in,out] msa Vector of aligned sequences to normalize. All sequences
 *                     are modified in place to have the same length after
 *                     this function returns. If @p msa is empty, no action is taken.
 */
void normalize_msa_lengths(std::vector<std::string>& msa) {
  if (msa.empty()) return;

  std::size_t max_len = 0;
  for (const auto& row : msa) {
    max_len = std::max(max_len, row.size());
  }

  for (auto& row : msa) {
    if (row.size() < max_len) {
      row.append(max_len - row.size(), '-');
    }
  }
}

/**
 * @brief Propagates newly introduced gap columns from a aligned representative
 *        sequence into its entire profile.
 *
 * Compares @p oldc (the pre-alignment representative) against @p newc (the
 * post-alignment representative) character by character. Wherever @p newc
 * introduces a @c '-' that does not exist in @p oldc, a corresponding column
 * of @c '-' characters is inserted at the same position in every string in
 * @p seqs. This preserves the structural integrity of the profile when
 * merging two sub-alignments.
 *
 * @param oldc Pre-alignment (ungapped) representative consensus string.
 * @param newc Post-alignment (gapped) representative consensus string.
 * @param[in,out] seqs Profile strings to update. All strings are extended
 *                     in place with the same gap insertions applied to @p newc.
 */
void projectGaps(const std::string& oldc, const std::string& newc,
                 std::vector<std::string>& seqs) {
  if (seqs.empty()) return;

  if (newc.empty()) {
    for (auto& s : seqs) {
      s.clear();
    }
    return;
  }

  std::vector<bool> insert_gap(newc.size(), false);
  std::size_t       old_idx = 0;

  for (std::size_t new_idx = 0; new_idx < newc.size(); ++new_idx) {
    if (newc[new_idx] == '-') {
      insert_gap[new_idx] = true;
    } else {
      if (old_idx < oldc.size()) {
        ++old_idx;
      }
    }
  }

  for (auto& s : seqs) {
    std::string result;
    result.reserve(newc.size());

    std::size_t src = 0;

    for (std::size_t i = 0; i < newc.size(); ++i) {
      if (insert_gap[i]) {
        result.push_back('-');
      } else {
        if (src < s.size()) {
          result.push_back(s[src]);
          ++src;
        } else {
          result.push_back('-');
        }
      }
    }

    s = std::move(result);
  }
}

static std::string ungap(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) if (c != '-') out.push_back(c);
  return out;
}

std::vector<std::string> restore_full_length_msa(
    const std::vector<std::string>& originals,
    const std::vector<std::string>& aligned) {

  // This will store the final MSA sorted in the original input order
  std::vector<std::string> out(originals.size());
  std::vector<bool> used_aligned(aligned.size(), false);

  for (size_t i = 0; i < originals.size(); ++i) {
    bool found = false;

    // Find which aligned sequence corresponds to this original sequence
    for (size_t j = 0; j < aligned.size(); ++j) {
      if (used_aligned[j]) continue;

      std::string core = ungap(aligned[j]);

      // Check if this aligned core belongs to the original sequence i
      size_t pos = originals[i].find(core);
      if (pos != std::string::npos) {
        // Re-attach unaligned prefixes and suffixes
        std::string prefix = originals[i].substr(0, pos);
        std::string suffix = originals[i].substr(pos + core.size());
        out[i] = prefix + aligned[j] + suffix;

        used_aligned[j] = true;
        found = true;
        break;
      }
    }

    if (!found) {
      throw std::runtime_error("Aligned core not found in original sequence for row " + std::to_string(i));
    }
  }

  normalize_msa_lengths(out);
  return out;
}

/**
 * @brief Builds a pairwise distance matrix from global alignment identity
 * scores.
 *
 * For each unique pair @f$(i, j)@f$, runs @c computeGlobalAlignment and
 * computes the fractional identity as:
 * @f[
 *   \text{identity} = \frac{\text{matches}}{L},\quad
 *   D_{ij} = 1 - \text{identity}
 * @f]
 * where @f$ L @f$ is the alignment length and matches are positions where
 * both aligned characters are equal and non-gap. The matrix is symmetric.
 *
 * All @f$ N(N-1)/2 @f$ pairs are computed in parallel using OpenMP.
 *
 * @param seqs Vector of ungapped input sequences.
 * @param mode Scoring mode for pairwise alignment.
 * @param fn   Substitution scoring function pointer.
 * @param p    Affine gap penalty parameters.
 * @return     Symmetric @f$ N \times N @f$ distance matrix @c D where
 *             @c D[i][i] = 0 and @c D[i][j] ? [0, 1].
 */
std::vector<std::vector<double>> computeDistanceMatrix(
    const std::vector<std::string>& seqs, ScoreMode mode, ScoreFn fn,
    const AlignParams& p) {
  size_t                           n = seqs.size();
  std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));

#pragma omp parallel for schedule(dynamic) default(none) \
    shared(seqs, D, mode, fn, p, n)
  for (int i = 0; i < static_cast<int>(n); ++i) {
    for (int j = i + 1; j < static_cast<int>(n); ++j) {
      std::string ax, ay;
      computeGlobalAlignment(seqs[i], seqs[j], mode, fn, ax, ay, p);
      const int L = checked_size_to_int(ax.size(), "ax.size()");
      int       matches = 0;
      int       compared = 0;
      for (int k = 0; k < L; ++k) {
        char ca = ax[k];
        char cb = ay[k];
        if (ca == '-' && cb == '-') continue;  // skip double-gap columns
        if (ca != '-' && cb != '-' && ca == cb) {
          ++matches;
        }
        ++compared;
      }
      double identity = 0.0;
      if (compared > 0) identity = static_cast<double>(matches) / compared;
      double dist = 1.0 - identity;
      D[i][j] = D[j][i] = dist;
    }
  }
  return D;
}

/**
 * @brief Stores all optimal pairwise residue-pair positions for an input set.
 *
 * For every ordered pair (i, k) where i < k, @c lib[i][k] contains the set
 * of (ungapped_pos_i, ungapped_pos_k) integer pairs that co-occur in the
 * optimal global alignment of sequences i and k. This library is the
 * foundation of the COFFEE Log-Expectation scoring used by
 * calculate_le_score() and build_profile_le().
 *
 * @see buildPairLib(), calculate_le_score(), build_profile_le()
 */
struct PairLib {
  int N{};  ///< Number of input sequences.
  /// @c lib[i][k] holds residue-pair positions from the optimal
  /// pairwise alignment of sequences i and k (i < k only).
  std::vector<std::vector<std::set<std::pair<int, int>>>> lib;
};

/**
 * @brief Constructs the pairwise alignment library for a set of sequences.
 *
 * For every unique pair (i, k) with i < k, runs computeGlobalAlignment() and
 * records every column where both sequences contribute a non-gap residue as a
 * (ungapped_pos_i, ungapped_pos_k) pair in @c PairLib::lib[i][k].
 *
 * The outer loop is parallelised with @c schedule(dynamic) OpenMP. Individual
 * insertions into the shared @c lib are protected by @c #pragma omp critical.
 *
 * Time complexity: O(N� � L�) for N sequences of average length L.
 *
 * @param seqs  Raw (ungapped) input sequences.
 * @param mode  Scoring mode ? @c MODE_DNA or @c MODE_PROTEIN.
 * @param fn    Raw scoring function pointer compatible with @p mode.
 * @param p     Alignment parameters (gap_open, gap_extend).
 *
 * @return A fully populated @c PairLib for @p seqs.
 *
 * @see PairLib, computeGlobalAlignment(), build_profile_le()
 */
PairLib buildPairLib(const std::vector<std::string>& seqs, ScoreMode mode,
                     ScoreFn fn, const AlignParams& p) {
  const int n = checked_size_to_int(seqs.size(), "seqs.size()");
  PairLib   pl;
  pl.N = n;
  pl.lib.assign(n, std::vector<std::set<std::pair<int, int>>>(n));

#pragma omp parallel for schedule(dynamic) default(none) \
    shared(seqs, pl, mode, fn, p, n)
  for (int i = 0; i < n; ++i) {
    for (int k = i + 1; k < n; ++k) {
      std::string ax, ay;
      computeGlobalAlignment(seqs[i], seqs[k], mode, fn, ax, ay, p);

      int pi = 0, pk = 0;
      for (size_t j = 0; j < ax.size(); ++j) {
        bool gi = (ax[j] == '-');
        bool gk = (ay[j] == '-');
        if (!gi && !gk) {
#pragma omp critical
          pl.lib[i][k].insert({pi, pk});
        }
        if (!gi) ++pi;
        if (!gk) ++pk;
      }
    }
  }
  return pl;
}

/**
 * @brief Computes a banded global alignment seeded from an FFT-derived offset.
 *
 * Runs the affine-gap Needleman-Wunsch recurrence but restricts filling to
 * cells within @p bandwidth diagonals of the seed diagonal:
 * @f[
 *   |j - i - \text{seed\_offset}| \leq \text{bandwidth}
 * @f]
 * Cells outside the band remain at @c INT_MIN/2 and are skipped. This reduces
 * time complexity from @f$ O(mn) @f$ to @f$ O((m+n) \cdot \text{bandwidth}) @f$
 * for sequences whose true alignment lies near the seed diagonal.
 *
 * If the band fails to reach the endpoint @f$ S[m][n] @f$ (i.e., the true
 * alignment lies outside the band), the function transparently falls back to
 * a full @c computeGlobalAlignment call so the result is always valid.
 *
 * @param x           First sequence (ungapped, uppercase).
 * @param y           Second sequence (ungapped, uppercase).
 * @param mode        Scoring mode, forwarded to @p fn.
 * @param fn          Substitution scoring function pointer.
 * @param[out] aligned_x Aligned version of @p x with gap characters inserted.
 * @param[out] aligned_y Aligned version of @p y with gap characters inserted.
 * @param p           Affine gap penalty parameters.
 * @param seed_offset Diagonal offset @f$(j - i)@f$ around which to centre
 *                    the band. Typically the peak index returned by
 *                    @c fftCrossCorrelation.
 * @param bandwidth   Half-width of the band in diagonal units. Default is 50.
 *                    Increase for more divergent sequences.
 * @return            Alignment score of the best path found within the band,
 *                    or the full NW score if the band fallback was triggered.
 */
int computeBandedAlignment(const std::string& x, const std::string& y,
                           ScoreMode mode, ScoreFn fn, std::string& aligned_x,
                           std::string& aligned_y, const AlignParams& p,
                           int seed_offset, int bandwidth = 50) {
  const int m       = checked_size_to_int(x.size(), "x.size()");
  const int n       = checked_size_to_int(y.size(), "y.size()");
  const int iOpen   = static_cast<int>(std::round(p.gap_open));
  const int iExtend = static_cast<int>(std::round(p.gap_extend));

  // For long sequences, cap bandwidth and only allocate band rows
  // Each row only needs cells in [max(1, i+seed_offset-bw), min(n,
  // i+seed_offset+bw)]
  const int bw =
      std::min(bandwidth, std::max(m, n));  // never wider than sequence

  // Flat allocation: only store 2 rows at a time for S, E, F
  // But we need full traceback ? if sequences are very long, use full NW
  // directly
  constexpr int LONG_SEQ_THRESHOLD = 2000;
  if (m > LONG_SEQ_THRESHOLD || n > LONG_SEQ_THRESHOLD) {
    // For long sequences FFT seeding isn't reliable anyway ? use full AVX2 NW
    return computeGlobalAlignment(x, y, mode, fn, aligned_x, aligned_y, p);
  }

  seed_offset = std::max(-(m - 1), std::min(n - 1, seed_offset));

  std::vector<std::vector<int>>  S(m + 1, std::vector<int>(n + 1, INT_MIN / 2));
  std::vector<std::vector<int>>  E(m + 1, std::vector<int>(n + 1, INT_MIN / 2));
  std::vector<std::vector<int>>  F(m + 1, std::vector<int>(n + 1, INT_MIN / 2));
  std::vector<std::vector<char>> tr(m + 1, std::vector<char>(n + 1, '?'));

  S[0][0]  = 0;
  tr[0][0] = 'M';
  for (int j = 1; j <= n; ++j) {
    int diag = -j;
    if (diag >= seed_offset - bw && diag <= seed_offset + bw) {
      E[0][j]  = (j == 1) ? iOpen + iExtend : E[0][j - 1] + iExtend;
      S[0][j]  = E[0][j];
      tr[0][j] = (j == 1 ? 'E' : 'e');
    }
  }
  for (int i = 1; i <= m; ++i) {
    int diag = i;
    if (diag >= seed_offset - bw && diag <= seed_offset + bw) {
      F[i][0]  = (i == 1) ? iOpen + iExtend : F[i - 1][0] + iExtend;
      S[i][0]  = F[i][0];
      tr[i][0] = (i == 1 ? 'F' : 'f');
    }
  }

  for (int i = 1; i <= m; ++i) {
    for (int j = 1; j <= n; ++j) {
      int diag = j - i - seed_offset;
      if (std::abs(diag) > bw) continue;

      int prev =
          (S[i - 1][j - 1] > INT_MIN / 2) ? S[i - 1][j - 1] : INT_MIN / 2;
      int match_score =
          (prev > INT_MIN / 2) ? prev + fn(x[i - 1], y[j - 1]) : INT_MIN / 2;

      E[i][j] = std::max(
          (S[i][j - 1] > INT_MIN / 2) ? S[i][j - 1] + iOpen : INT_MIN / 2,
          (E[i][j - 1] > INT_MIN / 2) ? E[i][j - 1] + iExtend : INT_MIN / 2);
      F[i][j] = std::max(
          (S[i - 1][j] > INT_MIN / 2) ? S[i - 1][j] + iOpen : INT_MIN / 2,
          (F[i - 1][j] > INT_MIN / 2) ? F[i - 1][j] + iExtend : INT_MIN / 2);
      S[i][j] = std::max({match_score, E[i][j], F[i][j]});

      if (S[i][j] == match_score)
        tr[i][j] = 'M';
      else if (S[i][j] == E[i][j])
        tr[i][j] = 'E';
      else
        tr[i][j] = 'F';
    }
  }

  if (S[m][n] <= INT_MIN / 2) {
    return computeGlobalAlignment(x, y, mode, fn, aligned_x, aligned_y, p);
  }

  aligned_x.clear();
  aligned_y.clear();
  int i = m, j = n;
  while (i > 0 || j > 0) {
    char t = tr[i][j];
    if (t == 'M') {
      aligned_x += x[i - 1];
      aligned_y += y[j - 1];
      --i;
      --j;
    } else if (t == 'F' || t == 'f') {
      aligned_x += x[i - 1];
      aligned_y += '-';
      --i;
    } else if (t == 'E' || t == 'e') {
      aligned_x += '-';
      aligned_y += y[j - 1];
      --j;
    } else {
      if (i > 0) {
        aligned_x += x[i - 1];
        aligned_y += '-';
        --i;
      } else {
        aligned_x += '-';
        aligned_y += y[j - 1];
        --j;
      }
    }
  }
  std::reverse(aligned_x.begin(), aligned_x.end());
  std::reverse(aligned_y.begin(), aligned_y.end());
  return S[m][n];
}

/**
 * @brief Builds a progressive MSA profile using FFT-seeded banded alignment.
 *
 * Recursively traverses the UPGMA guide tree. At each internal node, it
 * generates a consensus sequence for each child profile, uses FFT
 * cross-correlation (fftCrossCorrelation()) to estimate the best diagonal
 * offset between the two consensus sequences, and then performs a banded
 * Needleman-Wunsch alignment (computeBandedAlignment()) centred on that
 * offset. The resulting gap pattern is projected back into all sequences of
 * each child profile via projectGaps(), and the two profiles are merged.
 *
 * Compared to the plain progressive NW aligner (build_profile()), this method
 * is faster on long, similar sequences because the band restricts the DP to
 * O(L * bandwidth) cells instead of O(L^2). For short or highly diverged
 * sequences the banded aligner falls back automatically to full NW.
 *
 * @param n     Pointer to the current node in the binary guide tree.
 *              Leaf nodes must have their @c profile field pre-seeded with
 *              the single raw sequence (done by @c make_tree in main()).
 *              Pass @c nullptr to get an empty result.
 * @param mode  Scoring mode ? @c MODE_DNA uses the EDNAFULL matrix,
 *              @c MODE_PROTEIN uses BLOSUM62.
 * @param fn    Raw scoring function pointer compatible with @p mode
 *              (e.g. @c edna_score or @c blosum62_score).
 * @param p     Alignment parameters (gap_open, gap_extend).
 *
 * @return A @c std::vector<std::string> where every string is the same
 *         length, representing the column-aligned sequences rooted at @p n.
 *         Returns an empty vector if @p n is @c nullptr.
 *
 * @note This function is not thread-safe when called concurrently on nodes
 *       that share child pointers. In main(), three independent trees are
 *       parsed so each @c #pragma omp section owns its own tree.
 *
 * @see fftCrossCorrelation(), computeBandedAlignment(), projectGaps(),
 *      build_profile(), build_profile_le()
 */
std::vector<std::string> build_profile_fft(Node* n, const std::vector<std::string>& seqs, ScoreMode mode, ScoreFn fn,
                                           const AlignParams& p) {
  if (!n) return {};
  if (n->leaf) {
    int idx = n->seq_index;
    if (idx < 0 || idx >= checked_size_to_int(seqs.size(), "seqs.size()")) {
      std::ostringstream oss;
      oss << "Invalid leaf sequence index " << idx;
      throw std::runtime_error(oss.str());
    }
    return { seqs[idx] };
  }

  auto A = build_profile_fft(n->left, seqs, mode, fn, p);
  auto B = build_profile_fft(n->right, seqs, mode, fn, p);

  std::string consA = generate_consensus(A);
  std::string consB = generate_consensus(B);

  auto [offset, corr] = fftCrossCorrelation(consA, consB, mode);

  std::string aligned_consA, aligned_consB;
  int bw = std::max(50, checked_size_to_int(std::max(consA.size(), consB.size()) / 4,
                                            "consensus bandwidth"));
  computeBandedAlignment(consA, consB, mode, fn, aligned_consA, aligned_consB,
                         p, offset, bw);

  projectGaps(consA, aligned_consA, A);
  projectGaps(consB, aligned_consB, B);

  std::vector<std::string> merged = A;
  merged.insert(merged.end(), B.begin(), B.end());
  return merged;
}

/**
 * @brief Merges two MSA profiles using NW on their consensus sequences,
 *        guided by a pairwise alignment library (COFFEE LE approach).
 *
 * Generates a consensus for each profile, aligns the two consensus sequences
 * with computeGlobalAlignment(), projects the resulting gaps back into all
 * sequences of each group via projectGaps(), and concatenates the groups into
 * a single merged profile.
 *
 * @param A      Profile for the left  child group ? modified in-place to
 *               insert gap columns; pass a local copy if the original must
 *               be preserved.
 * @param B      Profile for the right child group ? same in-place semantics.
 * @param pl     Pairwise alignment library built by buildPairLib().
 *               Reserved for future LE-aware column scoring; currently the
 *               alignment step uses standard NW.
 * @param idx_A  Original sequence indices (into the full input @c seqs
 *               vector) for every sequence in profile @p A.
 * @param idx_B  Original sequence indices for every sequence in profile @p B.
 * @param mode   Scoring mode ? @c MODE_DNA or @c MODE_PROTEIN.
 * @param fn     Raw scoring function pointer compatible with @p mode.
 * @param p      Alignment parameters (gap_open, gap_extend).
 *
 * @return A merged @c std::vector<std::string> containing all sequences from
 *         @p A followed by all sequences from @p B, column-aligned.
 *
 * @see buildPairLib(), build_profile_le(), projectGaps(),
 *      computeGlobalAlignment()
 */
std::vector<std::string> merge_profiles_le(
    std::vector<std::string>& A, std::vector<std::string>& B, const PairLib& pl,
    const std::vector<int>& idx_A, const std::vector<int>& idx_B,
    ScoreMode mode, ScoreFn fn, const AlignParams& p) {
  std::string consA = generate_consensus(A);
  std::string consB = generate_consensus(B);

  // Use standard NW on consensus to get the column structure
  std::string aligned_consA, aligned_consB;
  computeGlobalAlignment(consA, consB, mode, fn, aligned_consA, aligned_consB,
                         p);

  projectGaps(consA, aligned_consA, A);
  projectGaps(consB, aligned_consB, B);

  std::vector<std::string> merged = A;
  merged.insert(merged.end(), B.begin(), B.end());
  return merged;
}

/**
 * @brief Result node for the LE-guided profile builder.
 *
 * Pairs an aligned profile (a set of equal-length strings) with the original
 * sequence indices that contributed to it. The index bookkeeping is required
 * by merge_profiles_le() to look up residue pairs in the PairLib.
 */
struct LENode {
  std::vector<std::string> profile;  ///< Column-aligned sequences at this node.
  std::vector<int> seq_indices;  ///< Indices into the original @c seqs vector.
};

/**
 * @brief Builds a progressive MSA profile using COFFEE Log-Expectation (LE)
 *        guided merging.
 *
 * Recursively traverses the UPGMA guide tree. At each internal node it calls
 * merge_profiles_le() to merge the left and right child profiles, passing
 * along the original sequence index lists so the pairwise library can be
 * consulted during column scoring.
 *
 * This method is designed to reward column pairings that are consistent with
 * the pre-computed pairwise alignments stored in @p pl, following the
 * T-Coffee / COFFEE objective.
 *
 * @param n        Pointer to the current node in the binary guide tree.
 *                 Leaf nodes are initialised directly from @p raw_seqs using
 *                 @c n->seq_index. Pass @c nullptr to get an empty result.
 * @param raw_seqs The original, ungapped input sequences indexed by their
 *                 position in the input file list.
 * @param pl       Pairwise alignment library produced by buildPairLib().
 *                 Contains every optimal pairwise residue-pair for all
 *                 (i, k) combinations.
 * @param mode     Scoring mode ? @c MODE_DNA or @c MODE_PROTEIN.
 * @param fn       Raw scoring function pointer compatible with @p mode.
 * @param p        Alignment parameters (gap_open, gap_extend).
 *
 * @return An @c LENode containing the merged, column-aligned profile and the
 *         full list of original sequence indices under this node.
 *         Returns a default-constructed (empty) @c LENode if @p n is
 *         @c nullptr.
 *
 * @see merge_profiles_le(), buildPairLib(), LENode, build_profile(),
 *      build_profile_fft()
 */
LENode build_profile_le(Node* n, const std::vector<std::string>& raw_seqs,
                        const PairLib& pl, ScoreMode mode, ScoreFn fn,
                        const AlignParams& p) {
  if (!n) return {};
  if (n->leaf) {
    int idx = n->seq_index;
    if (idx < 0 || idx >= checked_size_to_int(raw_seqs.size(), "raw_seqs.size()")) {
      std::ostringstream oss;
      oss << "Invalid leaf sequence index " << idx;
      throw std::runtime_error(oss.str());
    }
    return {{raw_seqs[idx]}, {idx}};
  }

  auto L = build_profile_le(n->left, raw_seqs, pl, mode, fn, p);
  auto R = build_profile_le(n->right, raw_seqs, pl, mode, fn, p);

  auto merged_seqs = merge_profiles_le(L.profile, R.profile, pl, L.seq_indices,
                                       R.seq_indices, mode, fn, p);

  std::vector<int> merged_idx = L.seq_indices;
  merged_idx.insert(merged_idx.end(), R.seq_indices.begin(),
                    R.seq_indices.end());

  return {merged_seqs, merged_idx};
}

/**
 * @brief Saves the pairwise identity matrix to a formatted text file.
 * The output is aligned in columns for readability. The identity is calculated
 * from the distance matrix (identity % = (1.0 - distance) * 100).
 *
 * @param D The distance matrix.
 * @param hdrs A vector of the sequence headers.
 * @param outdir The directory where the output file will be saved.
 */
void saveIdentityMatrix(const std::vector<std::vector<double>>& D,
                        const std::vector<std::string>&         hdrs,
                        const std::string&                      outdir) {
  if (D.empty() || D.size() != hdrs.size()) {
    std::cerr << "Warning: Cannot save identity matrix due to size mismatch or "
                 "empty data."
              << std::endl;
    return;
  }

  const std::string output_filepath = outdir + "/identity_matrix.txt";
  std::ofstream     matrix_file(output_filepath);

  if (!matrix_file.is_open()) {
    std::cerr << "Error: Could not open " << output_filepath << " for writing."
              << std::endl;
    return;
  }

  size_t n = hdrs.size();

  // Find the length of the longest header to format the output neatly
  size_t max_hdr_len = 0;
  for (const auto& h : hdrs) {
    if (h.length() > max_hdr_len) {
      max_hdr_len = h.length();
    }
  }
  // Add some padding
  max_hdr_len += 4;

  // Set formatting for floating point numbers
  matrix_file << std::fixed << std::setprecision(2);

  for (size_t i = 0; i < n; ++i) {
    // Print the row number (e.g., "1:") and the left-aligned header
    matrix_file << std::setw(5) << std::right << std::to_string(i + 1) + ":"
                << " " << std::setw(max_hdr_len) << std::left << hdrs[i];

    // Print the identity values for the entire row
    for (size_t j = 0; j < n; ++j) {
      double identity_percent = 0.0;
      if (i == j) {
        identity_percent = 100.0;
      } else {
        // The distance matrix D is symmetric
        identity_percent = (1.0 - D[i][j]) * 100.0;
      }
      matrix_file << std::setw(8) << std::right << identity_percent;
    }
    matrix_file << "\n";
  }

  matrix_file.close();
  // std::cout << "\nPairwise identity matrix saved to: " << output_filepath <<
  // std::endl;
}

/**
 * @brief Generate a unique job identifier of the form
 *        msalign-IYYYYMMDD-HHMMSS-RAND4-RAND8-p1m
 *
 * The timestamp is taken from the current system clock and formatted
 * according to UTC. Random numeric components are generated using
 * a mt19937 PRNG seeded with std::random_device.
 *
 * @return A string conforming to the required job ID specification.
 */
static std::string generate_job_id() {
  auto now      = std::chrono::system_clock::now();
  auto now_time = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &now_time);
#else
  gmtime_r(&now_time, &tm);
#endif
  std::ostringstream oss;
  oss << "msalign-I";
  oss << std::put_time(&tm, "%Y%m%d-%H%M%S");
  std::random_device rd;
  std::mt19937       gen(rd());
  std::uniform_int_distribution<int> dist4(0, 9999);
  std::uniform_int_distribution<int> dist8(0, 99999999);
  int r4 = dist4(gen);
  int r8 = dist8(gen);
  oss << "-" << std::setw(4) << std::setfill('0') << r4;
  oss << "-" << std::setw(8) << std::setfill('0') << r8;
  oss << "-p1m";
  return oss.str();
}

/**
 * @brief Compute a percent identity matrix from a multiple sequence alignment.
 *
 * The identity between two sequences i and j is defined as the ratio of
 * identical residues to the number of positions compared, where positions
 * where both sequences have a gap are excluded. If no positions are
 * comparable, the identity is defined as 0.0. The diagonal entries are 1.0.
 *
 * @param msa Aligned sequences of equal length.
 * @return A symmetric matrix of size N×N with values in [0,1].
 */
static std::vector<std::vector<double>>
compute_percent_identity_from_msa(const std::vector<std::string>& msa) {
  const std::size_t n = msa.size();
  std::vector<std::vector<double>> pim(n, std::vector<double>(n, 1.0));
  if (n == 0) return pim;
  const std::size_t L = msa[0].size();
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      std::size_t matches      = 0;
      std::size_t compared_pos = 0;
      for (std::size_t col = 0; col < L; ++col) {
        char a = msa[i][col];
        char b = msa[j][col];
        if (a == '-' && b == '-') continue;
        if (a == b && a != '-') ++matches;
        ++compared_pos;
      }
      double identity = 0.0;
      if (compared_pos > 0) identity = static_cast<double>(matches) / compared_pos;
      pim[i][j] = pim[j][i] = identity;
    }
  }
  return pim;
}

/**
 * @brief Write a percent identity matrix to a file using a format similar to
 * Clustal Omega.
 *
 * The matrix is prefaced by comment lines and then a table containing
 * indices, names and identity percentages. Values on the diagonal are
 * shown as 100.00. Off-diagonal values are scaled by 100.
 *
 * @param pim Matrix of identity fractions.
 * @param names Sequence identifiers corresponding to the rows/columns.
 * @param outpath Path to write the identity matrix file.
 */
static void write_percent_identity_matrix(const std::vector<std::vector<double>>& pim,
                                          const std::vector<std::string>& names,
                                          const std::string& outpath) {
  std::ofstream out(outpath);
  if (!out) {
    std::cerr << "Error writing percent identity matrix to " << outpath << std::endl;
    return;
  }
  out << "#\n#\n#  Percent Identity Matrix\n#\n#\n\n";
  const std::size_t n = names.size();
  std::size_t max_len = 0;
  for (const auto& nm : names) {
    if (nm.size() > max_len) max_len = nm.size();
  }
  for (std::size_t i = 0; i < n; ++i) {
    out << std::setw(6) << std::right << (i + 1) << ": "
        << std::setw(static_cast<int>(max_len) + 2) << std::left << names[i];
    for (std::size_t j = 0; j < n; ++j) {
      double val = (i == j ? 1.0 : pim[i][j]);
      out << std::setw(8) << std::right << std::fixed << std::setprecision(2)
          << (val * 100.0);
    }
    out << "\n";
  }
  out.close();
}

/**
 * @brief Write a multiple sequence alignment in CLUSTAL format with residue numbering.
 *
 * Sequence names are left-aligned to a uniform width. The alignment is printed
 * in blocks of a fixed column width. At the end of each sequence line, the
 * number of non-gap residues up to and including the last character of that
 * block is displayed. A consensus line is generated showing '*' for fully
 * conserved positions across all sequences and ' ' otherwise.
 *
 * @param msa Aligned sequences of equal length.
 * @param names Sequence identifiers corresponding to the rows.
 * @param outpath Path to write the CLUSTAL formatted alignment.
 * @param block_size Width of each alignment block (default 60).
 */
static void write_clustal_alignment(const std::vector<std::string>& msa,
                                    const std::vector<std::string>& names,
                                    const std::string& outpath,
                                    std::size_t block_size = 60) {
  std::ofstream out(outpath);
  if (!out) {
    std::cerr << "Error writing CLUSTAL alignment to " << outpath << std::endl;
    return;
  }
  out << "MSAlign multiple sequence alignment with numbers\n\n";
  const std::size_t n = msa.size();
  if (n == 0) {
    out.close();
    return;
  }
  const std::size_t L = msa[0].size();
  std::size_t name_width = 0;
  for (const auto& nm : names) {
    if (nm.size() > name_width) name_width = nm.size();
  }
  name_width += 2;
  std::vector<std::size_t> residue_counts(n, 0);
  for (std::size_t start = 0; start < L; start += block_size) {
    std::size_t end = std::min(start + block_size, L);
    for (std::size_t i = 0; i < n; ++i) {
      out << std::setw(static_cast<int>(name_width)) << std::left << names[i];
      for (std::size_t col = start; col < end; ++col) {
        out << msa[i][col];
      }
      std::size_t count = 0;
      for (std::size_t col = start; col < end; ++col) {
        if (msa[i][col] != '-') ++count;
      }
      residue_counts[i] += count;
      out << ' ' << std::setw(6) << residue_counts[i] << '\n';
    }
    out << std::setw(static_cast<int>(name_width)) << std::left << "";
    for (std::size_t col = start; col < end; ++col) {
      char c = msa[0][col];
      bool conserved = (c != '-');
      for (std::size_t i = 1; i < n && conserved; ++i) {
        if (msa[i][col] != c) conserved = false;
      }
      out << (conserved ? '*' : ' ');
    }
    out << "\n\n";
  }
  out.close();
}

/**
 * @brief Write a multiple sequence alignment in FASTA format.
 *
 * @param msa Aligned sequences of equal length.
 * @param names Sequence identifiers corresponding to the rows.
 * @param outpath Path to write the FASTA formatted alignment.
 */
static void write_fasta_alignment(const std::vector<std::string>& msa,
                                  const std::vector<std::string>& names,
                                  const std::string& outpath) {
  std::ofstream out(outpath);
  if (!out) {
    std::cerr << "Error writing FASTA alignment to " << outpath << std::endl;
    return;
  }
  const std::size_t wrap_width = 80;
  for (std::size_t i = 0; i < msa.size(); ++i) {
    out << '>' << names[i] << '\n';
    const std::string& seq = msa[i];
    for (std::size_t pos = 0; pos < seq.size(); pos += wrap_width) {
      out << seq.substr(pos, std::min(wrap_width, seq.size() - pos)) << '\n';
    }
  }
  out.close();
}

/**
 * @brief Write metadata about the run to a JSON file.
 */
static void write_metadata_json(const std::string& outpath, const std::string& job_id,
                                const std::string& tool_name, const std::string& tool_version,
                                const std::string& start_time, const std::string& end_time,
                                double duration_seconds, const std::string& cmdline,
                                const std::string& input_file, const std::string& sequence_type,
                                const std::vector<std::string>& names, int num_threads,
                                bool avx2_enabled, bool openmp_enabled,
                                const std::string& distance_model,
                                const std::string& guide_tree_alg,
                                const std::string& progressive_alg,
                                const std::string& refinement_method,
                                const std::string& best_method,
                                const std::map<std::string,std::string>& outputs) {
  std::ofstream out(outpath);
  if (!out) {
    std::cerr << "Error writing metadata JSON to " << outpath << std::endl;
    return;
  }
  out << "{\n";
  out << "  \"job_id\": \"" << job_id << "\",\n";
  out << "  \"tool_name\": \"" << tool_name << "\",\n";
  out << "  \"tool_version\": \"" << tool_version << "\",\n";
  out << "  \"run_started_at\": \"" << start_time << "\",\n";
  out << "  \"run_finished_at\": \"" << end_time << "\",\n";
  out << std::fixed << std::setprecision(3);
  out << "  \"duration_seconds\": " << duration_seconds << ",\n";
  out << "  \"command_line\": \"" << cmdline << "\",\n";
  out << "  \"input_file\": \"" << input_file << "\",\n";
  out << "  \"sequence_type\": \"" << sequence_type << "\",\n";
  out << "  \"num_sequences\": " << names.size() << ",\n";
  out << "  \"sequence_ids\": [";
  for (std::size_t i = 0; i < names.size(); ++i) {
    out << "\"" << names[i] << "\"";
    if (i + 1 < names.size()) out << ", ";
  }
  out << "],\n";
  out << "  \"num_threads\": " << num_threads << ",\n";
  out << "  \"simd\": {\n";
  out << "    \"avx2_enabled\": " << (avx2_enabled ? "true" : "false") << ",\n";
  out << "    \"runtime_detected\": " << (avx2_enabled ? "true" : "false") << "\n";
  out << "  },\n";
  out << "  \"openmp_enabled\": " << (openmp_enabled ? "true" : "false") << ",\n";
  out << "  \"algorithm\": {\n";
  out << "    \"distance_model\": \"" << distance_model << "\",\n";
  out << "    \"guide_tree\": \"" << guide_tree_alg << "\",\n";
  out << "    \"progressive_alignment\": \"" << progressive_alg << "\",\n";
  out << "    \"profile_profile_alignment\": \"" << best_method << "\",\n";
  out << "    \"refinement\": \"" << refinement_method << "\"\n";
  out << "  },\n";
  out << "  \"outputs\": {\n";
  std::size_t counter = 0;
  for (const auto& kv : outputs) {
    out << "    \"" << kv.first << "\": \"" << kv.second << "\"";
    if (++counter < outputs.size()) out << ",\n";
    else out << "\n";
  }
  out << "  },\n";
  out << "  \"status\": \"success\",\n";
  out << "  \"warnings\": [],\n";
  out << "  \"errors\": []\n";
  out << "}\n";
  out.close();
}

/**
 * @brief Write a submission XML file describing the run.
 */
static void write_submission_xml(const std::string& outpath, const std::string& job_id,
                                 const std::string& cmdline,
                                 const std::vector<std::string>& names,
                                 const std::map<std::string,std::string>& outputs,
                                 const std::string& sequence_type) {
  auto xml_escape = [](const std::string& s) -> std::string {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
      switch (c) {
        case '&': out += "&amp;"; break;
        case '<': out += "&lt;"; break;
        case '>': out += "&gt;"; break;
        case '"': out += "&quot;"; break;
        case '\'': out += "&apos;"; break;
        default: out.push_back(c); break;
      }
    }
    return out;
  };
  std::ofstream out(outpath);
  if (!out) {
    std::cerr << "Error writing submission XML to " << outpath << std::endl;
    return;
  }
  out << "<execution>\n";
  out << "  <commandLine>" << xml_escape(cmdline) << "</commandLine>\n";
  out << "  <userParameters class=\"linked-hash-map\">\n";
  out << "    <entry><string>program</string><string>msalign-compatible-tool</string></entry>\n";
  out << "    <entry><string>version</string><string>1.0</string></entry>\n";
  out << "    <entry><string>guidetreeout</string><boolean>true</boolean></entry>\n";
  out << "    <entry><string>dismatout</string><boolean>true</boolean></entry>\n";
  out << "    <entry><string>dealign</string><boolean>false</boolean></entry>\n";
  out << "    <entry><string>mbed</string><boolean>true</boolean></entry>\n";
  out << "    <entry><string>mbediteration</string><boolean>true</boolean></entry>\n";
  out << "    <entry><string>iterations</string><int>3</int></entry>\n";
  out << "    <entry><string>outfmt</string><string>clustal_num</string></entry>\n";
  out << "    <entry><string>order</string><string>aligned</string></entry>\n";
  out << "    <entry><string>stype</string><string>" << xml_escape(sequence_type) << "</string></entry>\n";
  out << "    <entry><string>sequence</string><string>" << xml_escape(outputs.at("sequence")) << "</string></entry>\n";
  out << "  </userParameters>\n";
  out << "  <outputLocation>file:" << xml_escape(outputs.at("zip")) << "</outputLocation>\n";
  out << "  <errorLocation>file:" << xml_escape(outputs.at("zip")) << "</errorLocation>\n";
  out << "  <toolId>msalign-compatible-tool</toolId>\n";
  out << "  <jobId>" << xml_escape(job_id) << "</jobId>\n";
  out << "  <jobTitle></jobTitle>\n";
  out << "  <emailNotification>false</emailNotification>\n";
  out << "  <executionContext>local</executionContext>\n";
  out << "</execution>\n";
  out.close();
}

/**
 * @brief Build a UPGMA guide tree via a min-heap in O(n^2 log n).
 *
 * This function performs agglomerative clustering under the UPGMA
 * algorithm. Each initial cluster is labelled by an entry from the
 * @p names vector. When two clusters merge, their labels are combined
 * into a parenthesis-delimited Newick subtree annotated with a branch
 * length equal to half the distance between them. The returned string is
 * the final Newick tree terminated by a semicolon. The input
 * @p names vector is passed by value so modifications do not affect
 * the caller. Labels should be unique; no validation is performed here.
 *
 * The distance update uses the classical UPGMA formula, computing a
 * weighted average of distances between the two merged clusters and all
 * remaining clusters based on the size of each cluster *before* the merge.
 * This avoids biasing the average by the newly combined cluster size.
 *
 * @param D Symmetric distance matrix.
 * @param names Cluster labels for the leaves. Each label will appear exactly
 *              once in the returned Newick string. Passing stable internal
 *              labels (e.g. "SEQ_0", "SEQ_1", ...) is recommended to
 *              avoid ambiguity.
 * @return Newick string representing the guide tree.
 */
std::string buildUPGMATree(std::vector<std::vector<double>> D,
                           std::vector<std::string>         names) {
  const int n = checked_size_to_int(names.size(), "names.size()");
  struct Cluster {
    std::string nwk;
    int         size{};
  };
  std::vector<Cluster> C(n);
  std::vector<bool>    alive(n, true);
  for (int i = 0; i < n; ++i) C[i] = {names[i], 1};

  // Min?heap of (distance, a, b)
  using Entry = std::tuple<double, int, int>;
  auto cmp    = [](Entry const& a, Entry const& b) {
    return std::get<0>(a) > std::get<0>(b);
  };
  std::priority_queue<Entry, std::vector<Entry>, decltype(cmp)> pq(cmp);

  // seed heap
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) pq.emplace(D[i][j], i, j);

  int remaining = n;
  while (remaining > 1) {
    auto [d, a, b] = pq.top();
    pq.pop();
    if (!alive[a] || !alive[b]) continue;  // stale entry

    // merge b into a
    double             half = d * 0.5;
    std::ostringstream nw;
    nw << "(" << C[a].nwk << ":" << half << "," << C[b].nwk << ":" << half
       << ")";
    C[a].nwk = nw.str();
    // Capture cluster sizes before update
    int size_a = C[a].size;
    int size_b = C[b].size;
    C[a].size  = size_a + size_b;
    alive[b] = false;
    --remaining;

    // update distances from a to all k
    for (int k = 0; k < n; ++k) {
      if (alive[k] && k != a) {
        // weighted average
        double dk = (D[a][k] * size_a + D[b][k] * size_b) /
                    static_cast<double>(size_a + size_b);
        D[a][k] = D[k][a] = dk;
        pq.emplace(dk, a, k);
      }
    }
  }

  // find the last alive cluster
  std::string tree = ";";
  for (int i = 0; i < n; ++i) {
    if (alive[i]) {
      tree = C[i].nwk + ";";
      break;
    }
  }
  return tree;
}

/**
 * @brief A single "worker" function that attempts to improve an MSA for a set
 * number of iterations. This function is designed to be called in parallel. It
 * starts with an initial MSA and returns the best one it could find in its
 * allotted iterations.
 *
 * @param initial_msa The starting MSA for this worker.
 * @param iterations The number of random splits/realigns to attempt.
 * @param mode The sequence mode (DNA or Protein).
 * @param fn The scoring function to use.
 * @param g A reference to a seeded random number generator.
 * @return The best MSA found by this worker.
 */
std::vector<std::string> refine_msa_worker(std::vector<std::string> initial_msa,
                                           int iterations, ScoreMode mode,
                                           ScoreFn fn, std::mt19937& g,
                                           const AlignParams& p) {
  if (initial_msa.size() <= 2) return initial_msa;

  std::vector<std::string> best_msa = std::move(initial_msa);
  normalize_msa_lengths(best_msa);
  long long best_score = calculate_sp_score(best_msa, mode, fn, p);

  std::vector<int> indices(best_msa.size());
  std::iota(indices.begin(), indices.end(), 0);

  for (int iter = 0; iter < iterations; ++iter) {
    std::shuffle(indices.begin(), indices.end(), g);

    const std::size_t split_point = indices.size() / 2;
    if (split_point == 0 || split_point >= indices.size()) continue;

    std::vector<std::string> profileA, profileB;
    std::vector<int>         idxA,    idxB;
    profileA.reserve(split_point);
    profileB.reserve(indices.size() - split_point);
    idxA.reserve(split_point);
    idxB.reserve(indices.size() - split_point);

    for (std::size_t t = 0; t < split_point; ++t) {
      idxA.push_back(indices[t]);
      profileA.push_back(best_msa[indices[t]]);
    }
    for (std::size_t t = split_point; t < indices.size(); ++t) {
      idxB.push_back(indices[t]);
      profileB.push_back(best_msa[indices[t]]);
    }

    // Build consensus directly from full profiles — do NOT strip gap columns.
    // generate_consensus() already skips gap-only columns via the plurality
    // vote; projectGaps() handles the column structure correctly.
    std::string repA = generate_consensus(profileA);
    std::string repB = generate_consensus(profileB);

    // Strip gap characters from the consensus before alignment so that we
    // pass ungapped sequences to computeGlobalAlignment (it expects raw seqs).
    std::string ungapped_repA = ungap(repA);
    std::string ungapped_repB = ungap(repB);

    if (ungapped_repA.empty() || ungapped_repB.empty()) continue;

    std::string aligned_repA, aligned_repB;
    computeGlobalAlignment(ungapped_repA, ungapped_repB, mode, fn,
                           aligned_repA, aligned_repB, p);

    projectGaps(ungapped_repA, aligned_repA, profileA);
    projectGaps(ungapped_repB, aligned_repB, profileB);

    std::vector<std::string> new_msa(best_msa.size());
    for (std::size_t k = 0; k < idxA.size(); ++k) new_msa[idxA[k]] = profileA[k];
    for (std::size_t k = 0; k < idxB.size(); ++k) new_msa[idxB[k]] = profileB[k];

    normalize_msa_lengths(new_msa);

    long long new_score = calculate_sp_score(new_msa, mode, fn, p);
    if (new_score > best_score) {
      best_score = new_score;
      best_msa   = std::move(new_msa);

#pragma omp critical
      std::cout << "Thread " << omp_get_thread_num()
                << " found a better score: " << best_score << std::endl;
    }
  }

  return best_msa;
}

/**
 * @brief Performs iterative refinement in parallel using multiple threads.
 */
std::vector<std::string> refine_msa(std::vector<std::string> initial_msa,
                                    int rounds, int iterations_per_round,
                                    ScoreMode mode, ScoreFn fn,
                                    const AlignParams& p) {
  std::vector<std::string> global_best_msa = std::move(initial_msa);
  long long                global_best_score =
      calculate_sp_score(global_best_msa, mode, fn, p);
  // std::cout << "MSA score: " << global_best_score << std::endl;

  for (int r = 0; r < rounds; ++r) {
    // std::cout << "\n--- Starting Refinement Round " << r + 1 << "/" << rounds
    //           << " (Best score so far: " << global_best_score << ") ---"
    //           << std::endl;

    int                                   num_threads = omp_get_max_threads();
    std::vector<std::vector<std::string>> thread_results(num_threads);

#pragma omp parallel default(none) \
    shared(thread_results, global_best_msa, iterations_per_round, mode, fn, p)
    {
      int thread_id = omp_get_thread_num();
      // Each thread gets its own random number generator, seeded uniquely
      std::mt19937 g(std::random_device{}() + thread_id);

      // Each thread starts from the current global best and runs its own
      // refinement search
      thread_results[thread_id] = refine_msa_worker(
          global_best_msa, iterations_per_round, mode, fn, g, p);
    }  // All threads finish and synchronize here

    // Now, back in serial, let's check the results from all threads
    for (int i = 0; i < num_threads; ++i) {
      long long thread_score =
          calculate_sp_score(thread_results[i], mode, fn, p);
      if (thread_score > global_best_score) {
        // std::cout << "Round " << r + 1 << " update: Thread " << i << " found
        // a better score: " << thread_score << std::endl;
        global_best_score = thread_score;
        global_best_msa   = thread_results[i];
      }
    }
  }

  // std::cout << "\nMSA Score: " << global_best_score << "\n\n" << std::endl;
  return global_best_msa;
}

/** print MSA with block positions rather than headers **/
void printColorMSA(const std::vector<std::string>& aln) {
  const int m = checked_size_to_int(aln.size(), "aln.size()");
  const int L = checked_size_to_int(
      aln[0].size(), "aln[0].size()");  // now all aln[i].size() == L

  // track ?current ungapped? position for each sequence
  std::vector<int> pos(m, 1);

  for (int start = 0; start < L; start += LINE_WIDTH) {
    int end = std::min(start + LINE_WIDTH, L);

    // for each sequence, compute the start coordinate of this block
    std::vector<int> block_start = pos;

    // now scan through the block to advance pos[]
    for (int j = start; j < end; ++j) {
      for (int i = 0; i < m; ++i) {
        if (aln[i][j] != '-') ++pos[i];
      }
    }

    // block_end is pos-1
    std::vector<int> block_end(m);
    for (int i = 0; i < m; ++i) {
      block_end[i] = pos[i] - 1;
    }

    // now print each sequence line:
    for (int i = 0; i < m; ++i) {
      // print ?start?end? positions, padded to width 8
      std::cout << std::setw(8) << block_start[i] << " ";

      // figure out column?conservation for color
      for (int j = start; j < end; ++j) {
        char c = aln[i][j];
        // check if this column is fully conserved non?gap
        bool fullyConserved = true;
        for (int x = 0; x < m; ++x) {
          if (aln[x][j] != aln[i][j] || aln[x][j] == '-') {
            fullyConserved = false;
            break;
          }
        }

        if (c == '-')
          std::cout << RED << c << RESET;
        else if (fullyConserved)
          std::cout << GREEN << c << RESET;
        else
          std::cout << CYAN << c << RESET;
      }

      // print end coordinate
      std::cout << " " << std::setw(8) << block_end[i] << "\n";
    }
    std::cout << "\n";
  }
}

/**
 * @brief Saves the final MSA to a styled HTML file for visualization.
 * This creates a color-coded alignment with sequence start and end positions
 * for each block, similar to the console output.
 *
 * @param aln The final multiple sequence alignment.
 * @param hdrs A vector of the original, finalized sequence headers (used for
 * tooltips).
 * @param outdir The directory where the output HTML file will be saved.
 */
void saveMSA_to_HTML(const std::vector<std::string>& aln,
                     const std::vector<std::string>& hdrs,
                     const std::string&              outdir) {
  if (aln.empty()) {
    std::cerr << "Warning: Cannot generate HTML for an empty alignment."
              << std::endl;
    return;
  }

  if (aln.size() != hdrs.size()) {
    std::cerr << "Error: Mismatch between alignment size and header size."
              << std::endl;
    return;
  }

  const std::string output_filepath = outdir + "/msa_visualization.html";
  std::ofstream     html_file(output_filepath);

  if (!html_file.is_open()) {
    std::cerr << "Error: Could not open " << output_filepath << " for writing."
              << std::endl;
    return;
  }

  // 1. Write the HTML header and CSS for the combined layout
  html_file << R"delimiter(<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MSA Visualization (Combined)</title>
  <style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f8f9fa; color: #212529; }
    h1 { color: #343a40; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; }
    pre { font-family: 'Consolas', 'Menlo', 'Courier New', monospace; font-size: 14px; line-height: 1.2; border: 1px solid #ced4da; padding: 15px; border-radius: 8px; background-color: #ffffff; white-space: pre; overflow-x: auto; }
    .line-container { display: flex; align-items: center; margin-bottom: 3px; }
    .header { flex: 0 0 220px; font-weight: bold; color: #495057; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .position { flex: 0 0 70px; text-align: right; font-weight: bold; color: #6c757d; }
    .sequence-block { margin: 0 15px; }
    .conserved { color: #ffffff; background-color: #198754; font-weight: bold; }
    .residue { color: #212529; background-color: #e9ecef; }
    .gap { color: #dc3545; font-weight: bold; }
  </style>
</head>
<body>
  <h1>Multiple Sequence Alignment</h1>
  <pre>)delimiter";

  // 2. Position calculation and alignment block writing
  const size_t num_sequences    = aln.size();
  size_t alignment_length = 0;
  for (const auto& row : aln) {
    alignment_length = std::max(alignment_length, row.size());
  }
  const int    LINE_WIDTH       = 80;

  // Track "current ungapped" position for each sequence, starting from 1
  std::vector<int> pos(num_sequences, 1);

  for (size_t start_col = 0; start_col < alignment_length;
       start_col += LINE_WIDTH) {
    size_t end_col = std::min(start_col + LINE_WIDTH, alignment_length);

    // For each sequence, get the start coordinate of this block
    std::vector<int> block_start = pos;

    // Scan through the block to advance the main 'pos' vector for the *next*
    // block
    for (size_t j = start_col; j < end_col; ++j) {
      for (size_t i = 0; i < num_sequences; ++i) {
        if (aln[i][j] != '-') {
          pos[i]++;
        }
      }
    }

    // The end coordinate for this block is the newly advanced position minus 1
    std::vector<int> block_end(num_sequences);
    for (size_t i = 0; i < num_sequences; ++i) {
      block_end[i] = pos[i] - 1;
    }

    // Print each sequence line for this block
    for (size_t i = 0; i < num_sequences; ++i) {
      html_file << "<div class=\"line-container\">";

      // Layout: [Header] [Start Pos] [Sequence] [End Pos]
      {
        std::string safeHdr = htmlEscape(hdrs[i]);
        html_file << "<span class=\"header\" title=\"" << safeHdr
                  << "\">" << safeHdr << "</span>"
                  << "<span class=\"position\">" << block_start[i]
                  << "</span>" << "<span class=\"sequence-block\">";
      }

      // Write the colored characters for the current block
      for (size_t j = start_col; j < end_col; ++j) {
        char current_char = (j < aln[i].size()) ? aln[i][j] : '-';
        bool is_fully_conserved = (current_char != '-');
        if (is_fully_conserved) {
          for (size_t x = 0; x < num_sequences; ++x) {
            if (j >= aln[x].size() || aln[x][j] != current_char) {
              is_fully_conserved = false;
              break;
            }
          }
        }
        if (current_char == '-') {
          html_file << "<span class=\"gap\">-</span>";
        } else if (is_fully_conserved) {
          html_file << "<span class=\"conserved\">" << current_char
                    << "</span>";
        } else {
          html_file << "<span class=\"residue\">" << current_char << "</span>";
        }
      }
      html_file << "</span>";  // Close sequence-block

      // Print the end position and close the line container
      html_file << "<span class=\"position\">" << block_end[i]
                << "</span></div>\n";
    }
    html_file << "\n";  // Blank line between blocks
  }

  // 3. Write HTML footer
  html_file << R"delimiter(  </pre>
</body>
</html>)delimiter";

  html_file.close();
}

/**
 * @brief Computes a multiple sequence alignment using the STAR method.
 * This function iteratively aligns sequences to a growing center sequence,
 * projecting gaps from the center into the aligned sequences.
 *
 * @param hdrs The headers of the sequences (used for output).
 * @param seqs The sequences to align.
 * @param mode The scoring mode (DNA or Protein).
 * @param fn The scoring function to use.
 * @return A vector of aligned sequences.
 */
std::vector<std::string> msa_star(const std::vector<std::string>& hdrs,
                                  const std::vector<std::string>& seqs,
                                  ScoreMode mode, ScoreFn fn,
                                  const AlignParams& p) {
  const int                n = checked_size_to_int(seqs.size(), "seqs.size()");
  std::vector<std::string> aligned(n);
  std::string              center = seqs[0];
  aligned[0]                      = center;
  for (int i = 1; i < n; ++i) {
    std::string ac, as;
    computeGlobalAlignment(center, seqs[i], mode, fn, ac, as, p);
    // project
    std::vector<std::string> prev(aligned.begin(), aligned.begin() + i);
    projectGaps(center, ac, prev);
    for (int k = 0; k < i; ++k) aligned[k] = prev[k];
    aligned[i] = as;
    center     = ac;
  }
  return aligned;
}

/**
 * @brief Parses a Newick formatted guide tree into a binary Node tree.
 *
 * Leaf labels in the Newick string must exactly match entries in @p names;
 * internal nodes are implied by parentheses and do not carry names. Branch
 * lengths following a colon (":") are ignored and skipped. Whitespace is
 * ignored. If a leaf label is not found in @p names or if the Newick string
 * is malformed (e.g. mismatched parentheses or nodes with more than two
 * children), a std::runtime_error is thrown. The caller is responsible for
 * freeing the returned tree via @c free_tree().
 *
 * @param nwk Newick formatted string representing the guide tree.
 * @param names A vector of sequence names corresponding to the leaf nodes.
 *              The index in this vector is used as the Node::seq_index.
 * @return Pointer to the root of the constructed binary tree.
 * @throws std::runtime_error if a leaf label is missing from @p names or if
 *         the tree is structurally invalid.
 */
Node* parseNewick(const std::string&              nwk,
                  const std::vector<std::string>& names) {
  // Build a name->index lookup for quick leaf lookup
  std::unordered_map<std::string, int> name_to_index;
  name_to_index.reserve(names.size());
  for (std::size_t i = 0; i < names.size(); ++i) {
    // duplicate names have already been checked upstream
    name_to_index.emplace(names[i], static_cast<int>(i));
  }
  // Helper to attach a leaf with the given token to the current parent
  auto attach_leaf = [&](const std::string& label, std::stack<Node*>& st,
                         Node*& root) {
    if (label.empty()) return;
    auto it = name_to_index.find(label);
    if (it == name_to_index.end()) {
      throw std::runtime_error("Guide tree leaf label '" + label +
                               "' was not found in the sequence label map");
    }
    Node* leaf = new Node{true, it->second, {}, nullptr, nullptr};
    if (st.empty()) {
      // If no context, this is the root
      if (root) {
        // existing root – orphaned leaf, free and throw
        delete leaf;
        throw std::runtime_error("Multiple roots encountered in Newick parse");
      }
      root = leaf;
      return;
    }
    Node* parent = st.top();
    if (!parent->left) {
      parent->left = leaf;
    } else if (!parent->right) {
      parent->right = leaf;
    } else {
      // Parent already has two children – invalid binary tree
      delete leaf;
      throw std::runtime_error("Invalid guide tree: node has more than two children");
    }
  };
  std::stack<Node*> st;
  Node* root = nullptr;
  std::string token;
  for (std::size_t i = 0; i < nwk.size(); ++i) {
    char c = nwk[i];
    if (std::isspace(static_cast<unsigned char>(c))) {
      continue;
    }
    if (c == '(') {
      // Start a new internal node
      Node* node = new Node{false, -1, {}, nullptr, nullptr};
      if (!st.empty()) {
        Node* parent = st.top();
        if (!parent->left) {
          parent->left = node;
        } else if (!parent->right) {
          parent->right = node;
        } else {
          delete node;
          throw std::runtime_error("Invalid guide tree: node has more than two children");
        }
      } else if (!root) {
        root = node;
      } else {
        delete node;
        throw std::runtime_error("Multiple roots encountered in Newick parse");
      }
      st.push(node);
    } else if (c == ',') {
      // Finish the preceding token as a leaf
      attach_leaf(token, st, root);
      token.clear();
    } else if (c == ')') {
      // Finish the preceding token as a leaf
      attach_leaf(token, st, root);
      token.clear();
      // Pop current internal node
      if (st.empty()) {
        throw std::runtime_error("Invalid guide tree: unmatched ')'");
      }
      st.pop();
    } else if (c == ':') {
      // Finish label before branch length
      attach_leaf(token, st, root);
      token.clear();
      // Skip branch length field (digits, decimal point, exponent, sign)
      std::size_t j = i + 1;
      while (j < nwk.size()) {
        char d = nwk[j];
        if (std::isdigit(static_cast<unsigned char>(d)) || d == '.' || d == 'e' || d == 'E' || d == '+' || d == '-') {
          ++j;
        } else {
          break;
        }
      }
      i = j - 1;
    } else if (c == ';') {
      // Finish any trailing token and break
      attach_leaf(token, st, root);
      token.clear();
      break;
    } else {
      // Accumulate label characters
      token += c;
    }
  }
  // Final trailing label if any
  if (!token.empty()) {
    attach_leaf(token, st, root);
    token.clear();
  }
  if (!st.empty()) {
    throw std::runtime_error("Invalid guide tree: unmatched '(' characters");
  }
  return root;
}

/**
 * @brief Formats a compact single-line Newick string into a multi-line,
 * indented format for better readability.
 *
 * @param nwk The single-line Newick string generated by the UPGMA function.
 * @return A std::string containing the formatted, multi-line tree.
 */
std::string formatNewickString(const std::string& nwk) {
  if (nwk.empty()) {
    return "";
  }

  std::ostringstream formatted_tree;
  int                indent_level = 0;
  const std::string  indent_unit =
      "    ";  // Use 4 spaces for each indentation level

  for (char c : nwk) {
    if (c == ')') {
      // A closing parenthesis decreases indentation and moves to a new line
      // before being printed.
      indent_level--;
      formatted_tree << "\n";
      for (int i = 0; i < indent_level; ++i) {
        formatted_tree << indent_unit;
      }
    }

    // Print the character itself
    formatted_tree << c;

    if (c == '(') {
      // An opening parenthesis increases indentation and starts a new line
      // after being printed.
      indent_level++;
      formatted_tree << "\n";
      for (int i = 0; i < indent_level; ++i) {
        formatted_tree << indent_unit;
      }
    } else if (c == ',') {
      // A comma starts a new line at the same indentation level.
      formatted_tree << "\n";
      for (int i = 0; i < indent_level; ++i) {
        formatted_tree << indent_unit;
      }
    }
  }

  formatted_tree << "\n";  // Add a final newline for a clean file ending.
  return formatted_tree.str();
}

/**
 * @brief Recursively builds a profile for a binary tree node.
 * This function constructs the MSA profile for each node in the tree,
 * aligning child profiles and projecting gaps as necessary.
 *
 * @param n The current node in the binary tree.
 * @param seqs The original sequences used to build the tree.
 * @param mode The scoring mode (DNA or Protein).
 * @param fn The scoring function to use.
 * @return A vector of aligned sequences representing the profile at this node.
 */
std::vector<std::string> build_profile(Node* n, const std::vector<std::string>& seqs, ScoreMode mode, ScoreFn fn,
                                       const AlignParams& p) {
  // Safety check for null pointers passed from parent nodes
  if (!n) {
    return {};
  }

  // Case 1: The node is a leaf. Its profile (a single sequence)
  // was already seeded in the main() function. Just return it.
  if (n->leaf) {
    // Validate leaf index before dereferencing
    int idx = n->seq_index;
    if (idx < 0 || idx >= checked_size_to_int(seqs.size(), "seqs.size()")) {
      std::ostringstream oss;
      oss << "Invalid leaf sequence index " << idx;
      throw std::runtime_error(oss.str());
    }
    return { seqs[idx] };
  }

  // Recursively build the profiles for the children.
  auto A = build_profile(n->left, seqs, mode, fn, p);
  auto B = build_profile(n->right, seqs, mode, fn, p);

  // Case 2: The node is "unary" (has only one child profile).
  // No alignment is needed. The profile is just the child's profile.
  if (A.empty()) {
    n->profile = B;
    return n->profile;
  }
  if (B.empty()) {
    n->profile = A;
    return n->profile;
  }

  // Case 3: The node is binary. Align the two child profiles (A and B).
  // Generate a consensus sequence to represent each profile.
  std::string consA = generate_consensus(A);
  std::string consB = generate_consensus(B);

  // Align the representative strings.
  std::string aligned_consA, aligned_consB;
  computeGlobalAlignment(consA, consB, mode, fn, aligned_consA, aligned_consB,
                         p);

  // Project the gaps from the alignment into ALL sequences of each profile.
  projectGaps(consA, aligned_consA, A);
  projectGaps(consB, aligned_consB, B);

  // Merge the two aligned profiles into a single new profile.
  std::vector<std::string> M;
  M.reserve(A.size() + B.size());
  M.insert(M.end(), A.begin(), A.end());
  M.insert(M.end(), B.begin(), B.end());

  n->profile = std::move(M);
  return n->profile;
}

/**
 * @brief Holds the result of a gap-penalty grid search.
 */
struct GapSearchResult {
  long long score =
      -LLONG_MAX;  ///< Best SP score found across all parameter combos.
  double gap_open   = 0.0;  ///< Gap-open penalty that achieved @c score.
  double gap_extend = 0.0;  ///< Gap-extend penalty that achieved @c score.
};

/**
 * @brief Performs a parallel grid search to find optimal affine gap penalties.
 *
 * Evaluates a fixed grid of (gap_open � gap_extend) combinations. For each
 * combination it:
 *   -# Computes the full pairwise distance matrix with computeDistanceMatrix().
 *   -# Builds a UPGMA guide tree with buildUPGMATree().
 *   -# Constructs a progressive MSA with build_profile().
 *   -# Scores the MSA with calculate_sp_score().
 *
 * The search runs with @c collapse(2) OpenMP parallelism ? each
 * (i, j) grid cell is a completely independent task with its own
 * stack-allocated @c AlignParams, so there is no shared mutable state and
 * no data races.
 *
 * Default grid:
 *   - @c gap_open   : -25.0, -22.5, ?, -10.0 (7 values, step 2.5)
 *   - @c gap_extend :  -5.0,  -4.0, ?,  -1.0 (5 values, step 1.0)
 *
 * @param initial_seqs  Raw (ungapped) input sequences.
 * @param initial_hdrs  Sequence headers corresponding to @p initial_seqs.
 *                      Used as node labels in the guide tree.
 * @param mode          Scoring mode ? @c MODE_DNA or @c MODE_PROTEIN.
 * @param fn            Raw scoring function pointer compatible with @p mode.
 *
 * @return A @c GapSearchResult with the best SP score and the corresponding
 *         gap penalties.
 *
 * @note Each grid point allocates and immediately frees a full guide tree
 *       (via free_tree()). Memory usage is O(N) per thread at any one time.
 *
 * @see computeDistanceMatrix(), buildUPGMATree(), build_profile(),
 *      calculate_sp_score(), GapSearchResult, AlignParams
 */
GapSearchResult find_optimal_gap_penalties(
    const std::vector<std::string>& initial_seqs,
    const std::vector<std::string>& initial_hdrs,
    ScoreMode mode,
    ScoreFn fn) {

  // std::cout << "\n--- Starting search for optimal gap penalties ---"
  //           << std::endl;

  std::vector<double> open_penalties;
  std::vector<double> extend_penalties;

  /*
   * Protein and DNA/RNA use the same affine-gap principle:
   * gap opening must be substantially more expensive than gap extension.
   *
   * Because penalties are negative, "larger penalty" means larger absolute value.
   *
   * Example:
   *   gap_open   = -14.0
   *   gap_extend = -1.0
   *
   * This is valid because:
   *   |gap_open| > |gap_extend|
   */

  if (mode == MODE_PROTEIN) {
    /*
     * Protein alignments usually tolerate a broader gap-open search space.
     * BLOSUM62-based scoring often benefits from stronger opening penalties.
     */
    for (double o = -30.0; o <= -6.0; o += 2.0) {
      open_penalties.push_back(o);
    }

    for (double e = -5.0; e <= -0.25; e += 0.25) {
      extend_penalties.push_back(e);
    }

  } else {
    /*
     * DNA/RNA alignments generally need a slightly narrower gap-extension range.
     * Very high extension penalties can fragment nucleotide alignments badly.
     */
    for (double o = -25.0; o <= -5.0; o += 2.0) {
      open_penalties.push_back(o);
    }

    for (double e = -4.0; e <= -0.25; e += 0.25) {
      extend_penalties.push_back(e);
    }
  }

  std::vector<GapSearchResult> results(open_penalties.size() *
                                       extend_penalties.size());

#pragma omp parallel for collapse(2) schedule(dynamic) default(none) \
    shared(open_penalties, extend_penalties, results, initial_seqs,  \
           initial_hdrs, mode, fn)
  for (int i = 0;
       i < checked_size_to_int(open_penalties.size(),
                               "open_penalties.size()");
       ++i) {

    for (int j = 0;
         j < checked_size_to_int(extend_penalties.size(),
                                 "extend_penalties.size()");
         ++j) {

      AlignParams p;
      p.gap_open   = open_penalties[i];
      p.gap_extend = extend_penalties[j];

      const int index =
          i * checked_size_to_int(extend_penalties.size(),
                                  "extend_penalties.size()") +
          j;

      /*
       * Biological affine-gap sanity constraint.
       *
       * Opening a new gap should be substantially more expensive than
       * extending an existing gap.
       *
       * Because both values are negative:
       *
       *   std::abs(gap_open) >= 3.0 * std::abs(gap_extend)
       *
       * means:
       *
       *   -12.0 / -1.0 is valid
       *   -6.0  / -5.0 is rejected
       */
      if (std::abs(p.gap_open) < 3.0 * std::abs(p.gap_extend)) {
        results[index] = {LLONG_MIN, p.gap_open, p.gap_extend};
        continue;
      }

      auto D = computeDistanceMatrix(initial_seqs, mode, fn, p);

      /*
       * Use stable internal labels during temporary grid-search tree parsing.
       * This avoids Newick parsing problems if biological IDs contain
       * punctuation. The final production guide tree can still use preserved
       * sequence identifiers.
       */
      std::vector<std::string> internal_labels =
          make_internal_tree_labels(initial_hdrs.size());

      auto nwk = buildUPGMATree(D, internal_labels);

      Node* root = nullptr;
      try {
        root = parseNewick(nwk, internal_labels);
      } catch (const std::exception& e) {
        results[index] = {LLONG_MIN, p.gap_open, p.gap_extend};
        continue;
      }

      std::queue<Node*> q;
      if (root) {
        q.push(root);
      }

      while (!q.empty()) {
        Node* u = q.front();
        q.pop();

        if (!u) {
          continue;
        }

        if (u->leaf) {
          int idx = u->seq_index;

          if (idx < 0 ||
              idx >= checked_size_to_int(initial_seqs.size(),
                                         "initial_seqs.size()")) {
            free_tree(root);
            results[index] = {LLONG_MIN, p.gap_open, p.gap_extend};
            root = nullptr;
            break;
          }

          u->profile = {initial_seqs[idx]};

        } else {
          if (u->left) {
            q.push(u->left);
          }

          if (u->right) {
            q.push(u->right);
          }
        }
      }

      if (!root) {
        continue;
      }

      auto msa = build_profile(root, initial_seqs, mode, fn, p);
      free_tree(root);

      normalize_msa_lengths(msa);

      long long sc = calculate_sp_score(msa, mode, fn, p);

      results[index] = {sc, p.gap_open, p.gap_extend};
    }
  }

  GapSearchResult best_result;
  best_result.score = LLONG_MIN;

  for (const auto& result : results) {
    if (result.score > best_result.score) {
      best_result = result;
    }
  }

  if (best_result.score == LLONG_MIN) {
    throw std::runtime_error(
        "Gap-penalty grid search failed: no valid gap-open/gap-extend "
        "combination passed the affine-gap sanity constraints.");
  }

  // std::cout << "--- Finished gap penalty search ---\n"
  //           << "Best GAP_OPEN=" << best_result.gap_open
  //           << ", GAP_EXTEND=" << best_result.gap_extend
  //           << ", score=" << best_result.score
  //           << std::endl;

  return best_result;
}

/**
 * @brief Analyses a final MSA for per-column consensus matches and writes a
 *        detailed report and a FASTA consensus file to disk.
 *
 * Iterates over every column of the alignment. For columns where the consensus
 * character is not a gap, it reports which sequences carry a matching residue
 * together with that residue's 1-based position in the original (ungapped)
 * sequence. Output files:
 *   - @c <outdir>/consensus.fasta  ? FASTA-formatted consensus sequence.
 *   - @c <outdir>/consensus_details.txt ? per-column match report.
 *
 * @param msa           The final column-aligned MSA produced by refine_msa().
 * @param hdrs          Original sequence headers, one per row in @p msa.
 * @param consensus_seq Pre-computed consensus string from generate_consensus().
 *                      Must be the same length as every row of @p msa.
 * @param outdir        Output directory path (must already exist).
 *
 * @pre  @p msa is non-empty and all rows have equal length.
 * @pre  @p consensus_seq.size() == @p msa[0].size().
 *
 * @see generate_consensus(), refine_msa()
 */
void analyze_and_save_consensus(const std::vector<std::string>& msa,
                                const std::vector<std::string>& hdrs,
                                const std::string&              consensus_seq,
                                const std::string&              outdir) {
  if (msa.empty()) {
    return;
  }

  // Open the output file for the detailed report
  std::ofstream report_file(outdir + "/consensus_details.txt");
  if (!report_file) {
    std::cerr << "Warning: Could not open consensus_details.txt for writing."
              << std::endl;
    return;
  }

  report_file << "Consensus Sequence Match Report\n";
  report_file << "---------------------------------\n\n";

  const int num_seqs  = checked_size_to_int(msa.size(), "msa.size()");
  const int align_len = checked_size_to_int(msa[0].size(), "msa[0].size()");

  // This vector tracks the original, un-aligned position for each sequence
  std::vector<int> original_positions(num_seqs, 0);

  // Iterate through each column of the final alignment
  for (int j = 0; j < align_len; ++j) {
    char consensus_char = consensus_seq[j];

    // We only report on columns where the consensus is not a gap
    if (consensus_char != '-') {
      report_file << "Alignment Column: " << std::setw(5) << j + 1
                  << "  |  Consensus: " << consensus_char << "\n";

      // Check each sequence in the column for a match
      for (int i = 0; i < num_seqs; ++i) {
        char seq_char = msa[i][j];

        // If the character in the sequence is not a gap, its original position
        // counter increases
        if (seq_char != '-') {
          original_positions[i]++;
        }

        // If this sequence's character matches the consensus character, record
        // it
        if (seq_char == consensus_char) {
          report_file << "  - Match: Seq '" << hdrs[i]
                      << "' at original position " << original_positions[i]
                      << "\n";
        }
      }
      report_file << "\n";  // Add a newline for readability between columns
    } else {
      // If the consensus is a gap, we still need to advance the position
      // counters for any sequence that has a residue in this column.
      for (int i = 0; i < num_seqs; ++i) {
        if (msa[i][j] != '-') {
          original_positions[i]++;
        }
      }
    }
  }
  report_file.close();
}

/**
 * @brief Main function to run the MSA pipeline.
 * This function orchestrates reading input files, processing sequences,
 * computing the MSA, and saving results.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Exit status code.
 */
int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " [--mode dna|protein] [--gap_open O] [--gap_extend E] "
                 "outdir file1.fasta file2.fasta [...]\n";
    return 1;
  }

  // 1) Parse flags
  ScoreMode   mode = MODE_DNA;
  ScoreFn     fn   = edna_score;
  AlignParams params;
  bool        user_gap_open   = false;
  bool        user_gap_extend = false;
  int         argi            = 1;
  while (argi < argc && std::string(argv[argi]).rfind("--", 0) == 0) {
    std::string opt = argv[argi++];
    if (opt == "--mode") {
      if (argi >= argc) {
        std::cerr << "Error: --mode requires 'dna' or 'protein'.\n";
        return 1;
      }
      std::string m = argv[argi++];
      if (m == "dna") {
        mode = MODE_DNA;
        fn   = edna_score;
      } else if (m == "protein") {
        mode = MODE_PROTEIN;
        fn   = blosum62_score;
      } else {
        std::cerr << "Error: unknown mode '" << m << "'. Use dna or protein.\n";
        return 1;
      }
    } else if (opt == "--gap_open") {
      if (argi >= argc) {
        std::cerr << "Error: --gap_open requires a value.\n";
        return 1;
      }
      params.gap_open = std::stod(argv[argi++]);
      user_gap_open   = true;
    } else if (opt == "--gap_extend") {
      if (argi >= argc) {
        std::cerr << "Error: --gap_extend requires a value.\n";
        return 1;
      }
      params.gap_extend = std::stod(argv[argi++]);
      user_gap_extend   = true;
    } else {
      // unrecognized flag: step back and break
      --argi;
      break;
    }
  }

  // capture run start time for metadata
  auto run_start_point = std::chrono::system_clock::now();

  // 2) Next arg is outdir
  if (argi >= argc) {
    std::cerr << "Error: missing outdir.\n";
    return 1;
  }
  std::string outdir = argv[argi++];
  std::filesystem::create_directories(outdir);

  // 3) Remaining args are FASTA files
  if (argi >= argc) {
    std::cerr << "Error: need at least one FASTA file.\n";
    return 1;
  }
  std::vector<std::string> files;
  for (; argi < argc; ++argi) {
    files.emplace_back(argv[argi]);
  }

  // 4) Read sequences and build canonical headers
  const int n = checked_size_to_int(files.size(), "files.size()");
  std::vector<std::string> hdrs(n), seqs(n);
  for (int i = 0; i < n; ++i) {
    std::string raw_hdr;
    processFasta(files[i], raw_hdr, seqs[i]);
    // Generate a canonical label for each header
    hdrs[i] = canonicalize_fasta_header(raw_hdr, mode);
  }
  // Ensure that all canonical headers are unique
  {
    std::unordered_set<std::string> seen;
    for (const auto& h : hdrs) {
      if (!seen.insert(h).second) {
        throw std::runtime_error("Duplicate canonical header detected: " + h);
      }
    }
  }

  if (!user_gap_open || !user_gap_extend) {
    auto best = find_optimal_gap_penalties(seqs, hdrs, mode, fn);
    if (!user_gap_open) params.gap_open = best.gap_open;
    if (!user_gap_extend) params.gap_extend = best.gap_extend;
    std::cout << "\nOptimal parameters found: GAP_OPEN=" << params.gap_open
              << ", GAP_EXTEND=" << params.gap_extend
              // << " with score: " << best.score << "\n"
              << std::endl;
  } else {
    std::cout << "\nGAP_OPEN=" << params.gap_open
              << ", GAP_EXTEND=" << params.gap_extend << "\n"
              << std::endl;
  }

  try {
    // Compute pairwise distance matrix from global alignments
    auto D = computeDistanceMatrix(seqs, mode, fn, params);

    // Build guide tree using canonical sequence identifiers
    auto nwk = buildUPGMATree(D, hdrs);

    // Build pairwise residue library for LE-guided alignment
    auto pl = buildPairLib(seqs, mode, fn, params);

    // Helper: parse tree with canonical names and seed leaf profiles
    auto make_tree = [&]() -> Node* {
      Node* root = nullptr;
      try {
        root = parseNewick(nwk, hdrs);
      } catch (const std::exception& e) {
        std::cerr << "Error parsing guide tree: " << e.what() << "\n";
        return nullptr;
      }
      std::queue<Node*> q;
      if (root) q.push(root);
      while (!q.empty()) {
        Node* u = q.front();
        q.pop();
        if (!u) continue;
        if (u->leaf) {
          int idx = u->seq_index;
          if (idx < 0 || idx >= checked_size_to_int(seqs.size(), "seqs.size()")) {
            std::cerr << "\nFATAL LOGIC ERROR: Invalid seq_index " << idx
                      << std::endl;
            free_tree(root);
            return nullptr;
          }
          u->profile = {seqs[idx]};
        } else {
          if (u->left) q.push(u->left);
          if (u->right) q.push(u->right);
        }
      }
      return root;
    };

    // Run the three progressive alignment strategies in parallel
    std::vector<std::string> msa_nw, msa_fft, msa_le;
    long long score_nw = LLONG_MIN;
    long long score_fft = LLONG_MIN;
    long long score_le = LLONG_MIN;

#pragma omp parallel sections default(none) \
    shared(make_tree, msa_nw, msa_fft, msa_le, score_nw, score_fft, score_le, \
           mode, fn, params, seqs, pl)
    {
#pragma omp section
      {
        Node* root_nw = make_tree();
        if (root_nw) {
          msa_nw = build_profile(root_nw, seqs, mode, fn, params);
          score_nw = calculate_sp_score(msa_nw, mode, fn, params);
          free_tree(root_nw);
        }
      }
#pragma omp section
      {
        Node* root_fft = make_tree();
        if (root_fft) {
          msa_fft = build_profile_fft(root_fft, seqs, mode, fn, params);
          score_fft = calculate_sp_score(msa_fft, mode, fn, params);
          free_tree(root_fft);
        }
      }
#pragma omp section
      {
        Node* root_le = make_tree();
        if (root_le) {
          auto le_result = build_profile_le(root_le, seqs, pl, mode, fn, params);
          msa_le = le_result.profile;
          score_le = calculate_sp_score(msa_le, mode, fn, params);
          free_tree(root_le);
        }
      }
    }

    // Select best initial alignment
    std::vector<std::string> msa;
    std::string best_method;
    long long best_score = score_nw;
    msa = msa_nw;
    best_method = "Progressive NW";
    if (score_fft >= best_score) {
      best_score = score_fft;
      msa = msa_fft;
      best_method = "FFT-Seeded Banded";
    }
    if (score_le >= best_score) {
      best_score = score_le;
      msa = msa_le;
      best_method = "COFFEE LE-Guided";
    }

    // Refinement
    msa = refine_msa(msa, 3, 10, mode, fn, params);
    normalize_msa_lengths(msa);
    // Validate shape
    validate_msa_shape(msa, hdrs.size());

    // Compute percent identity matrix from final MSA
    auto pim = compute_percent_identity_from_msa(msa);

    // Generate job identifier
    std::string job_id = generate_job_id();

    // Determine file paths with required naming scheme
    std::string prefix = outdir + "/" + job_id;
    std::map<std::string,std::string> outputs;
    // Sequence file
    std::string seq_file = prefix + ".sequence";
    {
      std::ofstream seq_out(seq_file);
      if (!seq_out) {
        throw std::runtime_error("Failed to write sequence file");
      }
      for (std::size_t i = 0; i < hdrs.size(); ++i) {
        seq_out << ">" << hdrs[i] << "\n";
        const std::string& s = seqs[i];
        const std::size_t wrap_width = 80;
        for (std::size_t pos = 0; pos < s.size(); pos += wrap_width) {
          seq_out << s.substr(pos, std::min(wrap_width, s.size() - pos)) << "\n";
        }
      }
      seq_out.close();
    }
    outputs["sequence"] = seq_file;
    // CLUSTAL numbered alignment
    std::string clustal_file = prefix + ".aln-clustal_num";
    write_clustal_alignment(msa, hdrs, clustal_file);
    outputs["clustal_num"] = clustal_file;
    // FASTA aligned
    std::string fasta_file = prefix + ".fa";
    write_fasta_alignment(msa, hdrs, fasta_file);
    outputs["fasta"] = fasta_file;
    // Guide tree
    std::string tree_file = prefix + ".phylotree";
    {
      std::ofstream tf(tree_file);
      if (!tf) {
        throw std::runtime_error("Failed to write guide tree file");
      }
      // Optionally format the Newick for readability
      std::string formatted = formatNewickString(nwk);
      tf << formatted;
      tf.close();
    }
    outputs["guide_tree"] = tree_file;
    // Percent identity matrix
    std::string pim_file = prefix + ".pim";
    write_percent_identity_matrix(pim, hdrs, pim_file);
    outputs["percent_identity_matrix"] = pim_file;
    // Metadata JSON
    std::string metadata_file = prefix + ".metadata.json";
    outputs["metadata"] = metadata_file;
    // We'll fill after writing metadata
    // Submission XML
    std::string submission_file = prefix + ".submission";
    outputs["submission"] = submission_file;
    // Placeholder for zip path
    std::string zip_file = prefix + ".zip";
    outputs["zip"] = zip_file;

    // Compose command line string
    std::ostringstream cmdline_ss;
    for (int idx = 0; idx < argc; ++idx) {
      if (idx > 0) cmdline_ss << ' ';
      cmdline_ss << argv[idx];
    }
    std::string cmdline = cmdline_ss.str();

    // Determine sequence type string
    std::string seq_type = (mode == MODE_PROTEIN ? "protein" : "dna");

    // Record end time and compute duration
    auto run_end_point = std::chrono::system_clock::now();
    double duration_seconds = std::chrono::duration<double>(run_end_point - run_start_point).count();
    // ISO format times
    auto to_iso = [](std::chrono::system_clock::time_point tp) {
      auto tt = std::chrono::system_clock::to_time_t(tp);
      std::tm tm{};
#ifdef _WIN32
      gmtime_s(&tm, &tt);
#else
      gmtime_r(&tt, &tm);
#endif
      std::ostringstream oss;
      oss << std::put_time(&tm, "%FT%TZ");
      return oss.str();
    };
    std::string start_iso = to_iso(run_start_point);
    std::string end_iso = to_iso(run_end_point);
    // Determine number of threads
    int num_threads = omp_get_max_threads();
    bool avx2_enabled = true; // assume compiled with AVX2
#ifdef __AVX2__
    avx2_enabled = true;
#else
    avx2_enabled = false;
#endif
    bool openmp_enabled = true;
#ifndef _OPENMP
    openmp_enabled = false;
#endif
    // Write metadata JSON
    write_metadata_json(metadata_file, job_id, "msalign-compatible-tool", "1.0", start_iso, end_iso,
                        duration_seconds, cmdline, seq_file, seq_type, hdrs,
                        num_threads, avx2_enabled, openmp_enabled,
                        "global_alignment_identity", "UPGMA",
                        best_method, "iterative_split_refinement", best_method, outputs);
    // Write submission XML
    write_submission_xml(submission_file, job_id, cmdline, hdrs, outputs, seq_type);
    // Create zip archive containing outputs
    {
      std::ostringstream file_list;
      file_list << " \"" << seq_file << "\"";
      file_list << " \"" << clustal_file << "\"";
      file_list << " \"" << fasta_file << "\"";
      file_list << " \"" << tree_file << "\"";
      file_list << " \"" << pim_file << "\"";
      file_list << " \"" << metadata_file << "\"";
      file_list << " \"" << submission_file << "\"";
      std::string cmd = "zip -j -q \"" + zip_file + "\"" + file_list.str();
      int rc = std::system(cmd.c_str());
      if (rc != 0) {
        std::cerr << "Warning: zip command returned non-zero exit code" << std::endl;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}