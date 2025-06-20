/* @file main.cpp
 * @brief Main file for a multiple sequence alignment tool.
 * This tool supports both DNA and protein sequences, implements
 * multiple sequence alignment using a star method,
 * and allows for iterative refinement of the alignment.
 *
 * Author: Abhinav Mishra
 */

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <climits>
#include <array>
#include <set>
#include <immintrin.h>
#include <omp.h>
#include <queue>
#include <tuple>
#include <stack>
#include <cctype>
#include <random>
#include "EDNAFULL.h"
#include "EBLOSUM62.h"

// ANSI color codes
#define RESET "\033[0m"
#define GREEN "\033[32m"
#define RED   "\033[31m"
#define CYAN  "\033[36m"

// Gap penalties
// Use mutable variables
double GAP_OPEN   = -5.0;
double GAP_EXTEND = -1.0;
static const int LINE_WIDTH = 80;

// Scoring modes
enum ScoreMode { MODE_DNA, MODE_PROTEIN };
using ScoreFn = int(*)(char,char);

// DNA / EDNAFULL lookup
static const std::array<uint8_t,256> char2idx = [](){
    std::array<uint8_t,256> m{}; m.fill(255);
    m['A']=0; m['C']=1; m['G']=2; m['T']=3; m['U']=3;
    m['R']=4; m['Y']=5; m['S']=6; m['W']=7;
    m['K']=8; m['M']=9; m['B']=10; m['D']=11;
    m['H']=12; m['V']=13; m['N']=14; m['X']=14;
    return m;
}();

// Protein / BLOSUM62 lookup
static const std::array<uint8_t,256> prot_idx = [](){
    std::array<uint8_t,256> m{}; m.fill(255);
    const char* AA="ARNDCQEGHILKMFPSTWYVBZX*";
    for(int i=0; AA[i]; ++i) m[(uint8_t)AA[i]] = i;
    return m;
}();

/**
* @brief Computes the BLOSUM62 score for a pair of amino acids.
* This function maps characters to their corresponding indices in the BLOSUM62 matrix
* and returns the score for the pair.
*
* @param x The first amino acid character.
* @param y The second amino acid character.
* @return The BLOSUM62 score for the pair.
*/
inline int blosum62_score(char x, char y) {
    // map to uppercase in case it slipped through
    x = static_cast<char>(std::toupper((unsigned char)x));
    y = static_cast<char>(std::toupper((unsigned char)y));

    uint8_t ix = prot_idx[static_cast<uint8_t>(x)];
    if (ix == 255) ix = prot_idx['X'];         // unknown → X
    uint8_t iy = prot_idx[static_cast<uint8_t>(y)];
    if (iy == 255) iy = prot_idx['X'];

    return static_cast<int>(std::round(EBLOSUM62_matrix[ix][iy]));
}

/**
* @brief Computes the score for a pair of characters using the appropriate scoring matrix.
* This function uses the EDNAFULL matrix for DNA sequences and the BLOSUM62 matrix for protein sequences.
*
* @param x The first character.
* @param y The second character.
* @return The score for the character pair.
*/
inline int edna_score(char x, char y) {
    x = static_cast<char>(std::toupper((unsigned char)x));
    y = static_cast<char>(std::toupper((unsigned char)y));

    uint8_t ix = char2idx[static_cast<uint8_t>(x)];
    if (ix == 255) ix = char2idx['N'];         // unknown → N
    uint8_t iy = char2idx[static_cast<uint8_t>(y)];
    if (iy == 255) iy = char2idx['N'];

    return static_cast<int>(std::round(EDNAFULL_matrix[ix][iy]));
}

/**
 * @brief Computes the score for a pair of characters based on the specified scoring mode.
 * This function uses the EDNAFULL matrix for DNA sequences and the BLOSUM62 matrix for protein sequences.
 *
 * @param x The first character.
 * @param y The second character.
 * @param mode The scoring mode (DNA or Protein).
 * @return The score for the character pair.
 */
inline int score(char x,char y,ScoreMode mode){
    return mode==MODE_DNA ? edna_score(x,y) : blosum62_score(x,y);
}

/**
 * @brief Computes the global alignment score between two sequences using a modified
 *        Needleman-Wunsch algorithm with AVX2 vectorization.
 *
 * @param x The first sequence.
 * @param y The second sequence.
 * @param mode The scoring mode (DNA or Protein).
 * @param score_fn The scoring function to use for matches/mismatches.
 * @param aligned_x Output string for the aligned first sequence.
 * @param aligned_y Output string for the aligned second sequence.
 * @return The final alignment score.
 */
int computeGlobalAlignment(const std::string &x,
                           const std::string &y,
                           ScoreMode mode,
                           ScoreFn score_fn,
                           std::string &aligned_x,
                           std::string &aligned_y)
{
    const int m = x.size();
    const int n = y.size();

    // Round n up to a multiple of 8
    int n8 = (n + 7) & ~7;

    // Allocate aligned buffers of length n8+1
    std::vector<int> S_prev(n8+1), S_cur(n8+1);
    std::vector<int> E_cur(n8+1), F_prev(n8+1);
    std::vector<char> trace_buf((m+1)*(n+1));

    const __m256i vGapOpen   = _mm256_set1_epi32(int(GAP_OPEN));
    const __m256i vGapExtend = _mm256_set1_epi32(int(GAP_EXTEND));

    // init row 0
    S_prev[0] = 0;
    for(int j = 1; j <= n; ++j) {
        int e = (j==1 ? S_prev[j-1] + GAP_OPEN : E_cur[j-1] + GAP_EXTEND);
        S_prev[j] = e;
        E_cur[j]  = e;
        F_prev[j] = INT_MIN/2;
        trace_buf[j] = (j==1 ? 'E' : 'e');
    }
    // pad to n8
    for(int j = n+1; j <= n8; ++j) {
        S_prev[j] = INT_MIN/2;
        E_cur[j]  = INT_MIN/2;
        F_prev[j] = INT_MIN/2;
    }

    // temp arrays for vector loads/stores
    int *Sp  = S_prev.data();
    int *Sc  = S_cur.data();
    int *Ec  = E_cur.data();
    int *Fp  = F_prev.data();

    for(int i = 1; i <= m; ++i) {
        // scalar first column
        int openF = Sp[0] + GAP_OPEN;
        int extF  = Fp[0] + GAP_EXTEND;
        Sc[0] = std::max(openF, extF);
        Ec[0] = INT_MIN/2;
        Fp[0] = Sc[0];
        trace_buf[i*(n+1) + 0] = (Sc[0] == openF ? 'F' : 'f');

        // broadcast S_prev[i-1][j-1] lanewise
        __m256i vSdiag = _mm256_set1_epi32(Sp[0]);

        // vectorized inner loop: j=1..n8 step 8
        for(int j = 1; j <= n8; j += 8) {
            // load previous vectors
            __m256i vSp   = _mm256_loadu_si256((__m256i*)&Sp[j]);
            __m256i vFp   = _mm256_loadu_si256((__m256i*)&Fp[j]);
            __m256i vEc   = _mm256_loadu_si256((__m256i*)&Ec[j-1]); // note shift for E
            __m256i vScm1 = _mm256_loadu_si256((__m256i*)&Sc[j-1]); // current S[j-1]

            // compute F = max(S_prev + gapOpen, F_prev + gapExtend)
            __m256i vF = _mm256_max_epi32(
                _mm256_add_epi32(vSp, vGapOpen),
                _mm256_add_epi32(vFp, vGapExtend)
            );

            // compute E = max(S_cur[j-1] + gapOpen, E_cur[j-1] + gapExtend)
            __m256i vE = _mm256_max_epi32(
                _mm256_add_epi32(vScm1, vGapOpen),
                _mm256_add_epi32(vEc,   vGapExtend)
            );

            // build vSdiag: prev_S[j-1], prev_F[j-1], prev_E[j-1]
            __m256i vFp_shift = _mm256_loadu_si256((__m256i*)&Fp[j-1]);
            __m256i vEp_shift = _mm256_loadu_si256((__m256i*)&Ec[j-1]);
            __m256i vBestPrev = _mm256_max_epi32(
                _mm256_max_epi32(vSdiag, vFp_shift),
                vEp_shift
            );

            // compute match/mismatch scores vector
            int scores[8];
            for(int k=0; k<8; ++k) {
                int idx = j+k-1;
                int sc = 0;
                if (idx < n) sc = score_fn(x[i-1], y[idx]);
                scores[k] = sc;
            }
            __m256i vMatch = _mm256_loadu_si256((__m256i*)scores);

            // M = BestPrev + match
            __m256i vM = _mm256_add_epi32(vBestPrev, vMatch);

            // S = max(M, E, F)
            __m256i vS = _mm256_max_epi32(
                _mm256_max_epi32(vM, vE),
                vF
            );

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
            for(int k=0; k<8; ++k) {
                if (j+k <= n) {
                    char ptr = 'M'; // Default to Match
                    // The highest score determines the path.
                    // avoids the out-of-bounds read.
                    if (Sblock[k] == Eblock[k]) {
                        ptr = 'E'; // Gap in sequence X (Insertion)
                    } else if (Sblock[k] == Fblock[k]) {
                        ptr = 'F'; // Gap in sequence Y (Deletion)
                    }
                    trace_buf[i*(n+1) + (j+k)] = ptr;
                }
            }
        }

        // swap rows
        std::swap(Sp, Sc);
        std::swap(Fp, Ec);
    }

    // final score
    int finalScore = Sp[n];

    // scalar traceback: identical to your existing code
    aligned_x.clear();
    aligned_y.clear();
    int i = m, j = n;
    while (i > 0 || j > 0) {
        char p = trace_buf[i*(n+1) + j];
        if      (p == 'M') { aligned_x.push_back(x[i-1]); aligned_y.push_back(y[j-1]); --i; --j; }
        else if (p=='F'||p=='f') { aligned_x.push_back(x[i-1]); aligned_y.push_back('-'); --i; }
        else if (p=='E'||p=='e') { aligned_x.push_back('-'); aligned_y.push_back(y[j-1]); --j; }
        else {
            if (i>0) { aligned_x.push_back(x[i-1]); aligned_y.push_back('-'); --i; }
            else     { aligned_x.push_back('-'); aligned_y.push_back(y[j-1]); --j; }
        }
    }
    std::reverse(aligned_x.begin(), aligned_x.end());
    std::reverse(aligned_y.begin(), aligned_y.end());
    return finalScore;
}

/** @brief Sanitizes a FASTA header by replacing spaces with underscores.
 * This is useful for ensuring that headers are valid identifiers in various contexts.
 * @param header The header string to sanitize.
 */
void sanitize_header(std::string& header) {
    for (char &c : header) {
        if (isspace(static_cast<unsigned char>(c))) {
            c = '_';
        }
    }
}

/**
 * @brief Processes a FASTA file to extract the header and sequence.
 * This function reads a FASTA file, extracts the first header line,
 * and concatenates all sequence lines into a single uppercase string.
 *
 * @param fn The filename of the FASTA file.
 * @param hdr Output parameter for the header (without '>' prefix).
 * @param seq Output parameter for the sequence (uppercase, no gaps).
 */
void processFasta(const std::string &fn, std::string &hdr, std::string &seq) {
    std::ifstream f(fn);
    if (!f) throw std::runtime_error("Cannot open " + fn);
    hdr.clear();
    seq.clear();
    std::string line;
    bool gotHdr = false;

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            if (!gotHdr) {  // only first header
                hdr = line.substr(1);
                gotHdr = true;
            }
            continue;
        }
        // sanitize: for each char, if A–Z or a–z, convert to uppercase and append
        for (char c : line) {
            if (std::isalpha(static_cast<unsigned char>(c))) {
                seq.push_back(std::toupper(static_cast<unsigned char>(c)));
            }
        }
    }
}

/**
 * @brief Generates a consensus sequence from a multiple sequence alignment profile.
 * This function computes the most frequent character in each column of the profile,
 * ignoring gaps ('-'), and returns the resulting consensus sequence.
 *
 * @param profile A vector of strings representing the aligned sequences.
 * @return The consensus sequence as a string.
 */
std::string generate_consensus(const std::vector<std::string>& profile) {
    if (profile.empty() || profile[0].empty()) {
        return "";
    }
    std::string consensus = "";
    int align_len = profile[0].size();
    int num_seqs = profile.size();

    for (int j = 0; j < align_len; ++j) {
        std::array<int, 256> counts{}; // Initialize all counts to 0
        int max_count = 0;
        char best_char = '-';

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
 * @brief Calculates the Sum-of-Pairs (SP) score for a given MSA with a true affine gap penalty.
 * This version iterates over each pair of sequences to correctly apply open and extend penalties.
 */
long long calculate_sp_score(const std::vector<std::string>& msa, ScoreMode mode, ScoreFn fn) {
    if (msa.empty() || msa[0].empty()) {
        return 0;
    }
    long long total_score = 0;
    int num_seqs = msa.size();
    int align_len = msa[0].size();

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
                    total_score += score(char_i, char_k, mode);
                    in_gap_for_this_pair = false; // The gap (if any) has ended.

                } else if (char_i != char_k) { // This condition is true only for Residue vs Gap
                    // --- Case 2: Residue vs Gap ---
                    if (in_gap_for_this_pair) {
                        // We are already in a gap, so this is an extension.
                        total_score += (long long)GAP_EXTEND;
                    } else {
                        // This is the first column of a new gap for this pair.
                        // Apply both the open and the first extend penalty.
                        total_score += (long long)GAP_OPEN + (long long)GAP_EXTEND;
                    }
                    in_gap_for_this_pair = true; // We are now in a gap state.

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
 * @brief Projects gaps from a newly aligned representative sequence into a profile.
 * This version correctly compares the old and new representatives to only insert
 * newly added gaps, preventing alignment corruption.
 */
void projectGaps(const std::string &oldc, const std::string &newc, std::vector<std::string> &seqs) {
    if (seqs.empty()) return;

    size_t old_idx = 0;
    for (size_t new_idx = 0; new_idx < newc.size(); ++new_idx) {
        // Check if the character in the new sequence corresponds to a character from the old one
        if (old_idx < oldc.size() && newc[new_idx] == oldc[old_idx]) {
            old_idx++; // advance the pointer.
        } else {
            // This is a new gap that was inserted.
            // Insert a column of gaps into the profile at this new position.
            for (auto &s : seqs) {
                s.insert(s.begin() + new_idx, '-');
            }
        }
    }
}

/**
 * @brief Compute identity‐based distance matrix in parallel.
 */
std::vector<std::vector<double>>
computeDistanceMatrix(const std::vector<std::string>& seqs,
                      ScoreMode mode, ScoreFn fn)
{
    size_t n = seqs.size();
    std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)n; ++i) {
        for (int j = i+1; j < (int)n; ++j) {
            std::string ax, ay;
            computeGlobalAlignment(seqs[i], seqs[j], mode, fn, ax, ay);
            int L = ax.size(), match = 0;
            // count matches
            for (int k = 0; k < L; ++k) {
                if (ax[k] == ay[k] && ax[k] != '-') ++match;
            }
            double identity = (L>0 ? double(match)/L : 0.0);
            double dist = 1.0 - identity;
            D[i][j] = D[j][i] = dist;
        }
    }
    return D;
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
                        const std::vector<std::string>& hdrs,
                        const std::string& outdir)
{
    if (D.empty() || D.size() != hdrs.size()) {
        std::cerr << "Warning: Cannot save identity matrix due to size mismatch or empty data." << std::endl;
        return;
    }

    const std::string output_filepath = outdir + "/identity_matrix.txt";
    std::ofstream matrix_file(output_filepath);

    if (!matrix_file.is_open()) {
        std::cerr << "Error: Could not open " << output_filepath << " for writing." << std::endl;
        return;
    }

    size_t n = hdrs.size();

    // Find the length of the longest header to format the output neatly
    size_t max_hdr_len = 0;
    for(const auto& h : hdrs) {
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
    // std::cout << "\nPairwise identity matrix saved to: " << output_filepath << std::endl;
}

/**
 * @brief Build UPGMA tree via a min-heap in O(n^2 log n).
 *
 * @param D     Symmetric distance matrix.
 * @param names Cluster labels (will be updated in place).
 * @return      Newick string.
 */
std::string buildUPGMATree(std::vector<std::vector<double>> D,
                           std::vector<std::string> names)
{
    int n = names.size();
    struct Cluster { std::string nwk; int size; };
    std::vector<Cluster> C(n);
    std::vector<bool> alive(n,true);
    for (int i = 0; i < n; ++i) C[i] = { names[i], 1 };

    // Min‐heap of (distance, a, b)
    using Entry = std::tuple<double,int,int>;
    auto cmp = [](Entry const &a, Entry const &b){
        return std::get<0>(a) > std::get<0>(b);
    };
    std::priority_queue<Entry, std::vector<Entry>, decltype(cmp)> pq(cmp);

    // seed heap
    for (int i = 0; i < n; ++i)
        for (int j = i+1; j < n; ++j)
            pq.emplace(D[i][j], i, j);

    int remaining = n;
    while (remaining > 1) {
        auto [d, a, b] = pq.top(); pq.pop();
        if (!alive[a] || !alive[b]) continue;  // stale entry

        // merge b into a
        double half = d * 0.5;
        std::ostringstream nw;
        nw << "(" << C[a].nwk << ":" << half
           << "," << C[b].nwk << ":" << half << ")";
        C[a].nwk = nw.str();
        C[a].size += C[b].size;
        alive[b] = false;
        --remaining;

        // update distances from a to all k
        for (int k = 0; k < n; ++k) {
            if (alive[k] && k != a) {
                // weighted average
                double dk = (D[a][k]*C[a].size + D[b][k]*C[b].size)
                            / (C[a].size + C[b].size);
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
 * @brief A single "worker" function that attempts to improve an MSA for a set number of iterations.
 * This function is designed to be called in parallel. It starts with an initial MSA and
 * returns the best one it could find in its allotted iterations.
 *
 * @param initial_msa The starting MSA for this worker.
 * @param iterations The number of random splits/realigns to attempt.
 * @param mode The sequence mode (DNA or Protein).
 * @param fn The scoring function to use.
 * @param g A reference to a seeded random number generator.
 * @return The best MSA found by this worker.
 */
std::vector<std::string> refine_msa_worker(std::vector<std::string> initial_msa, int iterations, ScoreMode mode, ScoreFn fn, std::mt19937& g) {
    // A profile with 2 or fewer sequences cannot be split and refined.
    if (initial_msa.size() <= 2) {
        return initial_msa;
    }

    std::vector<std::string> best_msa = initial_msa;
    long long best_score = calculate_sp_score(best_msa, mode, fn);

    // Create a vector of indices [0, 1, 2, ...] to shuffle for random splitting.
    std::vector<int> indices(initial_msa.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Main refinement loop for this worker
    for (int i = 0; i < iterations; ++i) {
        // Shuffle the indices to create a random partition
        std::shuffle(indices.begin(), indices.end(), g);

        // 1. Split the current best alignment into two random, non-empty profiles (A and B)
        std::vector<std::string> profileA, profileB;
        int split_point = initial_msa.size() / 2;
        for (size_t j = 0; j < initial_msa.size(); ++j) {
            if (j < split_point) {
                profileA.push_back(best_msa[indices[j]]);
            } else {
                profileB.push_back(best_msa[indices[j]]);
            }
        }

        // 2. Re-align the two profiles by aligning their consensus sequences
        std::string repA = generate_consensus(profileA);
        std::string repB = generate_consensus(profileB);

        std::string aligned_repA, aligned_repB;
        computeGlobalAlignment(repA, repB, mode, fn, aligned_repA, aligned_repB);

        // 3. Project the newly introduced gaps back into the full profiles
        projectGaps(repA, aligned_repA, profileA);
        projectGaps(repB, aligned_repB, profileB);

        // 4. Merge the realigned profiles to create the new candidate MSA
        std::vector<std::string> new_msa = profileA;
        new_msa.insert(new_msa.end(), profileB.begin(), profileB.end());

        // 5. If the new alignment has a better score, adopt it as the new best
        long long new_score = calculate_sp_score(new_msa, mode, fn);
        if (new_score > best_score) {
            best_score = new_score;
            best_msa = new_msa;

            // This console output is preserved. It will show progress from individual threads.
            // It's helpful for seeing how active the search is.
            // #pragma omp critical
            // {
            //     std::cout << "  Thread " << omp_get_thread_num()
            //               << " found a better score: " << best_score << std::endl;
            // }
        }
    }
    // Return the best alignment this specific worker was able to find.
    return best_msa;
}

/**
 * @brief Performs iterative refinement in parallel using multiple threads.
 */
std::vector<std::string> refine_msa(std::vector<std::string> initial_msa, int rounds, int iterations_per_round, ScoreMode mode, ScoreFn fn) {
    std::vector<std::string> global_best_msa = initial_msa;
    long long global_best_score = calculate_sp_score(global_best_msa, mode, fn);
    std::cout << "Initial MSA score: " << global_best_score << std::endl;

    for (int r = 0; r < rounds; ++r) {
        std::cout << "\n--- Starting Refinement Round " << r + 1 << "/" << rounds
                  << " (Best score so far: " << global_best_score << ") ---" << std::endl;

        int num_threads = omp_get_max_threads();
        std::vector<std::vector<std::string>> thread_results(num_threads);

        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            // Each thread gets its own random number generator, seeded uniquely
            std::mt19937 g(std::random_device{}() + thread_id);

            // Each thread starts from the current global best and runs its own refinement search
            thread_results[thread_id] = refine_msa_worker(global_best_msa, iterations_per_round, mode, fn, g);
        } // All threads finish and synchronize here

        // Now, back in serial, let's check the results from all threads
        for (int i = 0; i < num_threads; ++i) {
            long long thread_score = calculate_sp_score(thread_results[i], mode, fn);
            if (thread_score > global_best_score) {
                // std::cout << "Round " << r + 1 << " update: Thread " << i << " found a better score: " << thread_score << std::endl;
                global_best_score = thread_score;
                global_best_msa = thread_results[i];
            }
        }
    }

    std::cout << "\nFinished Iterative refinement. Final score: " << global_best_score << "\n\n" << std::endl;
    return global_best_msa;
}

/** print MSA with block positions rather than headers **/
void printColorMSA(const std::vector<std::string> &aln) {
    int m = aln.size();
    int L = static_cast<int>(aln[0].size());  // now all aln[i].size() == L

    // track “current ungapped” position for each sequence
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
            // print “start–end” positions, padded to width 8
            std::cout << std::setw(8) << block_start[i]
                      << " ";

            // figure out column‐conservation for color
            for (int j = start; j < end; ++j) {
                char c = aln[i][j];
                // check if this column is fully conserved non‐gap
                bool fullyConserved = true;
                for (int x = 0; x < m; ++x) {
                    if (aln[x][j] != aln[i][j] || aln[x][j] == '-') {
                        fullyConserved = false;
                        break;
                    }
                }

                if (c == '-')        std::cout << RED << c << RESET;
                else if (fullyConserved) std::cout << GREEN << c << RESET;
                else                  std::cout << CYAN << c << RESET;
            }

            // print end coordinate
            std::cout << " " << std::setw(8) << block_end[i]
                      << "\n";
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
 * @param hdrs A vector of the original, finalized sequence headers (used for tooltips).
 * @param outdir The directory where the output HTML file will be saved.
 */
void saveMSA_to_HTML(const std::vector<std::string>& aln, const std::vector<std::string>& hdrs, const std::string& outdir) {
    if (aln.empty()) {
        std::cerr << "Warning: Cannot generate HTML for an empty alignment." << std::endl;
        return;
    }

    if (aln.size() != hdrs.size()) {
        std::cerr << "Error: Mismatch between alignment size and header size." << std::endl;
        return;
    }

    const std::string output_filepath = outdir + "/msa_visualization.html";
    std::ofstream html_file(output_filepath);

    if (!html_file.is_open()) {
        std::cerr << "Error: Could not open " << output_filepath << " for writing." << std::endl;
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
    const size_t num_sequences = aln.size();
    const size_t alignment_length = aln[0].length();
    const int LINE_WIDTH = 80;

    // Track "current ungapped" position for each sequence, starting from 1
    std::vector<int> pos(num_sequences, 1);

    for (size_t start_col = 0; start_col < alignment_length; start_col += LINE_WIDTH) {
        size_t end_col = std::min(start_col + LINE_WIDTH, alignment_length);

        // For each sequence, get the start coordinate of this block
        std::vector<int> block_start = pos;

        // Scan through the block to advance the main 'pos' vector for the *next* block
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
            html_file << "<span class=\"header\" title=\"" << hdrs[i] << "\">" << hdrs[i] << "</span>"
                      << "<span class=\"position\">" << block_start[i] << "</span>"
                      << "<span class=\"sequence-block\">";

            // Write the colored characters for the current block
            for (size_t j = start_col; j < end_col; ++j) {
                char current_char = aln[i][j];
                bool is_fully_conserved = (current_char != '-');
                if (is_fully_conserved) {
                    for (size_t x = 0; x < num_sequences; ++x) {
                        if (aln[x][j] != current_char) {
                            is_fully_conserved = false;
                            break;
                        }
                    }
                }
                if (current_char == '-') { html_file << "<span class=\"gap\">-</span>"; }
                else if (is_fully_conserved) { html_file << "<span class=\"conserved\">" << current_char << "</span>"; }
                else { html_file << "<span class=\"residue\">" << current_char << "</span>"; }
            }
            html_file << "</span>"; // Close sequence-block

            // Print the end position and close the line container
            html_file << "<span class=\"position\">" << block_end[i] << "</span></div>\n";
        }
        html_file << "\n"; // Blank line between blocks
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
std::vector<std::string> msa_star(const std::vector<std::string> &hdrs,
                                  const std::vector<std::string> &seqs,
                                  ScoreMode mode, ScoreFn fn)
{
    int n=seqs.size();
    std::vector<std::string> aligned(n);
    std::string center = seqs[0]; aligned[0]=center;
    for(int i=1;i<n;++i){
        std::string ac, as;
        computeGlobalAlignment(center, seqs[i], mode, fn, ac, as);
        // project
        std::vector<std::string> prev(aligned.begin(), aligned.begin()+i);
        projectGaps(center, ac, prev);
        for(int k=0;k<i;++k) aligned[k]=prev[k];
        aligned[i] = as;
        center = ac;
    }
    return aligned;
}

// a tiny binary tree node for your guide‐tree
struct Node {
    bool              leaf;
    int               seq_index;   // which original sequence, if leaf
    std::vector<std::string> profile; // current MSA block under this node
    Node             *left=nullptr, *right=nullptr;
};

/**
 * @brief Parses a Newick formatted tree string into a binary tree structure.
 * This function constructs a binary tree from a Newick string, where each leaf node
 * corresponds to a sequence index from the provided names vector.
 *
 * @param nwk The Newick formatted string representing the guide tree.
 * @param names A vector of sequence names corresponding to the leaf nodes.
 * @return A pointer to the root of the constructed binary tree.
 */
Node* parseNewick(const std::string& nwk, const std::vector<std::string>& names) {
    auto find_name_index = [&](const std::string& name) {
        for (size_t i = 0; i < names.size(); ++i) {
            if (names[i] == name) {
                return static_cast<int>(i);
            }
        }
        // This is a critical failure point. A name from the tree was not in the input file list.
        std::cerr << "\nFATAL PARSING ERROR: The name '" << name
                  << "' from the guide tree was not found in the list of input sequence headers." << std::endl;
        return -1;
    };

    std::stack<Node*> node_stack;
    Node* root = nullptr;
    std::string current_text;

    for (size_t i = 0; i < nwk.length(); ++i) {
        char c = nwk[i];

        if (isspace(c)) continue; // Ignore whitespace

        // Check for delimiters that end a text block (a name)
        if (c == ',' || c == ')' || c == ':' || c == ';') {
            if (!current_text.empty()) {
                if (node_stack.empty()) { // Case for a single-node tree like "A;"
                     root = new Node{true, find_name_index(current_text), {}, nullptr, nullptr};
                } else {
                    Node* parent = node_stack.top();
                    Node* leaf = new Node{true, find_name_index(current_text), {}, nullptr, nullptr};
                    if (parent->left == nullptr) parent->left = leaf;
                    else parent->right = leaf;
                }
                current_text.clear();
            }

            if (c == ')') {
                if (!node_stack.empty()) node_stack.pop();
            } else if (c == ':') {
                // Skip over the branch length that follows the colon
                while (i + 1 < nwk.length() && (isdigit(nwk[i + 1]) || nwk[i + 1] == '.' || nwk[i + 1] == 'e' || nwk[i + 1] == '-')) {
                    i++;
                }
            }
        } else if (c == '(') {
            Node* new_node = new Node{false, -1, {}, nullptr, nullptr};
            if (root == nullptr) {
                root = new_node;
            }
            if (!node_stack.empty()) {
                Node* parent = node_stack.top();
                if (parent->left == nullptr) parent->left = new_node;
                else parent->right = new_node;
            }
            node_stack.push(new_node);
        } else {
            // It's a regular character, part of a name
            current_text += c;
        }
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
    int indent_level = 0;
    const std::string indent_unit = "    "; // Use 4 spaces for each indentation level

    for (char c : nwk) {
        if (c == ')') {
            // A closing parenthesis decreases indentation and moves to a new line before being printed.
            indent_level--;
            formatted_tree << "\n";
            for (int i = 0; i < indent_level; ++i) {
                formatted_tree << indent_unit;
            }
        }

        // Print the character itself
        formatted_tree << c;

        if (c == '(') {
            // An opening parenthesis increases indentation and starts a new line after being printed.
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

    formatted_tree << "\n"; // Add a final newline for a clean file ending.
    return formatted_tree.str();
}

/**
 * @brief Recursively builds a profile for a binary tree node.
 * This function constructs the MSA profile for each node in the tree,
 * aligning child profiles and projecting gaps as necessary.
 *
 * @param n The current node in the binary tree.
 * @param mode The scoring mode (DNA or Protein).
 * @param fn The scoring function to use.
 * @return A vector of aligned sequences representing the profile at this node.
 */
std::vector<std::string>
build_profile(Node *n, ScoreMode mode, ScoreFn fn) {
    // Safety check for null pointers passed from parent nodes
    if (!n) {
        return {};
    }

    // Case 1: The node is a leaf. Its profile (a single sequence)
    // was already seeded in the main() function. Just return it.
    if (n->leaf) {
        return n->profile;
    }

    // Recursively build the profiles for the children.
    auto A = build_profile(n->left,  mode, fn);
    auto B = build_profile(n->right, mode, fn);

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
    computeGlobalAlignment(consA, consB, mode, fn, aligned_consA, aligned_consB);

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

// A simple struct to hold the results of our search
struct GapSearchResult {
    long long score = -__LONG_LONG_MAX__;
    double gap_open = 0.0;
    double gap_extend = 0.0;
};

/**
 * @brief Searches for the optimal gap penalties in parallel.
 * This function performs a grid search over a range of gap open and extend penalties,
 * building a complete MSA for each combination and returning the best scoring one.
 *
 * @param initial_seqs The initial sequences to align.
 * @param initial_hdrs The headers corresponding to the initial sequences.
 * @param mode The scoring mode (DNA or Protein).
 * @param fn The scoring function to use.
 * @return A GapSearchResult containing the best score and corresponding gap penalties.
 */
GapSearchResult find_optimal_gap_penalties(
    const std::vector<std::string>& initial_seqs,
    const std::vector<std::string>& initial_hdrs,
    ScoreMode mode,
    ScoreFn fn)
{
    std::cout << "\n--- Starting search for optimal gap penalties ---" << std::endl;

    // 1. Define the grid of parameters to search
    std::vector<double> open_penalties;
    for (double o = -25.0; o <= -10.0; o += 2.5) open_penalties.push_back(o); // e.g., -25, -22.5, ... -10

    std::vector<double> extend_penalties;
    for (double e = -5.0; e <= -1.0; e += 1.0) extend_penalties.push_back(e); // e.g., -5, -4, -3, -2, -1

    std::vector<GapSearchResult> results(open_penalties.size() * extend_penalties.size());

    // 2. Perform the grid search in parallel
    // The collapse(2) clause maps the nested loops to a single parallel loop.
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < open_penalties.size(); ++i) {
        for (int j = 0; j < extend_penalties.size(); ++j) {
            // Each thread works with its own private copy of the gap penalties
            GAP_OPEN = open_penalties[i];
            GAP_EXTEND = extend_penalties[j];

            // 3. For each parameter set, build a complete initial MSA from scratch
            // Note: We are NOT passing gap penalties as arguments, because they are threadprivate
            auto D = computeDistanceMatrix(initial_seqs, mode, fn);
            auto nwk = buildUPGMATree(D, initial_hdrs);
            Node* root = parseNewick(nwk, initial_hdrs);

            // Seed the leaves of the tree
            std::queue<Node*> q;
            if (root) q.push(root);
            while(!q.empty()) {
                Node* u = q.front(); q.pop();
                if (!u) continue;
                if (u->leaf) u->profile = { initial_seqs[u->seq_index] };
                else {
                    if (u->left) q.push(u->left);
                    if (u->right) q.push(u->right);
                }
            }

            // Build the MSA and calculate its score
            auto msa = build_profile(root, mode, fn);
            long long score = calculate_sp_score(msa, mode, fn);

            // Store the result
            int index = i * extend_penalties.size() + j;
            results[index] = {score, GAP_OPEN, GAP_EXTEND};

            // #pragma omp critical
            // {
            //     std::cout << "  Tested (Open=" << GAP_OPEN << ", Extend=" << GAP_EXTEND << "). Score: " << score << std::endl;
            // }

            // A more complete implementation would free the memory allocated by parseNewick here.
        }
    }

    // 4. Find the best result from all the parallel tasks
    GapSearchResult best_result;
    for (const auto& result : results) {
        if (result.score > best_result.score) {
            best_result = result;
        }
    }

    std::cout << "\n--- Finished parameter search ---" << std::endl;
    return best_result;
}

/**
 * @brief Analyzes a final MSA to find positional matches for the consensus sequence
 * and saves the consensus and a detailed report to files.
 *
 * @param msa The final multiple sequence alignment.
 * @param hdrs A vector of the original sequence headers.
 * @param consensus_seq The pre-computed consensus sequence of the MSA.
 * @param outdir The directory where the output files will be saved.
 */
void analyze_and_save_consensus(
    const std::vector<std::string>& msa,
    const std::vector<std::string>& hdrs,
    const std::string& consensus_seq,
    const std::string& outdir)
{
    if (msa.empty()) {
        return;
    }

    // Open the output file for the detailed report
    std::ofstream report_file(outdir + "/consensus_details.txt");
    if (!report_file) {
        std::cerr << "Warning: Could not open consensus_details.txt for writing." << std::endl;
        return;
    }

    report_file << "Consensus Sequence Match Report\n";
    report_file << "---------------------------------\n\n";

    int num_seqs = msa.size();
    int align_len = msa[0].size();

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

                // If the character in the sequence is not a gap, its original position counter increases
                if (seq_char != '-') {
                    original_positions[i]++;
                }

                // If this sequence's character matches the consensus character, record it
                if (seq_char == consensus_char) {
                    report_file << "  - Match: Seq '" << hdrs[i]
                                << "' at original position " << original_positions[i] << "\n";
                }
            }
            report_file << "\n"; // Add a newline for readability between columns
        } else {
            // If the consensus is a gap, we still need to advance the position counters
            // for any sequence that has a residue in this column.
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
    ScoreMode mode = MODE_DNA;
    ScoreFn   fn   = edna_score;
    int argi = 1;
    while (argi < argc && std::string(argv[argi]).rfind("--",0) == 0) {
        std::string opt = argv[argi++];
        if (opt == "--mode") {
            if (argi >= argc) {
                std::cerr<<"Error: --mode requires 'dna' or 'protein'.\n";
                return 1;
            }
            std::string m = argv[argi++];
            if      (m=="dna")     { mode = MODE_DNA;     fn = edna_score;   }
            else if (m=="protein") { mode = MODE_PROTEIN; fn = blosum62_score; }
            else {
                std::cerr<<"Error: unknown mode '"<<m<<"'. Use dna or protein.\n";
                return 1;
            }
        }
        else if (opt == "--gap_open") {
            if (argi >= argc) {
                std::cerr<<"Error: --gap_open requires a value.\n";
                return 1;
            }
            GAP_OPEN = std::stod(argv[argi++]);
        }
        else if (opt == "--gap_extend") {
            if (argi >= argc) {
                std::cerr<<"Error: --gap_extend requires a value.\n";
                return 1;
            }
            GAP_EXTEND = std::stod(argv[argi++]);
        }
        else {
            // unrecognized flag: step back and break
            --argi;
            break;
        }
    }

    // 2) Next arg is outdir
    if (argi >= argc) {
        std::cerr<<"Error: missing outdir.\n";
        return 1;
    }
    std::string outdir = argv[argi++];
    std::filesystem::create_directories(outdir);

    // 3) Remaining args are FASTA files
    if (argi >= argc) {
        std::cerr<<"Error: need at least one FASTA file.\n";
        return 1;
    }
    std::vector<std::string> files;
    for (; argi < argc; ++argi) {
        files.emplace_back(argv[argi]);
    }

    // 4) Read and Finalize Headers
    int n = files.size();
    std::vector<std::string> hdrs(n), seqs(n);
    for (int i = 0; i < n; ++i) {
        processFasta(files[i], hdrs[i], seqs[i]);

        // First, sanitize all whitespace to underscores
        sanitize_header(hdrs[i]);

        // Second, simplify the header to its final form
        if (mode == MODE_PROTEIN) {
            auto &h = hdrs[i];
            size_t p1 = h.find('|');
            size_t p2 = (p1==std::string::npos ? std::string::npos : h.find('|',p1+1));
            if (p1!=std::string::npos && p2!=std::string::npos)
                hdrs[i] = h.substr(p1+1, p2-p1-1);
        } else { // DNA mode
            auto &h = hdrs[i];
            // Since spaces are now underscores, we simplify by taking the part before the first underscore.
            size_t sp = h.find('_');
            if (sp != std::string::npos) {
                hdrs[i] = h.substr(0, sp);
            }
        }
    }


    // Find Optimal Gap Penalties before proceeding
    auto best_params = find_optimal_gap_penalties(seqs, hdrs, mode, fn);
    GAP_OPEN = best_params.gap_open;
    GAP_EXTEND = best_params.gap_extend;
    std::cout << "\nOptimal parameters found: GAP_OPEN=" << GAP_OPEN << ", GAP_EXTEND=" << GAP_EXTEND
              << " with score: " << best_params.score << "\n" << std::endl;

    // 5) Simplify headers
    for (int i = 0; i < n; ++i) {
        if (mode == MODE_PROTEIN) {
            auto &h = hdrs[i];
            size_t p1 = h.find('|');
            size_t p2 = (p1==std::string::npos ? std::string::npos : h.find('|',p1+1));
            if (p1!=std::string::npos && p2!=std::string::npos)
                hdrs[i] = h.substr(p1+1, p2-p1-1);
        } else {
            auto &h = hdrs[i];
            size_t sp = h.find_first_of(" \t");
            if (sp!=std::string::npos) hdrs[i] = h.substr(0, sp);
        }
    }

    // 6) Compute guide tree
    auto D    = computeDistanceMatrix(seqs, mode, fn);
    // Save the identity matrix to a file
    saveIdentityMatrix(D, hdrs, outdir);
    auto nwk  = buildUPGMATree(D, hdrs);

    // Format the Newick string for readability before saving
    std::string nwk_formatted = formatNewickString(nwk);

    std::ofstream tf(outdir+"/guide_tree.nwk");
    tf << nwk_formatted; // The formatted string already contains newlines
    tf.close();

    // 7) Progressive MSA by walking the UPGMA tree
    //    build a binary‐tree of Nodes from the Newick
    Node* root = parseNewick(nwk, hdrs);

    //    seed each leaf with its raw sequence
    std::queue<Node*> q;
    if (root) q.push(root); // Push only if the root is valid
    while (!q.empty()) {
        Node* u = q.front(); q.pop();
        if (!u) continue; // Skip if a null pointer somehow got on the queue

        if (u->leaf) {
            // Verify the index is valid before accessing the 'seqs' vector.
            if (u->seq_index < 0 || u->seq_index >= seqs.size()) {
                std::cerr << "\nFATAL LOGIC ERROR: Invalid sequence index " << u->seq_index
                          << " detected for a leaf node. Cannot retrieve sequence." << std::endl;
                exit(1); // Exit with an error
            }
            u->profile = { seqs[u->seq_index] };
        } else {
            // Push children only if they are not null
            if (u->left) q.push(u->left);
            if (u->right) q.push(u->right);
        }
    }

    // recursively build the full MSA
    auto msa = build_profile(root, mode, fn);

    // Refine the MSA using multiple threads over several rounds
    int total_rounds = 3;
    int iterations_per_thread_per_round = 10; // Total work = rounds * iterations * num_threads
    msa = refine_msa(msa, total_rounds, iterations_per_thread_per_round, mode, fn);

    // 10) Print and save colored MSA
    printColorMSA(msa);
    saveMSA_to_HTML(msa, hdrs, outdir);

    // 11) Analyze and Save Consensus Sequence
    // First, generate the final consensus sequence from the final MSA
    std::string final_consensus = generate_consensus(msa);
    // Second, write the simple consensus sequence to a FASTA file
    std::ofstream cf(outdir + "/consensus.fasta");
    cf << ">consensus\n" << final_consensus << "\n";
    cf.close();
    // Third, create the detailed report of positional matches
    analyze_and_save_consensus(msa, hdrs, final_consensus, outdir);

    // 12) Write out main MSA in FASTA format
    std::ofstream mf(outdir+"/msa.fasta");
    for (int i = 0; i < n; ++i) {
        mf << ">" << hdrs[i] << "\n"
           << msa[i]  << "\n";
    }
    mf.close();

    return 0;
}