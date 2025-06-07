//
// Created by abhinavmishra on 7/6/25.
//
/*
 * @file msa_full.cpp
 * @brief Standalone progressive MSA with guide-tree (UPGMA), color output, and file saving.
 * @author ChatGPT
 * @date 2025-06-07
 *
 * Usage: ./msa_full outdir file1.fasta file2.fasta [...]
 */
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
#include "EDNAFULL.h"
#include "EBLOSUM62.h"

// ANSI color codes
#define RESET "\033[0m"
#define GREEN "\033[32m"
#define RED   "\033[31m"
#define CYAN  "\033[36m"

// Gap penalties
// instead of “static const”, use mutable globals:
double GAP_OPEN   = -5.0;
double GAP_EXTEND = -1.0;
static const int LINE_WIDTH = 80;

enum ScoreMode { MODE_DNA, MODE_PROTEIN };
using ScoreFn = int(*)(char,char);

// DNA / EDNAFULL lookup
static const std::array<uint8_t,256> char2idx = [](){
    std::array<uint8_t,256> m; m.fill(255);
    m['A']=0; m['C']=1; m['G']=2; m['T']=3; m['U']=3;
    m['R']=4; m['Y']=5; m['S']=6; m['W']=7;
    m['K']=8; m['M']=9; m['B']=10; m['D']=11;
    m['H']=12; m['V']=13; m['N']=14; m['X']=14;
    return m;
}();

// Protein / BLOSUM62 lookup
static const std::array<uint8_t,256> prot_idx = [](){
    std::array<uint8_t,256> m; m.fill(255);
    const char* AA="ARNDCQEGHILKMFPSTWYVBZX*";
    for(int i=0; AA[i]; ++i) m[(uint8_t)AA[i]] = i;
    return m;
}();

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

inline int edna_score(char x, char y) {
    x = static_cast<char>(std::toupper((unsigned char)x));
    y = static_cast<char>(std::toupper((unsigned char)y));

    uint8_t ix = char2idx[static_cast<uint8_t>(x)];
    if (ix == 255) ix = char2idx['N'];         // unknown → N
    uint8_t iy = char2idx[static_cast<uint8_t>(y)];
    if (iy == 255) iy = char2idx['N'];

    return static_cast<int>(std::round(EDNAFULL_matrix[ix][iy]));
}


inline int score(char x,char y,ScoreMode mode){
    return mode==MODE_DNA ? edna_score(x,y) : blosum62_score(x,y);
}

/** in-memory global alignment **/
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
    const __m256i vNegInf    = _mm256_set1_epi32(INT_MIN/2);

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
                    char ptr = 'M';
                    if (Sblock[k] == Eblock[k]) ptr = (Eblock[k] == (Ec[j+k] - GAP_EXTEND) ? 'e' : 'E');
                    else if (Sblock[k] == Fblock[k]) ptr = (Fblock[k] == (Fp[j+k] - GAP_EXTEND) ? 'f' : 'F');
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

/** read first FASTA record, uppercase & strip non‐letters **/
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


/** star‐alignment projection **/
void projectGaps(const std::string &oldc, const std::string &newc, std::vector<std::string> &seqs){
    size_t idx_old=0;
    for(size_t j=0;j<newc.size();++j){
        if(newc[j]=='-'){
            for(auto &s:seqs) s.insert(s.begin()+j,'-');
        } else {
            ++idx_old;
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
                bool isGap = (c == '-');
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


/** star MSA **/
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
 * @brief Parses a Newick format string into a binary tree of Nodes.
 * This version is more robust and correctly handles names, delimiters, and branch lengths.
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
 * @brief Recursively builds a multiple sequence alignment by walking the guide tree.
 * * This corrected version handles leaf nodes, unary nodes (one child), and binary
 * nodes (two children) to prevent segmentation faults.
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
    // Pick a "representative" string from each profile to align.
    std::string consA = A[0];
    std::string consB = B[0];

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

    // 4) Read sequences
    int n = files.size();
    std::vector<std::string> hdrs(n), seqs(n);
    for (int i = 0; i < n; ++i) {
        processFasta(files[i], hdrs[i], seqs[i]);
    }

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
    auto nwk  = buildUPGMATree(D, hdrs);
    std::ofstream tf(outdir+"/guide_tree.nwk");
    tf << nwk << "\n";
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
            // --- ADDED SAFETY CHECK ---
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

    //    recursively build the full MSA
    auto msa = build_profile(root, mode, fn);

    // 8) Pad to uniform length
    size_t L = 0;
    for (auto &s : msa) L = std::max(L, s.size());
    for (auto &s : msa) if (s.size()<L) s.append(L - s.size(), '-');

    // 9) Compute sum‐of‐pairs score
    long long msa_score = 0;
    int m = msa.size();
    for (size_t j = 0; j < L; ++j) {
        for (int i = 0; i < m; ++i) {
            for (int k = i+1; k < m; ++k) {
                char a = msa[i][j], b = msa[k][j];
                if (a!='-' && b!='-')
                    msa_score += score(a,b,mode);
                else if ((a=='-') ^ (b=='-'))
                    msa_score += (long long)GAP_OPEN;
            }
        }
    }
    std::cout << "\nMSA sum-of-pairs score: " << msa_score << "\n";
    double avg_col  = double(msa_score)/L;
    double avg_pair = double(msa_score)/(L * (m*(m-1)/2));
    std::cout << "avg per column = " << avg_col
              << ", avg per pair = "  << avg_pair << "\n\n";

    // 10) Print colored MSA
    printColorMSA(msa);

    // 11) Write out FASTA
    std::ofstream mf(outdir+"/msa.fasta");
    for (int i = 0; i < n; ++i) {
        mf << ">" << hdrs[i] << "\n"
           << msa[i]  << "\n";
    }
    mf.close();

    return 0;
}