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
#include <limits>
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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " [--mode dna|protein] [--gap_open O] [--gap_extend E] "
                     "outdir file1.fasta file2.fasta [...]\n";
        return 1;
    }

    // defaults
    ScoreMode mode = MODE_DNA;
    ScoreFn   fn   = edna_score;
    // GAP_OPEN and GAP_EXTEND are globals, defaulted above

    int argi = 1;
    // parse all leading “--” options
    while (argi < argc && std::string(argv[argi]).rfind("--", 0) == 0) {
        std::string opt = argv[argi++];
        if (opt == "--mode") {
            if (argi >= argc) {
                std::cerr<<"Error: --mode requires 'dna' or 'protein'.\n";
                return 1;
            }
            std::string m = argv[argi++];
            if (m == "dna")       { mode = MODE_DNA;     fn = edna_score; }
            else if (m == "protein") { mode = MODE_PROTEIN; fn = blosum62_score; }
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
            // not one of our flags → back up and break
            --argi;
            break;
        }
    }

    // next argument must be outdir
    if (argi >= argc) {
        std::cerr<<"Error: missing outdir.\n";
        return 1;
    }
    std::string outdir = argv[argi++];
    std::filesystem::create_directories(outdir);

    // remaining args are FASTA files
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

    for (int i = 0; i < n; ++i) {
        if (mode == MODE_PROTEIN) {
            // protein: header looks like "sp|P01308|INS_HUMAN …"
            auto &h = hdrs[i];
            size_t p1 = h.find('|');
            size_t p2 = (p1 == std::string::npos)
                        ? std::string::npos
                        : h.find('|', p1 + 1);
            if (p1 != std::string::npos && p2 != std::string::npos) {
                hdrs[i] = h.substr(p1 + 1, p2 - p1 - 1);
            }
        }
        else {
            // DNA: just take up to first whitespace
            auto &h = hdrs[i];
            size_t pos = h.find_first_of(" \t");
            if (pos != std::string::npos) {
                hdrs[i] = h.substr(0, pos);
            }
        }
    }

    // 5) Compute guide tree (identity distance)
    auto D = computeDistanceMatrix(seqs, mode, fn);
    auto tree = buildUPGMATree(D, hdrs);
    std::ofstream tf(outdir + "/guide_tree.nwk");
    tf << tree << "\n";
    tf.close();

    // 6) Do star‐alignment MSA
    auto msa = msa_star(hdrs, seqs, mode, fn);
    size_t L = 0;
    for (auto &s : msa) {
        L = std::max(L, s.size());
    }
    // pad shorter ones with gaps:
    for (auto &s : msa) {
        if (s.size() < L) {
            s.append(L - s.size(), '-');
        }
    }

    // 7) Print color to stdout
    long long msa_score = 0;
    int m = msa.size(); L = msa[0].size();
    for (int j = 0; j < L; ++j) {
        for (int i = 0; i < m; ++i) {
            for (int k = i+1; k < m; ++k) {
                char a = msa[i][j], b = msa[k][j];
                if (a!='-' && b!='-')
                    msa_score += score(a, b, mode);
                else if ((a=='-') ^ (b=='-'))
                    msa_score += int(GAP_OPEN);
            }
        }
    }
    std::cout << "\nMSA sum-of-pairs score: " << msa_score << "\n\n";

    double avg_per_col  = double(msa_score) / L;
    double avg_per_pair = double(msa_score)
                        / ( L * (m*(m-1)/2) );
    std::cout<<"avg per column = "<<avg_per_col
             <<", avg per pair = "<<avg_per_pair<<"\n\n";

    printColorMSA(msa);

    // 8) Write MSA to file
    std::ofstream mf(outdir + "/msa.fasta");
    for (int i = 0; i < n; ++i) {
        mf << ">" << hdrs[i] << "\n"
           << msa[i] << "\n";
    }
    mf.close();

    return 0;
}
