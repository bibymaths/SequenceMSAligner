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
#include <cstring>
#include <climits>
#include <array>
#include <set>
#include <immintrin.h>
#include "EDNAFULL.h"
#include "EBLOSUM62.h"

// ANSI color codes
#define RESET "\033[0m"
#define GREEN "\033[32m"
#define RED   "\033[31m"
#define CYAN  "\033[36m"

// Gap penalties
static const double GAP_OPEN   = -5.0;
static const double GAP_EXTEND = -1.0;
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
#include <immintrin.h>

// …

int computeGlobalAlignment(const std::string &x,
                           const std::string &y,
                           ScoreMode mode,
                           ScoreFn score_fn,
                           std::string &aligned_x,
                           std::string &aligned_y)
{
    const int m = x.size();
    const int n = y.size();

    // Round n up to a multiple of 16
    int n16 = (n + 15) & ~15;

    // DP arrays, padded to n16+1
    std::vector<int> S_prev(n16+1), S_cur(n16+1);
    std::vector<int> E_cur(n16+1), F_prev(n16+1);
    std::vector<char> trace((m+1)*(n+1));

    // broadcast gap penalties
    const __m512i vGapOpen512   = _mm512_set1_epi32(int(GAP_OPEN));
    const __m512i vGapExtend512 = _mm512_set1_epi32(int(GAP_EXTEND));
    const __m512i vNegInf512    = _mm512_set1_epi32(INT_MIN/2);

    // initialize row 0 (scalar)
    S_prev[0] = 0;
    for(int j=1; j<=n; ++j) {
        int e = (j==1 ? S_prev[j-1] + GAP_OPEN : E_cur[j-1] + GAP_EXTEND);
        S_prev[j] = e;
        E_cur[j]  = e;
        F_prev[j] = INT_MIN/2;
        trace[j]  = (j==1 ? 'E' : 'e');
    }
    // pad the rest
    for(int j=n+1; j<=n16; ++j) {
        S_prev[j] = E_cur[j] = F_prev[j] = INT_MIN/2;
    }

    // raw pointers for intrinsics
    int *Sp = S_prev.data();
    int *Sc = S_cur.data();
    int *Ec = E_cur.data();
    int *Fp = F_prev.data();

    for(int i=1; i<=m; ++i) {
        // scalar col 0
        int oF = Sp[0] + GAP_OPEN;
        int eF = Fp[0] + GAP_EXTEND;
        Sc[0] = std::max(oF, eF);
        Ec[0] = INT_MIN/2;
        Fp[0] = Sc[0];
        trace[i*(n+1)+0] = (Sc[0]==oF ? 'F' : 'f');

        // prepare “diagonal” broadcast
        __m512i vSdiag = _mm512_set1_epi32(Sp[0]);

        // process 16 columns per iteration
        for(int j=1; j<=n16; j+=16) {
            // load vectors
            __m512i vSp  = _mm512_loadu_si512((__m512i*)&Sp[j]);
            __m512i vFp  = _mm512_loadu_si512((__m512i*)&Fp[j]);
            __m512i vEcm1= _mm512_loadu_si512((__m512i*)&Ec[j-1]); // E_cur[j-1..]
            __m512i vScm1= _mm512_loadu_si512((__m512i*)&Sc[j-1]); // S_cur[j-1..]

            // F = max( S_prev + open, F_prev + extend )
            __m512i vF = _mm512_max_epi32(
                _mm512_add_epi32(vSp, vGapOpen512),
                _mm512_add_epi32(vFp, vGapExtend512)
            );
            // E = max( S_cur[j-1] + open, E_cur[j-1] + extend )
            __m512i vE = _mm512_max_epi32(
                _mm512_add_epi32(vScm1, vGapOpen512),
                _mm512_add_epi32(vEcm1, vGapExtend512)
            );
            // BestPrev = max( S_prev[j-1], F_prev[j-1], E_prev[j-1] )
            __m512i vFp_m1 = _mm512_loadu_si512((__m512i*)&Fp[j-1]);
            __m512i vEp_m1 = _mm512_loadu_si512((__m512i*)&Ec[j-1]);
            __m512i vBestPrev = _mm512_max_epi32(
                _mm512_max_epi32(vSdiag, vFp_m1),
                vEp_m1
            );
            // build match/mismatch vector
            int scbuf[16];
            for(int k=0; k<16; ++k) {
                int idx = j + k - 1;
                scbuf[k] = (idx < n ? score_fn(x[i-1], y[idx]) : 0);
            }
            __m512i vMatch = _mm512_loadu_si512((__m512i*)scbuf);

            // M = BestPrev + match
            __m512i vM = _mm512_add_epi32(vBestPrev, vMatch);
            // S = max( M, E, F )
            __m512i vS = _mm512_max_epi32(
                _mm512_max_epi32(vM, vE),
                vF
            );

            // store DP vectors
            _mm512_storeu_si512((__m512i*)&Sc[j], vS);
            _mm512_storeu_si512((__m512i*)&Fp[j], vF);
            _mm512_storeu_si512((__m512i*)&Ec[j], vE);

            // advance diagonal for next block
            vSdiag = vSp;

            // scalar traceback pointer for these 16
            int Sblk[16], Eblk[16], Fblk[16];
            _mm512_storeu_si512((__m512i*)Sblk, vS);
            _mm512_storeu_si512((__m512i*)Eblk, vE);
            _mm512_storeu_si512((__m512i*)Fblk, vF);
            for(int k=0; k<16; ++k) {
                int col = j + k;
                if (col <= n) {
                    char ptr = 'M';
                    if      (Sblk[k] == Eblk[k]) ptr = (Eblk[k] == (Ec[col] - GAP_EXTEND) ? 'e' : 'E');
                    else if (Sblk[k] == Fblk[k]) ptr = (Fblk[k] == (Fp[col] - GAP_EXTEND) ? 'f' : 'F');
                    trace[i*(n+1) + col] = ptr;
                }
            }
        }

        // swap pointers for next row
        std::swap(Sp, Sc);
        std::swap(Fp, Ec);
    }

    int finalScore = Sp[n];

    // scalar traceback (unchanged)
    aligned_x.clear();
    aligned_y.clear();
    int i = m, j = n;
    while (i>0 || j>0) {
        char p = trace[i*(n+1) + j];
        if      (p=='M') { aligned_x.push_back(x[i-1]); aligned_y.push_back(y[j-1]); --i; --j; }
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

/** compute identity-based distance matrix **/
std::vector<std::vector<double>> computeDistanceMatrix(const std::vector<std::string> &seqs, ScoreMode mode, ScoreFn fn){
    size_t n=seqs.size();
    std::vector<std::vector<double>> D(n, std::vector<double>(n,0.0));
    for(size_t i=0;i<n;++i) for(size_t j=i+1;j<n;++j){
        std::string ax, ay; computeGlobalAlignment(seqs[i],seqs[j],mode,fn,ax,ay);
        int L=ax.size(), match=0;
        for(int k=0;k<L;++k) if(ax[k]==ay[k]&&ax[k]!='-') ++match;
        double id = L? double(match)/L : 0.0;
        D[i][j]=D[j][i]=1.0 - id;
    }
    return D;
}

/** build UPGMA guide tree **/
std::string buildUPGMATree(std::vector<std::vector<double>> D, const std::vector<std::string> &names){
    int n=names.size();
    struct Cl{ std::string nwk; int size; };
    std::vector<Cl> C(n);
    std::vector<bool> alive(n,true);
    for(int i=0;i<n;++i){ C[i]={names[i],1}; }
    int aliveCount=n;
    while(aliveCount>1){
        // find min dist
        double best=std::numeric_limits<double>::infinity(); int a=-1,b=-1;
        for(int i=0;i<n;++i) if(alive[i]) for(int j=i+1;j<n;++j) if(alive[j]){
            if(D[i][j]<best){ best=D[i][j]; a=i; b=j; }
        }
        double h = best/2.0;
        // newick
        std::ostringstream oss;
        oss<<"("<<C[a].nwk<<":"<<h<<","<<C[b].nwk<<":"<<h<<")";
        C[a].nwk = oss.str(); C[a].size += C[b].size;
        // merge distances
        for(int k=0;k<n;++k) if(k!=a&&alive[k]){
            double d = (D[a][k]*C[a].size + D[b][k]*C[b].size)/(C[a].size+C[b].size);
            D[a][k]=D[k][a]=d;
        }
        alive[b]=false; --aliveCount;
    }
    // root cluster
    std::string tree;
    for(int i=0;i<n;++i) if(alive[i]) tree = C[i].nwk + ";";
    return tree;
}

/** print MSA with block positions rather than headers **/
void printColorMSA(const std::vector<std::string> &aln) {
    int m = aln.size();
    int L = aln[0].size();

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
                  << " [--mode dna|protein] outdir file1.fasta file2.fasta [...]\n";
        return 1;
    }

    // 1) Parse --mode
    ScoreMode mode = MODE_DNA;
    ScoreFn fn = edna_score;
    int argi = 1;
    if (std::string(argv[argi]) == "--mode") {
        if (argi + 1 >= argc) {
            std::cerr << "Error: --mode requires 'dna' or 'protein'.\n";
            return 1;
        }
        std::string m = argv[argi+1];
        if (m == "dna") {
            mode = MODE_DNA;
            fn = edna_score;
        }
        else if (m == "protein") {
            mode = MODE_PROTEIN;
            fn = blosum62_score;
        }
        else {
            std::cerr << "Error: unknown mode '" << m << "'. Use dna or protein.\n";
            return 1;
        }
        argi += 2;
    }

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
    for ( ; argi < argc; ++argi) {
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

    // 7) Print color to stdout
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