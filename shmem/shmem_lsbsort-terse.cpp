// this is a version of shmem_lsbsort.cpp with comments, printing code,
// and verification code removed for source code size comparisons.
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cassert>
#include <cstdint>

#include <unistd.h>

#include <shmem.h>
#include <pcg_random.hpp>

#define RADIX 16
#define N_DIGITS (64/RADIX)
#define N_BUCKETS (1 << RADIX)
#define COUNTS_SIZE (N_BUCKETS)
#define MASK (N_BUCKETS - 1)
using counts_array_t = std::array<int64_t, COUNTS_SIZE>;

struct SortElement {
  uint64_t key = 0;
  uint64_t val = 0;
};
bool operator==(const SortElement& x, const SortElement& y) {
  return x.key == y.key && x.val == y.val;
}

struct CountBufElt {
  int32_t digit = 0;
  int32_t rank = 0;
  int64_t count = 0;
};

struct ShuffleBufSortElement {
  uint64_t key = 0;
  uint64_t val = 0;
  int64_t  dstGlobalIdx = 0;
};

static inline int64_t divCeil(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template<typename EltType>
struct DistributedArray {
  struct RankAndLocalIndex {
    int rank = 0;
    int64_t locIdx = 0;
  };

  std::string name_;
  EltType* localPart_ = nullptr;
  int64_t numElementsTotal_ = 0;
  int64_t numElementsPerRank_ = 0;
  int64_t numElementsHere_ = 0;
  int myRank_ = 0;
  int numRanks_ = 0;

  static DistributedArray<EltType>
  create(std::string name, int64_t totalNumElements);

  ~DistributedArray() {
    if (localPart_ != nullptr) {
      shmem_free(localPart_);
    }
  }

  inline int64_t localIdxToGlobalIdx(int64_t locIdx) const {
    return myRank_*numElementsPerRank_ + locIdx;
  }
  inline RankAndLocalIndex globalIdxToLocalIdx(int64_t glbIdx) const {
    RankAndLocalIndex ret;
    int64_t rank = glbIdx / numElementsPerRank_;
    int64_t locIdx = glbIdx - rank*numElementsPerRank_;
    ret.rank = rank;
    ret.locIdx = locIdx;
    return ret;
  }

  inline const std::string& name() const { return name_; }
  inline const EltType* localPart() const { return localPart_; }
  inline EltType* localPart() { return localPart_; }
  inline int64_t numElementsTotal() const { return numElementsTotal_; }
  inline int64_t numElementsPerRank() const { return numElementsPerRank_; }
  inline int64_t numElementsHere() const { return numElementsHere_; }
  inline int myRank() const { return myRank_; }
  inline int numRanks() const { return numRanks_; }
};

template<typename EltType>
DistributedArray<EltType>
DistributedArray<EltType>::create(std::string name, int64_t totalNumElements) {
  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  int64_t eltsPerRank = divCeil(totalNumElements, numRanks);
  int64_t eltsHere = eltsPerRank;
  if (eltsPerRank*myRank + eltsHere > totalNumElements) {
    eltsHere = totalNumElements - eltsPerRank*myRank;
  }
  if (eltsHere < 0) eltsHere = 0;

  DistributedArray<EltType> ret;
  ret.name_ = std::move(name);
  ret.localPart_ = (EltType*) shmem_malloc(eltsPerRank * sizeof(EltType));
  ret.numElementsTotal_ = totalNumElements;
  ret.numElementsPerRank_ = eltsPerRank;
  ret.numElementsHere_ = eltsHere;
  ret.myRank_ = myRank;
  ret.numRanks_ = numRanks;

  return ret;
}

inline int getBucket(SortElement x, int d) {
  return (x.key >> (RADIX*d)) & MASK;
}

void localCount(std::vector<SortElement>& A,
                std::vector<SortElement>& B,
                counts_array_t& starts,
                counts_array_t& counts,
                int digit,
                int64_t n) {
  assert(A.size() == B.size());

  starts.fill(0);
  counts.fill(0);

  for (int64_t i = 0; i < n; i++) {
    SortElement elt = A[i];
    counts[getBucket(elt, digit)] += 1;
  }

  {
    int64_t sum = 0;
    for (int i = 0; i < COUNTS_SIZE; i++) {
      starts[i] = sum;
      sum += counts[i];
    }
  }

  for (int64_t i = 0; i < n; i++) {
    SortElement elt = A[i];
    int64_t &next = starts[getBucket(elt, digit)];
    B[next] = elt;
    next += 1;
  }
}

void copyCountsToGlobalCounts(counts_array_t& localCounts,
                              DistributedArray<int64_t>& GlobalCounts) {
  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  for (int64_t i = 0; i < COUNTS_SIZE;) {
    int64_t dstGlobalIdx = i*numRanks + myRank;
    auto dst = GlobalCounts.globalIdxToLocalIdx(dstGlobalIdx);

    int dstRank = dst.rank;
    int nToSameRank = 0;
    while (i+nToSameRank < COUNTS_SIZE) {
      int64_t ii = (i+nToSameRank)*numRanks + myRank;
      int nextRank = GlobalCounts.globalIdxToLocalIdx(ii).rank;
      if (nextRank != dstRank) {
        break;
      }
      nToSameRank++;
    }
    assert(nToSameRank >= 1);
    assert(i + nToSameRank <= COUNTS_SIZE);

    int64_t* GCA = &GlobalCounts.localPart()[0];

    shmem_int64_iput(GCA + dst.locIdx, &localCounts[i], numRanks,
                     1, nToSameRank, dstRank);

    i += nToSameRank;
  }

  shmem_barrier_all();
}

void exclusiveScan(const DistributedArray<int64_t>& Src,
                   DistributedArray<int64_t>& Dst) {
  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  int64_t myTotal = 0;
  for (int64_t i = 0; i < Src.numElementsHere(); i++) {
    myTotal += Src.localPart()[i];
  }

  int64_t* PerRankStarts = (int64_t*) shmem_malloc(sizeof(int64_t) * numRanks);

  shmem_int64_p(PerRankStarts + myRank, myTotal, 0);

  shmem_barrier_all();

  if (myRank == 0) {
    int64_t sum = 0;
    for (int i = 0; i < numRanks; i++) {
      int64_t count = PerRankStarts[i];
      PerRankStarts[i] = sum;
      sum += count;
    }
    for (int i = 0; i < numRanks; i++) {
      shmem_int64_p(PerRankStarts + i, PerRankStarts[i], i);
    }
  }

  shmem_barrier_all();

  int64_t myGlobalStart = PerRankStarts[myRank];

  {
    int64_t sum = myGlobalStart;
    int64_t nHere = Dst.numElementsHere();
    for (int64_t i = 0; i < nHere; i++) {
      Dst.localPart()[i] = sum;
      sum += Src.localPart()[i];
    }
  }

  shmem_free(PerRankStarts);
}

void copyStartsFromGlobalStarts(DistributedArray<int64_t>& GlobalStarts,
                                counts_array_t& localStarts) {
  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  for (int64_t i = 0; i < COUNTS_SIZE;) {
    int64_t srcGlobalIdx = i*numRanks + myRank;
    auto src = GlobalStarts.globalIdxToLocalIdx(srcGlobalIdx);
    int srcRank = src.rank;
    int nToSameRank = 0;
    while (i+nToSameRank < COUNTS_SIZE) {
      int64_t ii = (i+nToSameRank)*numRanks + myRank;
      int nextRank = GlobalStarts.globalIdxToLocalIdx(ii).rank;
      if (nextRank != srcRank) {
        break;
      }
      nToSameRank++;
    }
    assert(nToSameRank >= 1);

    int64_t* GSA = GlobalStarts.localPart();

    shmem_int64_iget(&localStarts[i], GSA + src.locIdx,
                     1, numRanks, nToSameRank, srcRank);

    i += nToSameRank;
  }


  shmem_barrier_all();
}

void globalShuffle(DistributedArray<SortElement>& A,
                   DistributedArray<SortElement>& B,
                   int digit) {
  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  auto starts = std::make_unique<counts_array_t>();
  auto counts = std::make_unique<counts_array_t>();

  starts->fill(0);
  counts->fill(0);

  int64_t locN = A.numElementsHere();
  SortElement* localPart = A.localPart();
  for (int64_t i = 0; i < locN; i++) {
    SortElement elt = localPart[i];
    (*counts)[getBucket(elt, digit)] += 1;
  }

  auto GlobalCounts = DistributedArray<int64_t>::create("GlobalCounts",
                                                        COUNTS_SIZE*numRanks);
  auto GlobalStarts = DistributedArray<int64_t>::create("GlobalStarts",
                                                        COUNTS_SIZE*numRanks);

  copyCountsToGlobalCounts(*counts, GlobalCounts);

  exclusiveScan(GlobalCounts, GlobalStarts);

  copyStartsFromGlobalStarts(GlobalStarts, *starts);

  SortElement* GB = B.localPart();
  for (int64_t i = 0; i < locN; i++) {
    SortElement elt = localPart[i];
    int bucket = getBucket(elt, digit);
    int64_t &next = (*starts)[bucket];
    int64_t dstGlobalIdx = next;
    next += 1;

    auto dst = B.globalIdxToLocalIdx(dstGlobalIdx);
    assert(0 <= dst.rank && dst.rank < numRanks);
    shmem_putmem(GB + dst.locIdx, &elt, sizeof(SortElement), dst.rank);
  }
}

void mySort(DistributedArray<SortElement>& A,
            DistributedArray<SortElement>& B) {
  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  assert(N_DIGITS % 2 == 0);
  for (int digit = 0; digit < N_DIGITS; digit += 2) {
    globalShuffle(A, B, digit);
    globalShuffle(B, A, digit+1);
  }
}

int main(int argc, char *argv[]) {
  shmem_init();

  int64_t n = 100*1000*1000;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--n") {
      n = std::stoll(argv[++i]);
    }
  }

  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  if (myRank == 0) {
    std::cout << "Total number of shmem PEs: " << numRanks << "\n";
    std::cout << "Problem size: " << n << "\n";
  }

  auto A = DistributedArray<SortElement>::create("A", n);
  auto B = DistributedArray<SortElement>::create("B", n);
  std::vector<SortElement> LocalInputCopy;

  {
    auto start = std::chrono::steady_clock::now();
    if (myRank == 0) {
      std::cout << "Generating random values\n";
    }

    auto rng = pcg64(myRank);
    int64_t locN = A.numElementsHere();
    for (int64_t i = 0; i < locN; i++) {
      auto& elt = A.localPart()[i];
      elt.key = rng();
      elt.val = A.localIdxToGlobalIdx(i);
    }

    shmem_barrier_all();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (myRank == 0) {
      std::cout << "Generated random values in " << elapsed.count() << " s\n";
      flushOutput();
    }
    shmem_barrier_all();
  }

  {
    if (myRank == 0) {
      std::cout << "Sorting\n";
      flushOutput();
    }

    shmem_barrier_all();
    auto start = std::chrono::steady_clock::now();

    mySort(A, B);

    shmem_barrier_all();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (myRank == 0) {
      std::cout << "Sorted " << n << " values in " << elapsed.count() << "\n";;
      std::cout << "That's " << n/elapsed.count()/1000.0/1000.0
                << " M elements sorted / s\n";
      flushOutput();
    }
    shmem_barrier_all();
  }

  // this seems to cause crashes/hangs with openmpi shmem / osss-ucx
  //shmem_finalize();

  return 0;
}
