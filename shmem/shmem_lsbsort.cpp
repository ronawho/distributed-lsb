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

// Note: this code assumes that errors returned by MPI calls do not need to be
// handled. That's the case if they are set to halt the program or
// raise exeptions.

// TODO: put this back to 16 after it is working
#define RADIX 4
#define N_DIGITS (64/RADIX)
#define N_BUCKETS (1 << RADIX)
#define COUNTS_SIZE (N_BUCKETS + 1)
#define MASK (N_BUCKETS - 1)
using counts_array_t = std::array<int64_t, COUNTS_SIZE>;

// the elements to sort
struct SortElement {
  uint64_t key = 0; // to sort by
  uint64_t val = 0; // carried along
};
std::ostream& printhex(std::ostream& os, uint64_t key) {
  std::ios oldState(nullptr);
  oldState.copyfmt(os);
  os << std::setfill('0') << std::setw(16) << std::hex << key;
  os.copyfmt(oldState);
  return os;
}
std::ostream& operator<<(std::ostream& os, const SortElement& x) {
  os << "(";
  printhex(os, x.key);
  os << "," << x.val << ")";
  return os;
}
bool operator==(const SortElement& x, const SortElement& y) {
  return x.key == y.key && x.val == y.val;
}

// to help in communicating the counts
struct CountBufElt {
  int32_t digit = 0;
  int32_t rank = 0;
  int64_t count = 0;
};
std::ostream& operator<<(std::ostream& os, const CountBufElt& x) {
  os << "(" << x.digit << "," << x.rank << "," << x.count << ")";
  return os;
}

// to help in communicating the elements
struct ShuffleBufSortElement {
  uint64_t key = 0;
  uint64_t val = 0;
  int64_t  dstGlobalIdx = 0;
};
std::ostream& operator<<(std::ostream& os, const ShuffleBufSortElement& x) {
  os << "(";
  printhex(os, x.key);
  os << "," << x.val << "," << x.dstGlobalIdx << ")";
  return os;
}


// to help with sending sort elements to remote locales
// helper to divide while rounding up
static inline int64_t divCeil(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// Store a different type for distributed arrays just to make the code
// clearer.
// This actually just stores the current rank's portion of a distributed
// array along with some metadata.
// It doesn't support communication directly. Communication is expected
// to happen in the form of shmem calls working with localPart().
template<typename EltType>
struct DistributedArray {
  struct RankAndLocalIndex {
    int rank = 0;
    int64_t locIdx = 0;
  };

  std::string name_;
  EltType* localPart_ = nullptr;
  int64_t numElementsTotal_ = 0;    // number of elements on all ranks
  int64_t numElementsPerRank_ = 0 ; // number per rank
  int64_t numElementsHere_ = 0;     // number this rank
  int myRank_ = 0;
  int numRanks_ = 0;

  static DistributedArray<EltType>
  create(std::string name, int64_t totalNumElements);

  ~DistributedArray() {
    if (localPart_ != nullptr) {
      shmem_free(localPart_);
    }
  }

  // convert a local index to a global index
  inline int64_t localIdxToGlobalIdx(int64_t locIdx) const {
    return myRank_*numElementsPerRank_ + locIdx;
  }
  // convert a global index into a local index
  inline RankAndLocalIndex globalIdxToLocalIdx(int64_t glbIdx) const {
    RankAndLocalIndex ret;
    int64_t rank = glbIdx / numElementsPerRank_;
    int64_t locIdx = glbIdx - rank*numElementsPerRank_;
    ret.rank = rank;
    ret.locIdx = locIdx;
    return ret;
  }

  // accessors
  inline const std::string& name() const { return name_; }
  inline const EltType* localPart() const { return localPart_; }
  inline EltType* localPart() { return localPart_; }
  inline int64_t numElementsTotal() const { return numElementsTotal_; }
  inline int64_t numElementsPerRank() const { return numElementsPerRank_; }
  inline int64_t numElementsHere() const { return numElementsHere_; }
  inline int myRank() const { return myRank_; }
  inline int numRanks() const { return numRanks_; }

  // helper to print part of the distributed array
  void print(int64_t nToPrintPerRank) const;
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

static void flushOutput() {
  // this is a workaround to make it more likely that the output is printed
  // to the terminal in the correct order.
  // *it might not work*
  std::cout << std::flush;
  usleep(100);
}

template<typename EltType>
void DistributedArray<EltType>::print(int64_t nToPrintPerRank) const {
  shmem_barrier_all();

  if (myRank_ == 0) {
    if (nToPrintPerRank*numRanks_ >= numElementsTotal_) {
      std::cout << name_ << ": displaying all "
                << numElementsTotal_ << " elements\n";
    } else {
      std::cout << name_ << ": displaying first " << nToPrintPerRank
                << " elements on each rank"
                << " out of " << numElementsTotal_ << " elements\n";
    }
  }

  for (int rank = 0; rank < numRanks_; rank++) {
    if (myRank_ == rank) {
      int64_t i = 0;
      for (i = 0; i < nToPrintPerRank && i < numElementsHere_; i++) {
        int64_t glbIdx = localIdxToGlobalIdx(i);
        std::cout << name_ << "[" << glbIdx << "] = " << localPart_[i] << " (rank " << myRank_ << ")\n";
      }
      if (i < numElementsHere_) {
        std::cout << "...\n";
      }
      flushOutput();
    }
    shmem_barrier_all();
  }
}

// compute the bucket for a value when sort is on digit 'd'
inline int getBucket(SortElement x, int d) {
  return (x.key >> (RADIX*d)) & MASK;
}

// rearranges data according values at 'digit' & returns count information
//   A: contains the input data
//   B: after it runs, contains the rearranged data
//   starts: should not be used after it runs
//   counts: after it runs, contains the number of elements for each bucket
//   digit: the current digit for shuffling
void localCount(std::vector<SortElement>& A,
                  std::vector<SortElement>& B,
                  counts_array_t& starts,
                  counts_array_t& counts,
                  int digit,
                  int64_t n) {
  assert(A.size() == B.size());

  // clear out starts and counts
  starts.fill(0);
  counts.fill(0);

  // compute the count for each digit
  for (int64_t i = 0; i < n; i++) {
    SortElement elt = A[i];
    counts[getBucket(elt, digit)] += 1;
  }

  // compute the starts with an exclusive scan
  {
    int64_t sum = 0;
    for (int i = 0; i < COUNTS_SIZE; i++) {
      starts[i] = sum;
      sum += counts[i];
    }
  }

  // shuffle the data
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

  // Now, each rank has an array of counts, like this
  //  [r0d0, r0d1, ... r0d255]  | on rank 0
  //  [r1d0, r1d1, ... r1d255]  | on rank 1
  //  ...
  //

  // We need to transpose these so that the counts have the
  // starting digits first
  //  [r0d0, r1d0, r2d0, ...]   | on rank 0
  //  [r0d1, r1d1, r2d1, ...]   |
  //  [r0d2, r1d2, r2d2, ...]   | on rank 1 ...
  //  ...

  for (int64_t i = 0; i < COUNTS_SIZE;) {
    // compute the number of elements that go to a particular destination rank
    int64_t dstGlobalIdx = i*numRanks + myRank;
    auto dst = GlobalCounts.globalIdxToLocalIdx(dstGlobalIdx);

    /*
    int64_t* GCA = &GlobalCounts.localPart()[0]; // it's symmetric
    shmem_int64_p(GCA + dst.locIdx, localCounts[i], dst.rank);

    i++;
    */

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

    int64_t* GCA = &GlobalCounts.localPart()[0]; // it's symmetric

    shmem_int64_iput( GCA + dst.locIdx, // region on the remote PE
                      &localCounts[i],  // region on the local PE
                      numRanks,         // stride for destination array
                      1,                // stride for source array
                      nToSameRank,      // number of elements
                      dstRank           // destination rank
                    );

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

  // Now compute the total of for each chunk of the global src array
  int64_t myTotal = 0;
  for (int64_t i = 0; i < Src.numElementsHere(); i++) {
    myTotal += Src.localPart()[i];
  }

  // allocate a remotely accessible array
  // only rank 0's values will be used
  int64_t* PerRankStarts = (int64_t*) shmem_malloc(sizeof(int64_t) * numRanks);

  // Send the total from each rank to rank 0
  shmem_int64_p(PerRankStarts + myRank, myTotal, 0);

  // wait for rank 0 to get all of them
  shmem_barrier_all();

  if (myRank == 0) {
    // change from counts per rank to starts per rank
    int64_t sum = 0;
    for (int i = 0; i < numRanks; i++) {
      int64_t count = PerRankStarts[i];
      PerRankStarts[i] = sum;
      sum += count;
    }
    // send the start for each rank to that rank
    for (int i = 0; i < numRanks; i++) {
      shmem_int64_p(PerRankStarts + i, PerRankStarts[i], i);
    }
  }

  // wait for rank 0 to dissemenate starts
  shmem_barrier_all();

  int64_t myGlobalStart = PerRankStarts[myRank];

  // scan the region in each rank
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

  // starts look like this:
  //  [r0d0, r1d0, r2d0, ...]   | on rank 0
  //  [r0d1, r1d1, r2d1, ...]   |
  //  [r0d2, r1d2, r2d2, ...]   | on rank 1 ...
  //  ...

  // Need to get the values for each rank so it's like this:
  //  [r0d0, r0d1, ... r0d255]  | on rank 0
  //  [r1d0, r1d1, ... r1d255]  | on rank 1
  //  ...
  //

  for (int64_t i = 0; i < COUNTS_SIZE;) {
    // compute the number of elements that come from a particular src rank
    int64_t srcGlobalIdx = i*numRanks + myRank;
    auto src = GlobalStarts.globalIdxToLocalIdx(srcGlobalIdx);
    int srcRank = src.rank;
    int nToSameRank = 0;
    while (i+nToSameRank < COUNTS_SIZE) {
      int64_t ii = i*numRanks + myRank;
      int nextRank = GlobalStarts.globalIdxToLocalIdx(ii).rank;
      if (nextRank != srcRank) {
        break;
      }
      nToSameRank++;
    }
    assert(nToSameRank >= 1);

    int64_t* GSA = GlobalStarts.localPart(); // it's symmetric

    shmem_int64_iget(  &localStarts[i],  // region on the local PE
                       GSA + src.locIdx, // region on the remote PE
                       1,                // stride for destination array
                       numRanks,         // stride for source array
                       nToSameRank,      // number of elements
                       srcRank           // source rank
                    );

    i += nToSameRank;
  }


  shmem_barrier_all();
}

// shuffles the data from A into B
void globalShuffle(DistributedArray<SortElement>& A,
                   DistributedArray<SortElement>& B,
                   int digit) {
  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  auto starts = std::make_unique<counts_array_t>();
  auto counts = std::make_unique<counts_array_t>();

  // clear out starts and counts
  starts->fill(0);
  counts->fill(0);

  // compute the count for each digit
  int64_t locN = A.numElementsHere();
  SortElement* localPart = A.localPart();
  for (int64_t i = 0; i < locN; i++) {
    SortElement elt = localPart[i];
    (*counts)[getBucket(elt, digit)] += 1;
  }

  // Now, each rank has an array of counts, like this
  //  [r0d0, r0d1, ... r0d255]  | on rank 0
  //  [r1d0, r1d1, ... r1d255]  | on rank 1
  //  ...
  //

  // We need to transpose these so that the counts have the
  // starting digits first
  //  [r0d0, r1d0, r2d0, ...]   | on rank 0
  //  [r0d1, r1d1, r2d1, ...]   |
  //  [r0d2, r1d2, r2d2, ...]   | on rank 1 ...
  //  ...

  // create a distributed array storing the result of this transposition
  auto GlobalCounts = DistributedArray<int64_t>::create("GlobalCounts",
                                                        COUNTS_SIZE*numRanks);
  // and one storing the start positions for each task
  // (that will be the result of a scan operation)
  auto GlobalStarts = DistributedArray<int64_t>::create("GlobalStarts",
                                                        COUNTS_SIZE*numRanks);

  // copy the per-bucket counts to the global counts array
  copyCountsToGlobalCounts(*counts, GlobalCounts);

  if (myRank == 0) std::cout << "GlobalCounts\n";
  GlobalCounts.print(100);

  // scan to fill in GlobalStarts
  exclusiveScan(GlobalCounts, GlobalStarts);

  // copy the per-bucket starts from the global counts array
  copyStartsFromGlobalStarts(GlobalStarts, *starts);

  if (myRank == 0) std::cout << "GlobalStarts\n";
  GlobalStarts.print(100);

  // Now go through the data in B assigning each element its final
  // position and sending that data to the other ranks
  // Leave the result in B
  SortElement* GB = B.localPart(); // it's symmetric
  for (int64_t i = 0; i < locN; i++) {
    SortElement elt = localPart[i];
    int bucket = getBucket(elt, digit);
    int64_t &next = (*starts)[bucket];
    int64_t dstGlobalIdx = next;
    next += 1;

    // store 'elt' into 'dstGlobalIdx'
    auto dst = B.globalIdxToLocalIdx(dstGlobalIdx);
    assert(0 <= dst.rank && dst.rank < numRanks);
    shmem_putmem(GB + dst.locIdx, &elt, sizeof(SortElement), dst.rank);
  }
}

// Sort the data in A, using B as scratch space.
void mySort(DistributedArray<SortElement>& A,
            DistributedArray<SortElement>& B) {
  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  assert(N_DIGITS % 2 == 0);
  for (int digit = 0; digit < N_DIGITS; digit += 2) {
    if (myRank == 0) std::cout << "Digit " << digit << " input:\n";
    A.print(10);
    globalShuffle(A, B, digit);
    if (myRank == 0) std::cout << "Digit " << digit << " output:\n";
    B.print(10);
    globalShuffle(B, A, digit+1);
  }
}

int main(int argc, char *argv[]) {
  shmem_init();

  // read in the problem size
  bool printSome = false;
  bool verifyLocally = false;
  bool verifyLocallySet = false;
  int64_t n = 100*1000*1000;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--n") {
      n = std::stoll(argv[++i]);
    } else if (std::string(argv[i]) == "--print") {
      printSome = true;
    } else if (std::string(argv[i]) == "--verify") {
      verifyLocally = true;
      verifyLocallySet = true;
    } else if (std::string(argv[i]) == "--no-verify") {
      verifyLocally = false;
      verifyLocallySet = true;
    }
  }

  if (!verifyLocallySet) {
    verifyLocally = (n < 128*1024*1024);
  }

  int myRank = 0;
  int numRanks = 0;
  myRank = shmem_my_pe();
  numRanks = shmem_n_pes();

  if (myRank == 0) {
    std::cout << "Total number of shmem PEs: " << numRanks << "\n";
    std::cout << "Problem size: " << n << "\n";
    flushOutput();
  }

  // create distributed arrays A and B
  auto A = DistributedArray<SortElement>::create("A", n);
  auto B = DistributedArray<SortElement>::create("B", n);
  std::vector<SortElement> LocalInputCopy;

  // set the keys to random values and the values to global indices
  {
    auto start = std::chrono::steady_clock::now();
    if (myRank == 0) {
      std::cout << "Generating random values\n";
      flushOutput();
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

  // Print out the first few elements on each locale
  if (printSome) {
    A.print(10);
  }

  /*
  if (verifyLocally) {
    // save the input to the sorting problem
    LocalInputCopy.resize(numRanks*A.numElementsPerRank());
    MPI_Gather(& A.localPart()[0], A.numElementsPerRank(), gSortElementMpiType,
               & LocalInputCopy[0], A.numElementsPerRank(),
               gSortElementMpiType, 0, MPI_COMM_WORLD);
  }*/

  // Shuffle the data in-place to sort by the current digit
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

  // Print out the first few elements on each locale
  if (printSome) {
    A.print(10);
  }

  /*
  if (verifyLocally) {
    // gather the output from sorting
    std::vector<SortElement> LocalOutputCopy;
    LocalOutputCopy.resize(numRanks*A.numElementsPerRank());
    MPI_Gather(& A.localPart()[0], A.numElementsPerRank(), gSortElementMpiType,
               & LocalOutputCopy[0], A.numElementsPerRank(),
               gSortElementMpiType, 0, MPI_COMM_WORLD);

    if (myRank == 0) {
      auto LocalSorted = LocalInputCopy;
      std::stable_sort(LocalSorted.begin(), LocalSorted.begin() + n,
                       [](const SortElement& a, const SortElement& b) {
                           return a.key < b.key;
                       });

      bool failures = false;
      for (int64_t i = 0; i < n; i++) {
        if (! (LocalSorted[i] == LocalOutputCopy[i])) {
          std::cout << "Sorted element " << i << " did not match\n";
          std::cout << "Expected: " << LocalSorted[i] << "\n";
          std::cout << "Got:      " << LocalOutputCopy[i] << "\n";
          failures = true;
        }
      }
      assert(!failures);
    }
  }*/

  return 0;
}
