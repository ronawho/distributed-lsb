#include <cassert>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

#include <mpi.h>
#include <pcg_random.hpp>

// Note: this code assumes that errors returned by MPI calls do not need to be
// handled. That's the case if they are set to halt the program or
// raise exeptions.

#define RADIX 16
#define N_BUCKETS (1 << RADIX)
#define COUNTS_SIZE (N_BUCKETS + 1)
#define MASK (N_BUCKETS - 1)
using counts_array_t = std::array<int64_t, COUNTS_SIZE>;

struct SortElement {
  uint64_t key; // to sort by
  uint64_t val; // carried along
};
static MPI_Datatype gSortElementMpiType;

struct CountBufElt {
  int32_t digit;
  int32_t rank;
  int64_t count;
};
static MPI_Datatype gCountBufEltMpiType;

static int64_t divCeil(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// compute the bucket for a value when sort is on digit 'd'
inline int getBucket(SortElement x, int d) {
  return (x.key >> (RADIX*d)) & MASK;
}

// rearranges data according values at 'digit' & returns count information
//   A: contains the input data
//   B: after it runs, contains the rearranged data
//   starts: after it runs, contains the start index for each bucket
//   counts: after it runs, contains the number of elements for each bucket
//   digit: the current digit for shuffling
void localShuffle(std::vector<SortElement>& A,
                  std::vector<SortElement>& B,
                  counts_array_t& starts,
                  counts_array_t& counts,
                  int digit) {
  assert(A.size() == B.size());

  // clear out starts and counts
  starts.fill(0);
  counts.fill(0);

  // compute the count for each digit
  for (SortElement elt : A) {
    counts[getBucket(elt)] += 1;
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
  for (SortElement elt : A) {
    int64_t &next = starts[getBucket(elt)];
    B[next] = elt;
    next += 1;
  }

  // recompute the starts array for returning it
  int64_t sum = 0;
  for (int i = 0; i < COUNTS_SIZE; i++) {
    starts[i] = sum;
    sum += counts[i];
  }
}

void globalShuffle(std::vector<SortElement>& A,
                   std::vector<SortElement>& B,
                   int digit) {
  int myRank = 0;
  int numRanks = 0;
  MPI_Comm_rank (MPI_COMM_WORLD, &myRank);
  MPI_Comm_size (MPI_COMM_WORLD, &numRanks);
 
  auto starts = std::make_unique<counts_array_t>();
  auto counts = std::make_unique<counts_array_t>();

  localShuffle(A, B, *starts, *counts, n, digit);

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

  // Conceptually, create a distributed array storing this transposition.
  int64_t globCountsPerNode = divCeil(COUNTS_SIZE*numRanks, numRanks);
  std::vector<int64_t> GlobalCounts;
  GlobalCounts.resize(globCountsPerNode);

  // And a distributed array storing the CountBufElt type.
  std::vector<CountBufElt> recvBuf;
  recvBuf.resize(globCountsPerNode);

  // Prepare for MPI_AllToAllv
  std::vector<int> sendCounts, sendDisplacements, recvCounts, recvDisplacements;
  sendCounts.resize(numRanks);
  sendDisplacements.resize(numRanks);
  recvCounts.resize(numRanks);
  recvDisplacements.resize(numRanks);
  std::vector<CountBufElt> sendBuf;
  sendBuf.resize(COUNTS_SIZE);
  int rankCount = 0;
  int rankStart = 0;
  int lastRank = -1;
  for (int i = 0; i < COUNTS_SIZE; i++) {
    int dstRank = i / globCountsPerNode;
    sendCounts[dstRank]++;
    CountBufElt c;
    c.digit = i;
    c.rank = myRank;
    c.count = counts[i];
    sendBuf[i] = c;
  }
  // compute sendDisplacements
  {
    int sum = 0;
    for (int i = 0; i < numRanks; i++) {
      sendDisplacements[i] = sum;
      sum += sendCounts[i];
    }
  }

  // Dissemenate the counts to send to each node
  // for use in recvCounts
  {
    std::vector<int> tmpDspl, tmpCounts;
    tmpDspl.resize(numRanks);
    tmpCounts.resize(numRanks);
    for (int i = 0; i < numRanks; i++) {
      tmpDspl[i] = i;
      tmpCounts[i] = 1;
    }
    MPI_Alltoallv(/*sendbuf*/ &sendCounts[0],
                  /*sendcounts*/ &tmpCounts[0],
                  /*sdispls*/ &tmpDspl[0],
                  MPI_INT,
                  /*recvbuf*/ &recvCounts[0],
                  /*recvcounts*/ &tmpCounts[0],
                  /*rdispls*/ &tmpDspl[0],
                  MPI_INT,
                  MPI_COMM_WORLD);
    // compute recvDisplacements
    int sum = 0;
    for (int i = 0; i < numRanks; i++) {
      recvDisplacements[i] = sum;
      sum += recvCounts[i];
    }
  }

  // send the counts, which won't be in the right order yet
  MPI_Alltoallv(/*sendbuf*/ &sendBuf[0],
                /*sendcounts*/ &sendCounts[0],
                /*sdispls*/ &sendDisplacements[0],
                gCountBufEltMpiType,
                /*recvbuf*/ &recvBuf[0],
                /*recvcounts*/ &recvCounts[0],
                /*rdispls*/ &recvDisplacements[0],
                gCountBufEltMpiType,
                MPI_COMM_WORLD);

  // now sort recvBuf according to digit
  std::sort(recvBuf.begin(), recvBuf.end(),
            [](const CountBufElt& a, const CountBufElt& b) {
                if a.digit != b.digit {
                  return a.digit < b.digit;
                }
                return a.rank < b.rank;
            });

  // now store from recvBuf into the global counts array C
  for (int64_t i = 0; i < globCountsPerNode; i++) {
    CountBufElt elt = recvBuf[i];
    int64_t globalIdx = elt.digit * numRanks + elt.rank;
    assert(globalIdx == myRank*globCountsPerNode + i);
    GlobalCounts[i] = elt.count;
  }

  // Now compute the total of the counts for each chunk of the global
  // counts array
  int64_t myTotalCount = 0;
  for (int64_t i = 0; i < globCountsPerNode; i++) {
    myTotalCount += GlobalCounts[i];
  }

  // Now use MPI_Scan to add up the total counts from each rank
  int64_t myGlobalStart = 0;
  MPI_Scan(&myTotalCount, &myGlobalStart, 1, MPI_LONG_LONG);

  // Now compute the global starts
  std::vector<int64_t> GlobalStarts;
  GlobalStarts.resize(globCountsPerNode);

  for (int64_t i = 0; i < globCountsPerNode; i++) {
    GlobalStarts[i] = myGlobalSTart;
    myGlobalStart += GlobalCounts[i];
  }

  // Now partition the data according to GlobalStarts
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  // read in the problem size
  int64_t n = 100*1000*1000;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--n") {
      n = std::stoll(argv[++i]);
    }
  }

  int myRank = 0;
  int numRanks = 0;
  MPI_Comm_rank (MPI_COMM_WORLD, &myRank);
  MPI_Comm_size (MPI_COMM_WORLD, &numRanks);
 
  if (myRank == 0) {
    printf("Total number of MPI ranks: %i\n", numRanks);
    printf("Problem size: %lli\n", (long long int) n);
  }

  // setup the global type
  assert(sizeof(unsigned long long) == sizeof(uint64_t));
  assert(2*sizeof(unsigned long long) == sizeof(SortElement));
  MPI_Type_contiguous(2, MPI_UNSIGNED_LONG_LONG, &gSortElementMpiType);
  MPI_Type_commit(&gSortElementMpiType);
  assert(2*sizeof(unsigned long long) == sizeof(CountBufElt));
  MPI_Type_contiguous(2, MPI_UNSIGNED_LONG_LONG, &gCountBufEltMpiType);
  MPI_Type_commit(&gCountBufEltMpiType);

  // create a "distributed array" in the form of different allocations
  // on each node.
  int64_t perNode = divCeil(n, numRanks);
  int64_t myChunkSize = perNode;
  if (perNode*myRank + myChunkSize > n) {
    myChunkSize = n - perNode*myRank;
    if (myChunkSize < 0) myChunkSize = 0;
  }

  std::vector<SortElement> A;
  A.resize(myChunkSize);
  std::vector<SortElement> B;
  B.resize(myChunkSize);
 
  // set up the values to random values and the indices to global indices
  {
    auto start = std::chrono::steady_clock::now();
    if (myRank == 0) {
      printf("Generating random values\n");
    }

    auto rng = pcg64(myRank);
    int64_t i = perNode * myRank;
    for (auto& elt : A) {
      elt.key = rng();
      elt.val = i;
      i++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (myRank == 0) {
      printf("Generated random values in %lf s\n", elapsed.count());
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Shuffle the data in-place to sort by the current digit
  {
    auto start = std::chrono::steady_clock::now();
    if (myRank == 0) {
      printf("Sorting (TODO)\n");

      // MPI_Scan

      // MPI_Alltoallv
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (myRank == 0) {
      printf("Sorted %lli values in %lf s\n",
             (long long int) n, elapsed.count());
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
 
  // Print out the first few elements on each locale
  size_t nToPrint = 10;
  if (myRank == 0) {
    printf("Displaying first %i elements on each rank\n", (int)nToPrint);
  }
  for (int rank = 0; rank < numRanks; rank++) {
    if (myRank == rank) {
      for (size_t i = 0; i < nToPrint && i < A.size; i++) {
        printf("A[%lli] = (%016llx,%llu)\n",
               (long long int) (myRank*perNode + i),
               (long long unsigned) A[i].key,
               (long long unsigned) A[i].val);
      }
      printf("...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}

// NOTE: To understand the Teuchos and Tpetra code, see the Teuchos and Tpetra
// Doxygen documentation online.
