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

static MPI_Datatype gMpiElementType;

struct SortElement {
  uint64_t key; // to sort by
  uint64_t val; // carried along
};

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
      sum += count;
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
    sum += count;
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

  localShuffle(A, B, *starts, *counts, digit);

  // Now, each rank has an array of counts, like this
  //  [r0d0, r0d1, ... r0d255]
  //  [r1d0, r1d1, ... r1d255]
  //  ...
  //

  // We need to transpose these so that the counts have the
  // starting digits first
  //  [r0d0, r1d0, r2d0, ...]
  //  [r0d1, r1d1, r2d1, ...]
  //  ...

  // Here we create a distributed array storing this transposition.
  int64_t globCountsPerNode = divCeil(COUNTS_SIZE, numRanks);
  std::vector<int64_t> C;
  C.resize(globCountsPerNode);

  Scatter
    AllToAll
    Gather
  // Transpose the counts
  MPI_Alltoallv(
int MPI_Alltoallv(const void *sendbuf, const int sendcounts[],
    const int sdispls[], MPI_Datatype sendtype,
    void *recvbuf, const int recvcounts[],

  // Do the global scan to compute the start position for each bucket
  // Note: this needs transposed order!
}


int MPI_Scan(const void *sendbuf, void *recvbuf, int count,
             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
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
  MPI_Type_contiguous(2, MPI_UNSIGNED_LONG_LONG, &gMpiElementType);

  // create a "distributed array" in the form of different allocations
  // on each node.
  int64_t perNode = divCeil(n, numRanks);

  std::vector<SortElement> A;
  A.resize(perNode);
  std::vector<SortElement> B;
  B.resize(perNode);
 
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
