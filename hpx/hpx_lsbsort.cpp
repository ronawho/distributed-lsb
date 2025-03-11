#include <hpx/config.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/modules/program_options.hpp>
//#include <hpx/collectives/exclusive_scan.hpp>
#include <hpx/modules/collectives.hpp>

#include <pcg_random.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <span>

#define RADIX 16
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
HPX_REGISTER_PARTITIONED_VECTOR(SortElement)
HPX_REGISTER_PARTITIONED_VECTOR(int64_t)

#if 0
// partitioned vector view of local elements,
// taken from HPX's examples/quickstart/partitioned_vector_spmd_foreach.cpp
// and modified to change the name & to enable the assertion in more cases
///////////////////////////////////////////////////////////////////////////////
//
// Define a view for a partitioned vector which exposes the part of the vector
// which is located on the current locality.
//
// This view does not own the data and relies on the partitioned_vector to be
// available during the full lifetime of the view.
//
template <typename T>
struct partitioned_vector_loc_view
{
private:
    typedef typename hpx::partitioned_vector<T>::iterator global_iterator;
    typedef typename hpx::partitioned_vector<T>::const_iterator
        const_global_iterator;

    typedef hpx::traits::segmented_iterator_traits<global_iterator> traits;
    typedef hpx::traits::segmented_iterator_traits<const_global_iterator>
        const_traits;

    typedef typename traits::local_segment_iterator local_segment_iterator;

public:
    typedef typename traits::local_raw_iterator iterator;
    typedef typename const_traits::local_raw_iterator const_iterator;
    typedef T value_type;

public:
    explicit partitioned_vector_loc_view(hpx::partitioned_vector<T>& data)
      : segment_iterator_(data.segment_begin(hpx::get_locality_id()))
    {
        // this view assumes that there is exactly one segment per locality
        typedef typename traits::local_segment_iterator local_segment_iterator;
        local_segment_iterator sit = segment_iterator_;
        assert(++sit == data.segment_end(hpx::get_locality_id()));
    }

    iterator begin()
    {
        return traits::begin(segment_iterator_);
    }
    iterator end()
    {
        return traits::end(segment_iterator_);
    }

    const_iterator begin() const
    {
        return const_traits::begin(segment_iterator_);
    }
    const_iterator end() const
    {
        return const_traits::end(segment_iterator_);
    }
    const_iterator cbegin() const
    {
        return begin();
    }
    const_iterator cend() const
    {
        return end();
    }

    value_type& operator[](std::size_t index)
    {
        return (*segment_iterator_)[index];
    }
    value_type const& operator[](std::size_t index) const
    {
        return (*segment_iterator_)[index];
    }

    std::size_t size() const
    {
        return (*segment_iterator_).size();
    }

private:
    local_segment_iterator segment_iterator_;
};

template <typename T>
struct const_partitioned_vector_loc_view
{
private:
    typedef typename hpx::partitioned_vector<T>::iterator global_iterator;
    typedef typename hpx::partitioned_vector<T>::const_iterator
        const_global_iterator;

    typedef hpx::traits::segmented_iterator_traits<global_iterator> traits;
    typedef hpx::traits::segmented_iterator_traits<const_global_iterator>
        const_traits;

    typedef typename traits::local_segment_iterator local_segment_iterator;

public:
    typedef typename traits::local_raw_iterator iterator;
    typedef typename const_traits::local_raw_iterator const_iterator;
    typedef T value_type;

public:
    explicit const_partitioned_vector_loc_view(const hpx::partitioned_vector<T>& data)
      : segment_iterator_(data.segment_begin(hpx::get_locality_id()))
    {
        // this view assumes that there is exactly one segment per locality
        typedef typename traits::local_segment_iterator local_segment_iterator;
        local_segment_iterator sit = segment_iterator_;
        assert(++sit == data.segment_end(hpx::get_locality_id()));
    }

    const_iterator begin()
    {
        return traits::begin(segment_iterator_);
    }
    const_iterator end()
    {
        return traits::end(segment_iterator_);
    }

    const_iterator begin() const
    {
        return const_traits::begin(segment_iterator_);
    }
    const_iterator end() const
    {
        return const_traits::end(segment_iterator_);
    }
    const_iterator cbegin() const
    {
        return begin();
    }
    const_iterator cend() const
    {
        return end();
    }

    value_type const& operator[](std::size_t index)
    {
        return (*segment_iterator_)[index];
    }
    value_type const& operator[](std::size_t index) const
    {
        return (*segment_iterator_)[index];
    }

    std::size_t size() const
    {
        return (*segment_iterator_).size();
    }

private:
    local_segment_iterator segment_iterator_;
};
#endif

// helper to divide while rounding up
static inline int64_t divCeil(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// helper to get the start..<end positions
// for a given task in an array
static inline std::pair<int64_t, int64_t> myChunk(int64_t n, int64_t threadId, int64_t numThreads) {
  int64_t perThread = divCeil(n, numThreads);
  int64_t start = threadId*perThread;
  int64_t end = start + perThread;
  if (end > n) end = n;
  if (end < start) end = start; 
  return std::make_pair(start, end);
}

// A helper for distributed arrays with a fixed segment size
template<typename EltType>
struct DistributedArray {
  struct RankAndLocalIndex {
    int rank = 0;
    int64_t locIdx = 0;
  };

  std::string name_;
  hpx::partitioned_vector<EltType> pv_;
  //partitioned_vector_loc_view<EltType> lv_;
  int64_t numElementsTotal_ = 0;    // number of elements on all ranks
  int64_t numElementsPerRank_ = 0 ; // number per rank
  int64_t numElementsHere_ = 0;     // number this rank
  int myRank_ = 0;
  int numRanks_ = 0;

  static DistributedArray<EltType>
  create(std::string name, int64_t totalNumElements);

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

  hpx::partitioned_vector<EltType>& pv() { return pv_; }
  const hpx::partitioned_vector<EltType>& pv() const { return pv_; }

  /*
  partitioned_vector_loc_view<EltType> localView() {
    return partitioned_vector_loc_view<EltType>(pv_);
  }
  const_partitioned_vector_loc_view<EltType> localView() const {
    return const_partitioned_vector_loc_view<EltType>(pv_);
  }*/

 private:
  EltType* locPtrImpl() const {
    auto id = hpx::get_locality_id();
    auto seg_it  = pv_.segment_begin(id);
    auto seg_end = pv_.segment_end(id);

    EltType* p = &((*seg_it)[0]);

    ++seg_it;
    assert(seg_it == seg_end); // should be 1 segment per locality

    return p;
  }

 public:
  EltType* locPtr() {
    return locPtrImpl();
  }
  const EltType* locPtr() const {
    return locPtrImpl();
  }

  std::span<EltType> localPart() {
    EltType* p = locPtr();
    return std::span<EltType>(p, numElementsHere_);
  }
  std::span<const EltType> localPart() const {
    const EltType* p = locPtr();
    return std::span<const EltType>(p, numElementsHere_);
  }

  /*
  const std::vector<EltType>& localPart() const {
    assert(lv_.begin() != lv_.end());
    return *lv_.begin(); // tries to return an EltType
  }
  std::vector<EltType>& localPart() {
    assert(lv_.begin() != lv_.end());
    return *lv_.begin(); // oop
  }*/
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
  // it is assumed that this is run from all localities (collectively)
  std::vector<hpx::id_type> localities = hpx::find_all_localities();

  auto myRank = hpx::get_locality_id();
  auto numRanks = localities.size();
  int64_t eltsPerRank = divCeil(totalNumElements, numRanks);
  int64_t eltsHere = eltsPerRank;
  if (eltsPerRank*myRank + eltsHere > totalNumElements) {
    eltsHere = totalNumElements - eltsPerRank*myRank;
  }
  if (eltsHere < 0) eltsHere = 0;
  int64_t totalNumEltsRounded = eltsPerRank * numRanks;

 //std::cout << "creating dist array " << name << " on rank " << myRank << "/" << numRanks << "\n";

  auto pv = hpx::partitioned_vector<EltType>();
  std::string latchName = name + "latch";
  hpx::distributed::latch l;
  if (0 == myRank) {
    pv = hpx::partitioned_vector<EltType>(totalNumEltsRounded,
                                          hpx::container_layout(localities));
    pv.register_as(name);
    l = hpx::distributed::latch(numRanks);
    l.register_as(latchName);
  } else {
    hpx::future<void> f1 = pv.connect_to(name);
    l.connect_to(latchName);
    f1.get();
  }

  /*if (0 == myRank) {
    std::cout << "created dist array " << name << " with size " << pv.size() << "\n";
  }*/

  //std::cout << " " << name << ".size " << pv.size() << " on " << myRank << "\n";

  // set up the lv view
  //auto lv = partitioned_vector_loc_view<EltType>(pv);

  hpx::distributed::barrier::synchronize();

  DistributedArray<EltType> ret;
  ret.name_ = std::move(name);
  ret.pv_ = std::move(pv);
  //ret.lv_ = std::move(lv);
  ret.numElementsTotal_ = totalNumElements;
  ret.numElementsPerRank_ = eltsPerRank;
  ret.numElementsHere_ = eltsHere;
  ret.myRank_ = myRank;
  ret.numRanks_ = numRanks;

  return ret;
}

template<typename EltType>
void DistributedArray<EltType>::print(int64_t nToPrintPerRank) const {
  hpx::distributed::barrier::synchronize();
  if (myRank_ == 0) {
    if (nToPrintPerRank*numRanks_ >= numElementsTotal_) {
      std::cout << name_ << ": displaying all "
                << numElementsTotal_ << " elements\n";
    } else {
      std::cout << name_ << ": displaying first " << nToPrintPerRank
                << " elements on each rank"
                << " out of " << numElementsTotal_ << " elements\n";
    }

    for (int rank = 0; rank < numRanks_; rank++) {
      int64_t i = 0;
      int64_t start = rank*numElementsPerRank_;
      for (i = 0; i < nToPrintPerRank && i < numElementsHere_; i++) {
        int64_t glbIdx = start+i;
        std::cout << name_ << "[" << glbIdx << "] = " << pv_[glbIdx] << "\n";
      }
      if (i < numElementsHere_) {
        std::cout << "...\n";
      }
    }
  }
  hpx::distributed::barrier::synchronize();
}

// compute the bucket for a value when sort is on digit 'd'
inline int getBucket(SortElement x, int d) {
  return (x.key >> (RADIX*d)) & MASK;
}

// counts the number of occurences of each digit value at position 'digit'
//   elts: pointer to the input data
//   nElts: number of elements pointed to by elts
//   counts: after it runs, contains the number of elements for each bucket
//   digit: the current digit for shuffling
void localCount(const SortElement* elts,
                int64_t nElts,
                counts_array_t& counts,
                int digit) {
  // clear out counts
  counts.fill(0);

  // compute the count for each digit
  for (int64_t i = 0; i < nElts; i++) {
    SortElement elt = elts[i];
    counts[getBucket(elt, digit)] += 1;
  }
}

void shufflePut(DistributedArray<SortElement>& Dst,
                const SortElement* elts,
                int64_t nElts,
                std::vector<int64_t>& starts,
                int digit) {

  const size_t maxBuf = 10000;
  std::vector<size_t> dstIndexes;
  std::vector<SortElement> dstValues;

  for (int64_t i = 0; i < nElts; i++) {
    if (dstIndexes.size() > maxBuf) {
      Dst.pv().set_values(hpx::launch::sync, dstIndexes, dstValues);
      dstIndexes.clear();
      dstValues.clear();
    }

    SortElement elt = elts[i];
    int bucket = getBucket(elt, digit);
    int64_t& bucketStartRef = starts[bucket];
    int64_t dstPos = bucketStartRef;
    bucketStartRef++;
    dstIndexes.push_back(dstPos);
    dstValues.push_back(elt);
  }

  if (dstIndexes.size() > 0) {
    Dst.pv().set_values(hpx::launch::sync, dstIndexes, dstValues);
    dstIndexes.clear();
    dstValues.clear();
  }
}

// compute a global transposed index for the distributed counts array
static inline int64_t getGlobalCountIndex(int64_t bucket,
                                          int64_t threadId,
                                          int64_t nThreads,
                                          int64_t rank,
                                          int64_t nRanks) {
  return bucket*nRanks*nThreads + rank*nThreads + threadId;
}

                                          
void exclusiveScan(const DistributedArray<int64_t>& Src,
                   DistributedArray<int64_t>& Dst,
                   int64_t generation) {

  uint32_t myRank = hpx::get_locality_id();
  uint32_t numRanks = hpx::get_num_localities(hpx::launch::sync);

  //std::size_t const nThreads = hpx::get_os_thread_count();

  // Now compute the total of for each segment of the global Src array
  auto LocSrc = Src.localPart();
  auto LocDst = Dst.localPart();


  int64_t totalHere = hpx::reduce(hpx::execution::par,
                                  LocSrc.begin(), LocSrc.end(),
                                  0, std::plus{});

  //printf("node %i totalHere %i\n", (int)myRank, (int) totalHere);

  // BUG hpx::collectives::exclusive_scan is an inclusive scan!!!

  // Now use the scan collective to add up the totals from each rank
  hpx::future<int64_t> myGlobalStartF =
    hpx::collectives::inclusive_scan("myScan",
                                     totalHere,
                                     std::plus<int64_t>{},
                                     hpx::collectives::num_sites_arg(),
                                     hpx::collectives::this_site_arg(),
                                     hpx::collectives::generation_arg(generation));


  /*
  std::string scanName = std::string("myScan") + std::to_string(generation);
  auto exclusive_scan_client = create_communicator(scanName,
                                                   num_sites_arg(numRanks),
                                                   this_site_arg(myRank));


  hpx::future<int64_t> overall_result = exclusive_scan(
      exclusive_scan_client, value, std::plus<std::uint32_t>{});*/

  int64_t myGlobalStart = myGlobalStartF.get();
  if (myRank == 0) {
    myGlobalStart = 0; // avoid odd behavior around rank 0
  } else {
    myGlobalStart -= totalHere; // inclusive -> exclusive scan
  }

  //printf("node %i myGlobalStart %i totalHere %i\n", (int)myRank, (int) myGlobalStart, (int) totalHere);

  // now compute parallel prefix sum / exclusive scan of
  // the elements on this rank
  hpx::exclusive_scan(hpx::execution::par, LocSrc.begin(), LocSrc.end(),
                      LocDst.begin(),
                      myGlobalStart,
                      std::plus<int64_t>{});
}

// shuffles based on digit from Src to Dst
void globalShuffle(DistributedArray<SortElement>& Src,
                   DistributedArray<SortElement>& Dst,
                   DistributedArray<int64_t>& GlobalCounts,
                   DistributedArray<int64_t>& GlobalStarts,
                   int digit) {
  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  auto myRank = hpx::get_locality_id();
  auto numRanks = localities.size();
  std::size_t const nThreads = hpx::get_os_thread_count();

  SortElement* locData = Src.locPtr();
  int64_t locDataN = Src.numElementsHere();

  // allocate the inner counts_array_t for perThreadCounts
  // count for the portion of the input this thread is responsible for
  // store the counts in the global counts array
  hpx::wait_all(hpx::parallel::execution::bulk_async_execute(
    hpx::execution::par,
    [=, &GlobalCounts](std::size_t tid) {
      std::unique_ptr<counts_array_t> myCountsOwned = std::make_unique<counts_array_t>();
      counts_array_t& myCounts = *myCountsOwned;
      auto [threadStart, threadEnd] = myChunk(locDataN, tid, nThreads);
      //printf("thread %i on node %i handling [%i,%i]\n", (int)tid, (int)myRank, (int) threadStart, (int)threadEnd );
      localCount(locData + threadStart, threadEnd-threadStart, myCounts, digit);

      // copy the counts to the global counts
      std::vector<size_t> dstIndexes(COUNTS_SIZE);
      std::vector<int64_t> dstValues(COUNTS_SIZE);
      
      for (int64_t bucket = 0; bucket < COUNTS_SIZE; bucket++) {
        int64_t globalCountIdx =
          getGlobalCountIndex(bucket, tid, nThreads, myRank, numRanks);
        dstIndexes[bucket] = globalCountIdx;
        dstValues[bucket] = myCounts[bucket];
      }

      GlobalCounts.pv().set_values(hpx::launch::sync, dstIndexes, dstValues);
    },
    nThreads));

  hpx::distributed::barrier::synchronize();

  //GlobalCounts.print(100); // debug

  // exclusive scan / prefix sum to compute GlobalStarts
  exclusiveScan(GlobalCounts, GlobalStarts, digit+1);

  hpx::distributed::barrier::synchronize();

  //GlobalStarts.print(100); // debug

  hpx::wait_all(hpx::parallel::execution::bulk_async_execute(
    hpx::execution::par,
    [=, &GlobalStarts, &Dst](std::size_t tid) {
      // copy the counts from the global counts
      std::vector<size_t> srcIndexes(COUNTS_SIZE);
      
      for (int64_t bucket = 0; bucket < COUNTS_SIZE; bucket++) {
        int64_t globalCountIdx =
          getGlobalCountIndex(bucket, tid, nThreads, myRank, numRanks);
        srcIndexes[bucket] = globalCountIdx;
      }

      hpx::future<std::vector<int64_t>> myStartsF =
        GlobalStarts.pv().get_values(srcIndexes);
      std::vector<int64_t> myStarts = myStartsF.get();

      auto [threadStart, threadEnd] = myChunk(locDataN, tid, nThreads);
      //printf("thread %i on node %i handling [%i,%i]\n", (int)tid, (int)myRank, (int) threadStart, (int)threadEnd );

      shufflePut(Dst, locData + threadStart, threadEnd-threadStart, myStarts, digit);
    },
    nThreads));
  
  hpx::distributed::barrier::synchronize();
}

// Sort the data in A, using B as scratch space.
void mySort(DistributedArray<SortElement>& A,
            DistributedArray<SortElement>& B,
            DistributedArray<int64_t>& GlobalCounts,
            DistributedArray<int64_t>& GlobalStarts) {
  auto here_id = hpx::get_locality_id();
  assert(N_DIGITS % 2 == 0);
  for (int digit = 0; digit < N_DIGITS; digit += 2) {
    //if (here_id == 0) printf("Processing digit %i\n", digit);
    globalShuffle(A, B, GlobalCounts, GlobalStarts, digit);
    globalShuffle(B, A, GlobalCounts, GlobalStarts, digit+1);
  }
}

void fill_random(DistributedArray<SortElement>& A) {
  auto here_id = hpx::get_locality_id();
  std::size_t const nThreads = hpx::get_os_thread_count();
  SortElement* locData = A.locPtr();
  int64_t locDataN = A.numElementsHere();

  //std::cout << "filling array (size " << A.numElementsTotal() << " localView.size() " << A.localPart().size() << " ) from " << here_id << std::endl;

  hpx::wait_all(hpx::parallel::execution::bulk_async_execute(
    hpx::execution::par,
    [=, &A](std::size_t tid) {
      auto [threadStart, threadEnd] = myChunk(locDataN, tid, nThreads);
      //printf("Setting locidxs [%i,%i] from node %i thread %i\n", (int)threadStart, (int)threadEnd, (int)here_id, (int)tid); 
      auto rng = pcg64(here_id*nThreads+tid);
      for (int64_t j = threadStart; j < threadEnd; j++) { 
        //printf("Setting locidx %i from node %i thread %i\n", (int)j, (int)here_id, (int)tid); 
        SortElement& elt = locData[j];
        elt.key = rng();
        elt.val = A.localIdxToGlobalIdx(j);
      }
    },
    nThreads));
}

int hpx_main(hpx::program_options::variables_map& vm)
{
  // this is run on all localities (nodes)

  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  auto myRank = hpx::get_locality_id();
  auto numRanks = localities.size();
  std::size_t const nThreads = hpx::get_os_thread_count();

  //std::cout << "hello from rank " << myRank << "/" << numRanks << "\n";

  int64_t numElems = 128*1024*1024;
  if (vm.count("numElems"))
    numElems = vm["numElems"].as<int64_t>();
  bool printSome = false;
  if (vm.count("print"))
    printSome = vm["print"].as<bool>();

  if (0 == myRank) {
    std::cout << "using numElems: " << numElems << "\n";
    std::cout << "creating partitioned vector" << "\n";
    std::cout << "num threads per node: " << nThreads << "\n";
  }

  auto A = DistributedArray<SortElement>::create("A", numElems);
  auto B = DistributedArray<SortElement>::create("B", numElems);
  auto GlobalCounts = DistributedArray<int64_t>::create("GlobalCounts",
                                                        COUNTS_SIZE*numRanks*nThreads);
  auto GlobalStarts = DistributedArray<int64_t>::create("GlobalStarts",
                                                        COUNTS_SIZE*numRanks*nThreads);

  {
    hpx::distributed::barrier::synchronize();
    auto start = std::chrono::steady_clock::now();
    if (myRank == 0) {
      std::cout << "Generating random values\n";
    }

    fill_random(A);

    hpx::distributed::barrier::synchronize();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (myRank == 0) {
      std::cout << "Generated random values in " << elapsed.count() << " s\n";
    }
    if (printSome) A.print(10);
  }
 

  {
    if (myRank == 0) {
      std::cout << "Sorting\n";
    }

    hpx::distributed::barrier::synchronize();
    auto start = std::chrono::steady_clock::now();

    mySort(A, B, GlobalCounts, GlobalStarts);

    hpx::distributed::barrier::synchronize();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (printSome) A.print(10);

    if (myRank == 0) {
      std::cout << "Sorted " << numElems << " values in " << elapsed.count() << "\n";
      std::cout << "That's " << numElems/elapsed.count()/1000.0/1000.0
                << " M elements sorted / s\n";
    }
  }

  hpx::distributed::barrier::synchronize();

  return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("numElems,n", value<int64_t>(),
            "the data array size to use (default: 128*1024*1024)")
        ("print,p", value<bool>(),
            "print some of the array (default: false)")
        ;

    // run hpx_main on all localities
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
