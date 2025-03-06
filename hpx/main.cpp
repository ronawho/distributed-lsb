#include <hpx/config.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/modules/program_options.hpp>

#include <pcg_random.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define RADIX 16
#define N_DIGITS (64/RADIX)
#define N_BUCKETS (1 << RADIX)
#define COUNTS_SIZE (N_BUCKETS + 1)
#define MASK (N_BUCKETS - 1)

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

template<typename T>
void print_partitioned_array(const hpx::partitioned_vector<T>& A, std::string name) {
  int64_t i = 0;
  for (auto elt: A) {
    std::cout << name << "[" << i << "] = " << elt << "\n";
    i++;
  }
}

// to help with sending sort elements to remote locales
// helper to divide while rounding up
static inline int64_t divCeil(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

void fill_local_array(hpx::partitioned_vector<SortElement> &A) {
  auto here_id = hpx::get_locality_id();

  std::cout << "filling array (size " << A.size() << " ) from " << here_id << std::endl;

  // TODO: ideally this would be thread-parallel,
  // but that is not a focus here.
  auto rng = pcg64(here_id);
  auto it = A.segment_begin(here_id);
  auto end = A.segment_end(here_id);
  for (; it != end; ++it) {
    std::cout << "have a segment from " << here_id << std::endl;
    int64_t i = A.get_global_index(it, 0);
    auto subit = it->begin();
    auto subend = it->end();

    /*
    auto sub = std::ranges::subrange{it->begin
    std::for_each(std::execution::par,
                  std::views::zip(nums, chars).begin(), std::views::zip(nums, chars).end(),
              [](auto const& pair) {
                  std::cout << std::get<0>(pair) << ": " << std::get<1>(pair) << std::endl;
              });
    */

    for (; subit != subend; ++subit, ++i) {
      SortElement& elt = *subit;
      elt.key = rng();
      elt.val = i;
      std::cout << elt << " created by " << here_id << "\n";
    }
  }
}

int hpx_main(hpx::program_options::variables_map& vm)
{
  // this is run on all localities (nodes)

  std::vector<hpx::id_type> localities = hpx::find_all_localities();

  auto myRank = hpx::get_locality_id();
  auto numRanks = localities.size();

  std::cout << "hello from rank " << myRank << "/" << numRanks << "\n";

  int64_t numElems = 128*1024*1024;
  if (vm.count("numElems"))
      numElems = vm["numElems"].as<int64_t>();

  if (0 == myRank) {
    std::cout << "using numElems: " << numElems << "\n";
    std::cout << "creating partitioned vector" << "\n";
  }

  auto A = hpx::partitioned_vector<SortElement>();
  auto B = hpx::partitioned_vector<SortElement>();
  hpx::distributed::latch l;
  if (0 == myRank) {
    A = hpx::partitioned_vector<SortElement>(numElems,
                                             hpx::container_layout(localities));
    A.register_as("A");
    B = hpx::partitioned_vector<SortElement>(numElems,
                                             hpx::container_layout(localities));
    B.register_as("B");
    l = hpx::distributed::latch(localities.size());
    l.register_as("latch");
  } else {
    hpx::future<void> f1 = A.connect_to("A");
    hpx::future<void> f2 = B.connect_to("B");
    l.connect_to("latch");
    f1.get(); f2.get();
  }

  if (0 == myRank) {
    std::cout << "created with size " << A.size() << "\n";
  }

  std::cout << " A.size " << A.size() << " on " << myRank << "\n";

  // set the keys to random values and the values to global indices
  fill_local_array(A);

  if (0 == myRank) {
    print_partitioned_array(A, "A");
  }

  // Wait for all localities to reach this point.
  l.arrive_and_wait();

  l.arrive_and_wait();

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
        ("seed,s", value<unsigned int>(),
            "the random number generator seed to use for this run (default: 1)")
        ;

    // run hpx_main on all localities
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
