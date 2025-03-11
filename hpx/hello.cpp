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


int hpx_main(hpx::program_options::variables_map& vm)
{
  // this is run on all localities (nodes)

  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  auto myRank = hpx::get_locality_id();
  auto numRanks = localities.size();
  std::size_t const nThreads = hpx::get_os_thread_count();

  hpx::distributed::barrier::synchronize();

  std::cout << "hello from rank " << myRank << "/" << numRanks <<
     " where there are " << nThreads << " threads\n";

  hpx::distributed::barrier::synchronize();

  return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // run hpx_main on all localities
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
