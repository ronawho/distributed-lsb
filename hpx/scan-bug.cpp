#include <hpx/config.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/collectives.hpp>

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
  auto myRank = hpx::get_locality_id();
  auto numRanks = hpx::get_num_localities(hpx::launch::sync);

  hpx::distributed::barrier::synchronize();

  int64_t totalHere = myRank + 100;

  int generation = 1;

  hpx::future<int64_t> myGlobalStartF =
    hpx::collectives::exclusive_scan("myScan",
                                     totalHere,
                                     std::plus<int64_t>{},
                                     hpx::collectives::num_sites_arg(),
                                     hpx::collectives::this_site_arg(),
                                     hpx::collectives::generation_arg(generation));


  int64_t myGlobalStart = myGlobalStartF.get();
  if (myRank == 0) {
    myGlobalStart = 0; // avoid odd behavior around rank 0
  }

  std::cout << "node " << myRank << " totalHere " << totalHere << " myGlobalStart " << myGlobalStart << "\n";

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

