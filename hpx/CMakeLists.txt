cmake_minimum_required(VERSION 3.19)
project(my_hpx_project CXX)
find_package(HPX REQUIRED)

set (CMAKE_CXX_STANDARD 20)

#add_executable(my_hpx_program main.cpp)
#target_link_libraries(my_hpx_program
#                      HPX::hpx
#                      # HPX::wrap_main
#                      HPX::iostreams_component
#                      HPX::partitioned_vector_component
#                     )
add_hpx_executable(my_hpx_program
  SOURCES main.cpp
  COMPONENT_DEPENDENCIES iostreams partitioned_vector)
target_include_directories(my_hpx_program PUBLIC pcg-cpp/include)


add_hpx_executable(scan-bug
  SOURCES scan-bug.cpp
  COMPONENT_DEPENDENCIES iostreams partitioned_vector)
