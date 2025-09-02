set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/conver/example)

find_package(OpenMP)
if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${CMAKE_SOURCE_DIR}/RLWE/phantom/include)
include_directories(${CMAKE_SOURCE_DIR}/RLWE/boot/include)
include_directories(${CMAKE_SOURCE_DIR}/RLWE/boot/include/boostrapping)
include_directories(${CMAKE_SOURCE_DIR}/RLWE/boot/include/boostrapping/common)

include_directories(${CMAKE_SOURCE_DIR}/LWE/cuTFHEpp/include)

include_directories(${CMAKE_SOURCE_DIR}/conversion/include)

include_directories(/usr/local/include)
include_directories(/usr/local/include/NTL)

link_libraries(cutfhepp)

# add_subdirectory(cuTFHEpp)

# set(CMAKE_CUDA_FLAGS "-g -G -std=c++20 -maxrregcount=64 -ccbin g++ -expt-relaxed-constexpr")
# set(CMAKE_CUDA_FLAGS "-O2 -std=c++20 -ccbin g++ -expt-relaxed-constexpr")

# set(opera_sources "")
add_subdirectory(src)

# target_link_libraries(conver cutfhe++ tfhe++ Phantom)

add_subdirectory(example)
