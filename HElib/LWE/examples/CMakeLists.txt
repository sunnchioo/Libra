set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/cuTFHEpp/examples)

include_directories(${CMAKE_SOURCE_DIR}/LWE/cuTFHEpp/include)
include_directories(${CMAKE_SOURCE_DIR}/RLWE/phantom/include)

# include_directories(/usr/local/include)
# include_directories(/usr/local/include/NTL)

link_libraries(cutfhepp)

# examples
add_executable(cupbs cupbs.cu)
target_link_libraries(cupbs PRIVATE ntl gmp m pthread)

