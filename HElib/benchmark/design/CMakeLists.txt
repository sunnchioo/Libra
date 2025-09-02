set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmark/design)

include_directories(${CMAKE_SOURCE_DIR}/RLWE/phantom/include)
include_directories(${CMAKE_SOURCE_DIR}/RLWE/boot/include)
include_directories(${CMAKE_SOURCE_DIR}/RLWE/boot/include/boostrapping)
include_directories(${CMAKE_SOURCE_DIR}/RLWE/boot/include/boostrapping/common)

include_directories(${CMAKE_SOURCE_DIR}/LWE/cuTFHEpp/include)

include_directories(${CMAKE_SOURCE_DIR}/conversion/include)

include_directories(/usr/local/include)
include_directories(/usr/local/include/NTL)

link_libraries(cutfhepp conver boot)

add_executable(compare compare.cu)
target_link_libraries(compare PRIVATE ntl gmp m pthread)

add_executable(findmin findmin.cu)
target_link_libraries(findmin PRIVATE ntl gmp m pthread)