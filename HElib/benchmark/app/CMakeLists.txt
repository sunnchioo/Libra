set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmark/app)

include_directories(${CMAKE_SOURCE_DIR}/RLWE/phantom/include)
include_directories(${CMAKE_SOURCE_DIR}/RLWE/boot/include)
include_directories(${CMAKE_SOURCE_DIR}/RLWE/boot/include/boostrapping)
include_directories(${CMAKE_SOURCE_DIR}/RLWE/boot/include/boostrapping/common)

include_directories(${CMAKE_SOURCE_DIR}/LWE/cuTFHEpp/include)

include_directories(${CMAKE_SOURCE_DIR}/conversion/include)

include_directories(/usr/local/include)
include_directories(/usr/local/include/NTL)

link_libraries(cutfhepp conver boot)

add_subdirectory(kmeans)
add_subdirectory(softmax)
add_subdirectory(repack)

# benchmark
# add_executable(kmeans kmeans.cu)
# target_link_libraries(kmeans PRIVATE ntl gmp m pthread)

# add_executable(kmeans-boot-dim kmeans-boot-dim.cu)
# target_link_libraries(kmeans-boot-dim PRIVATE ntl gmp m pthread)

# add_executable(kmeans-boot-points kmeans-boot-points.cu)
# target_link_libraries(kmeans-boot-points PRIVATE ntl gmp m pthread)

# add_executable(dec-round dec-round.cu)
# target_link_libraries(dec-round PRIVATE ntl gmp m pthread)

# add_executable(kmeans-boot-points-multicenters kmeans-boot-points-multicenters.cu)
# target_link_libraries(kmeans-boot-points-multicenters PRIVATE ntl gmp m pthread)

# add_executable(kmeans-multi-scheme kmeans-multi-scheme.cu)
# target_link_libraries(kmeans-multi-scheme PRIVATE ntl gmp m pthread)