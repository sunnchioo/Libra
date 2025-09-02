include_directories(${CMAKE_SOURCE_DIR}/RLWE/phantom/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/boostrapping)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/boostrapping/common)

include_directories(/usr/local/include)
include_directories(/usr/local/include/NTL)

add_subdirectory(src)
add_subdirectory(examples)
