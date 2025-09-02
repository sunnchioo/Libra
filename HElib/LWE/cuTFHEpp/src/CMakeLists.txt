set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include_directories(${CMAKE_SOURCE_DIR}/LWE/cuTFHEpp/thirdparty/TFHEpp/include)
include_directories(${CMAKE_SOURCE_DIR}/LWE/cuTFHEpp/include)

file(GLOB_RECURSE CUTFHEPP_SRCS 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu )

# link_libraries(tfhe++)
# target_link_libraries(tfhe++)
link_libraries(tfhe++)

add_library(cutfhepp
            STATIC 
            ${CUTFHEPP_SRCS})

# target_link_libraries(cutfhepp
#     PUBLIC
#     tfhe++
# )
