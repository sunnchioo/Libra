# set(CMAKE_CUDA_ARCHITECTURES native) # auto-detect GPU

set(CMAKE_CUDA_FLAGS "-maxrregcount=64 -ccbin g++ -expt-relaxed-constexpr")

# set(cutfhe++_sources "")
# include_directories(${CMAKE_SOURCE_DIR}/LWE/cuTFHEpp/thirdparty/TFHEpp/include)

add_subdirectory(thirdparty/TFHEpp)

add_subdirectory(src)

# target_link_libraries(tfhe++)
# link_libraries(tfhe++)

add_subdirectory(test)
