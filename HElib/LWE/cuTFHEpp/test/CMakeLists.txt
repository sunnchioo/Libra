set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/cuTFHEpp/tests)

include_directories(${CMAKE_SOURCE_DIR}/LWE/cuTFHEpp/include)

option(CUTFHEPP_BUILD_TESTS "Build the cuTFHEpp tests" ON)

if(CUTFHEPP_BUILD_TESTS)
  message(STATUS "Building the cuTFHEpp tests")
  add_subdirectory(fft)
  add_subdirectory(keyswitch)
  add_subdirectory(gatebootstrapping)
endif()
