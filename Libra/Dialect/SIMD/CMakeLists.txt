include_directories(${CMAKE_CURRENT_SOURCE_DIR}/IR)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/IR)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../SCFHE/IR)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../SCFHE/IR)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Pass)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/Pass)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Common)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/Common)

add_subdirectory(IR)
add_subdirectory(Pass)
