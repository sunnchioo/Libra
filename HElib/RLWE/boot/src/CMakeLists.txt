include_directories(${CMAKE_SOURCE_DIR}/RLWE/boot/include)
include_directories(${CMAKE_SOURCE_DIR}/RLWE/boot/include/bootstrapping)
include_directories(${CMAKE_SOURCE_DIR}/RLWE/boot/include/bootstrapping/common)

include_directories(/usr/local/include)
include_directories(/usr/local/include/NTL)

# link_libraries(phantomfhe)

file(GLOB_RECURSE BOOT_SOURCES 
        ${CMAKE_CURRENT_SOURCE_DIR}/bootstrapping/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/bootstrapping/common/*.cu)

add_library(boot STATIC 
        argmax.cu  
        gelu.cu  
        layer_norm.cu  
        matrix_mul.cu  
        softmax.cu
        # ckks_evaluator.cu

        ${BOOT_SOURCES})

target_link_libraries(boot PUBLIC phantomfhe ntl)