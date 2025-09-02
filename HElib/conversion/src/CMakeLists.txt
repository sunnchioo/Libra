include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

file(GLOB_RECURSE CONVER_SOURCES 
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)

# link_libraries(boot)

add_library(conver STATIC   
        ${CONVER_SOURCES})
target_link_libraries(conver PRIVATE boot phantomfhe cutfhepp)
