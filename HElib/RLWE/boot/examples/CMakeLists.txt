set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/boot/examples)

link_libraries(boot)
# link_libraries(boot)

add_executable(nn nn.cu)
target_link_libraries(nn PRIVATE ntl gmp m pthread)

add_executable(bootstrapping bootstrapping.cu)
target_link_libraries(bootstrapping PRIVATE ntl gmp m pthread)

add_executable(bootstrapping_alpha6 bootstrapping_alpha6.cu)
target_link_libraries(bootstrapping_alpha6 PRIVATE ntl gmp m pthread)

add_executable(test_s2c test_s2c.cu)
target_link_libraries(test_s2c PRIVATE ntl gmp m pthread)

# add_executable(mvm mvm.cu)
# target_link_libraries(mvm PRIVATE ntl gmp m pthread)

add_executable(test_rotate test_rotate.cu)
# target_link_libraries(test_rotate PRIVATE ntl gmp m pthread)
