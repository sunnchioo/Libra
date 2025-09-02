add_executable(repackapp repack.cu)
target_link_libraries(repackapp PRIVATE ntl gmp m pthread)