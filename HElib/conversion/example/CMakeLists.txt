link_libraries(conver)

add_executable(repack repack.cu)
target_link_libraries(repack PRIVATE ntl gmp m pthread)

add_executable(extract extract.cu)
target_link_libraries(extract PRIVATE ntl gmp m pthread)

add_executable(Lvl1L Lvl1L.cu)
target_link_libraries(Lvl1L PRIVATE ntl gmp m pthread)

add_executable(ntt ntt.cu)
target_link_libraries(ntt PRIVATE ntl gmp m pthread)

add_executable(mvm mvm.cu)
target_link_libraries(mvm PRIVATE ntl gmp m pthread)