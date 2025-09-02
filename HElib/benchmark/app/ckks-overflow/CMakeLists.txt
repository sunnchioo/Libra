add_executable(dec-round dec-round.cu)
target_link_libraries(dec-round PRIVATE ntl gmp m pthread)
