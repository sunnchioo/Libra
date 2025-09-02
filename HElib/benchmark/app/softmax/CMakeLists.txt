link_libraries(phantomfhe boot)

add_executable(softmax softmax.cu)
target_link_libraries(softmax PRIVATE ntl gmp m pthread)


add_executable(softmax-mlti-scheme softmax-mlti-scheme.cu)
target_link_libraries(softmax-mlti-scheme PRIVATE ntl gmp m pthread)