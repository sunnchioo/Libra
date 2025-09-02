link_libraries(bitwise)

add_executable(example_boolean example_boolean.cu)
add_executable(example_boolean-batch example_boolean-batch.cu)

add_executable(example_eval-function example_eval-function.cu)
add_executable(example_eval-function-batch example_eval-function-batch.cu)

add_executable(example_eval-decomp example_eval-decomp.cu)
add_executable(example_eval-flooring example_eval-flooring.cu)
add_executable(example_eval-sign example_eval-sign.cu)

##### new add
# add_executable(bin-test-mult bin-test-mult.cu)
