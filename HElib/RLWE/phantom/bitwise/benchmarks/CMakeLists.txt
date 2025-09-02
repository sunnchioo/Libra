link_libraries(bitwise)

add_executable(bench_boolean bench_boolean.cu)
target_link_libraries(bench_boolean benchmark::benchmark)

add_executable(bench_boolean-32bit bench_boolean-32bit.cu)
target_link_libraries(bench_boolean-32bit benchmark::benchmark)

add_executable(bench_eval-function bench_eval-function.cu)
target_link_libraries(bench_eval-function benchmark::benchmark)

add_executable(bench_fourstep_ntt bench_fourstep_ntt.cu)
target_link_libraries(bench_fourstep_ntt nvbench::main)
