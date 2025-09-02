# link_libraries(boot conver)

# benchmark
# add_executable(kmeans kmeans.cu)
# target_link_libraries(kmeans PRIVATE ntl gmp m pthread)

add_executable(kmeans-boot-dim kmeans-boot-dim.cu)
target_link_libraries(kmeans-boot-dim PRIVATE ntl gmp m pthread)

add_executable(kmeans-boot-points kmeans-boot-points.cu)
target_link_libraries(kmeans-boot-points PRIVATE ntl gmp m pthread)

add_executable(kmeans-boot-points-multicenters kmeans-boot-points-multicenters.cu)
target_link_libraries(kmeans-boot-points-multicenters PRIVATE ntl gmp m pthread)

add_executable(kmeans-multi-scheme kmeans-multi-scheme.cu)
target_link_libraries(kmeans-multi-scheme PRIVATE ntl gmp m pthread)