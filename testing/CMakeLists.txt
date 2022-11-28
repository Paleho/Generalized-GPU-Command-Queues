message("--- Unihelpers-Testing Invoked ---")

message("Unihelpers-Testing - CMAKE_RUNTIME_OUTPUT_DIRECTORY: " ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable (singleQueueDaxpy singleQueueDaxpy.cpp)
target_link_libraries(singleQueueDaxpy ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

add_executable (threeQueuesDaxpy threeQueuesDaxpy.cpp)
target_link_libraries(threeQueuesDaxpy ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

add_executable (singleQueueDgemm singleQueueDgemm.cpp)
target_link_libraries(singleQueueDgemm ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

add_executable (threeQueuesDgemm threeQueuesDgemm.cpp)
target_link_libraries(threeQueuesDgemm ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

add_executable (multiDevMarixMul multiDevMarixMul.cpp)
target_link_libraries(multiDevMarixMul ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

add_executable (interDeviceDependencies interDeviceDependencies.cpp)
target_link_libraries(interDeviceDependencies ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

add_executable (matrixAdd matrixAdd.cpp)
target_link_libraries(matrixAdd ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

add_test(single_queue_daxpy singleQueueDaxpy)
add_test(three_queues_daxpy threeQueuesDaxpy)
add_test(single_queue_dgemm singleQueueDgemm)
add_test(three_queues_dgemm threeQueuesDgemm)
