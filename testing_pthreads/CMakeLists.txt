message("--- Unihelpers-Testing Invoked ---")

message("Unihelpers-Testing - CMAKE_RUNTIME_OUTPUT_DIRECTORY: " ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable (singleQueueTasks singleQueueTasks.cpp)
target_link_libraries(singleQueueTasks ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

add_executable (parallelQueues parallelQueues.cpp)
target_link_libraries(parallelQueues ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

add_executable (twoQueuesEvents twoQueuesEvents.cpp)
target_link_libraries(twoQueuesEvents ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

add_executable (matrixMultInMultipleQueues matrixMultInMultipleQueues.cpp)
target_link_libraries(matrixMultInMultipleQueues ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

add_executable (wrappedMatrixMultQueues wrappedMatrixMultQueues.cpp)
target_link_libraries(wrappedMatrixMultQueues ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

# CUDA tests
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

add_executable (simpleInterDevice simpleInterDevice.cpp)
target_link_libraries(simpleInterDevice ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

add_executable (memcpy2DAsync memcpy2DAsync.cpp)
target_link_libraries(memcpy2DAsync ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

add_executable (orderSingleQueue orderSingleQueue.cpp)
target_link_libraries(orderSingleQueue ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

add_test(NAME matrix_mult_in_multiple_queues_256 COMMAND matrixMultInMultipleQueues 256 synched)
add_test(NAME matrix_mult_in_multiple_queues_256_desynched COMMAND matrixMultInMultipleQueues 256 desynched)
set_tests_properties(
    matrix_mult_in_multiple_queues_256_desynched
PROPERTIES
    WILL_FAIL TRUE
)
add_test(NAME matrix_mult_in_multiple_queues_512 COMMAND matrixMultInMultipleQueues 512 synched)
add_test(NAME matrix_mult_in_multiple_queues_512_desynched COMMAND matrixMultInMultipleQueues 512 desynched)
set_tests_properties(
    matrix_mult_in_multiple_queues_512_desynched
PROPERTIES
    WILL_FAIL TRUE
)

add_test(NAME wrapped_matrix_mult_queues_256 COMMAND wrappedMatrixMultQueues 256 synched)
add_test(NAME wrapped_matrix_mult_queues_256_desynched COMMAND wrappedMatrixMultQueues 256 desynched)
set_tests_properties(
    wrapped_matrix_mult_queues_256_desynched
PROPERTIES
    WILL_FAIL TRUE
)
add_test(NAME wrapped_matrix_mult_queues_512 COMMAND wrappedMatrixMultQueues 512 synched)
add_test(NAME wrapped_matrix_mult_queues_512_desynched COMMAND wrappedMatrixMultQueues 512 desynched)
set_tests_properties(
    wrapped_matrix_mult_queues_512_desynched
PROPERTIES
    WILL_FAIL TRUE
)

add_test(single_queue_daxpy singleQueueDaxpy)
add_test(three_queues_daxpy threeQueuesDaxpy)
add_test(single_queue_dgemm singleQueueDgemm)
add_test(three_queues_dgemm threeQueuesDgemm)
add_test(interdevice_dependencies interDeviceDependencies)

add_test(NAME simple_interdevice_256 COMMAND simpleInterDevice 256 synched)
add_test(NAME simple_interdevice_256_desynched COMMAND simpleInterDevice 256 desynched)
set_tests_properties(
    simple_interdevice_256_desynched
PROPERTIES
    WILL_FAIL TRUE
)
add_test(NAME simple_interdevice_512 COMMAND simpleInterDevice 512 synched)
add_test(NAME simple_interdevice_512_desynched COMMAND simpleInterDevice 512 desynched)
set_tests_properties(
    simple_interdevice_512_desynched
PROPERTIES
    WILL_FAIL TRUE
)