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