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
