message("--- Unihelpers-Testing Invoked ---")

message("Unihelpers-Testing - CMAKE_RUNTIME_OUTPUT_DIRECTORY: " ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable (singleQueueDaxpy singleQueueDaxpy.cpp)
target_link_libraries(singleQueueDaxpy ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

add_executable (threeQueuesDaxpy threeQueuesDaxpy.cpp)
target_link_libraries(threeQueuesDaxpy ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)

add_test(single_queue_daxpy singleQueueDaxpy)
add_test(three_queues_daxpy threeQueuesDaxpy)
