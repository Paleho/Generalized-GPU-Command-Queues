message("--- Unihelpers-Testing Invoked ---")

message("Unihelpers-Testing - CMAKE_RUNTIME_OUTPUT_DIRECTORY: " ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable (singleQueueTasks singleQueueTasks.cpp)
target_link_libraries(singleQueueTasks ${UNIHELP_INSTALL_LIB}/libunihelpers.so unihelpers)
