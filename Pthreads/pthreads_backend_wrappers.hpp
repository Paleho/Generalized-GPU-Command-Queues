///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief 
///

// #include <unihelpers.hpp>
#include <pthread.h>

using namespace std;

typedef struct pthread_task { 
    void* func; 
    void* data; 
}* pthread_task_p; 

typedef struct backend_data { 
    void * taskQueue;
    pthread_t threadId;
    int queueLock; 
    bool terminate;
}* backend_data_p; 