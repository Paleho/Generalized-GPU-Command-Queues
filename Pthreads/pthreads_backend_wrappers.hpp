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

typedef struct queue_data { 
    void * taskQueue;
    pthread_t threadId;
    int queueLock; 
    bool terminate;
}* queue_data_p; 

typedef struct pthread_event {
    void * taskQueue;
    pthread_t threadId;
    event_status estate;
}* pthread_event_p; 