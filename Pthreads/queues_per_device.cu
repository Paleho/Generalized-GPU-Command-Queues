#include <cuda.h>
#include "queues_per_device.hpp"

int queues_per_dev_lock = 0;

inline void get_queues_per_dev_lock(){
	while(__sync_lock_test_and_set (&queues_per_dev_lock, 1)){
		;
	}
}
inline void release_queues_per_dev_lock(){
	__sync_lock_release(&queues_per_dev_lock);
}

std::vector<CommandQueue *> * queuesPerDevice;
bool queuesPerDeviceInitialized = false;

void InitializeQueuesPerDevice(){
    get_queues_per_dev_lock();
    if(queuesPerDeviceInitialized){
        release_queues_per_dev_lock();
        warning("InitializeQueuesPerDevice(): already initialized\n");
        return;
    }

    // Get number of devices
    int dev_count;
    cudaError_t err = cudaGetDeviceCount(&dev_count);
    massert(cudaSuccess == err, "InitializeQueuesPerDevice(): cudaGetDeviceCount() failed - %s\n", cudaGetErrorString(err));

    queuesPerDevice = new std::vector<CommandQueue *>[dev_count]();
    if(!queuesPerDevice){
        release_queues_per_dev_lock();
        error("InitializeQueuesPerDevice(): malloc failed\n");
        return;
    }

    queuesPerDeviceInitialized = true;
    release_queues_per_dev_lock();
}

void UninitializeQueuesPerDevice(){
    get_queues_per_dev_lock();
    if(!queuesPerDeviceInitialized){
        release_queues_per_dev_lock();
        error("UninitializeQueuesPerDevice(%): cannot destroy, structure is not initialized\n");
        return;
    }

    delete [] queuesPerDevice;

    queuesPerDeviceInitialized = false;
    release_queues_per_dev_lock();
}

void AssignQueueToDevice(CommandQueue * queue, int dev){
    get_queues_per_dev_lock();
#ifdef DEBUG
	lprintf(1, "AssignQueueToDevice(%p, %d)\n", queue, dev);
#endif
    if(!queuesPerDeviceInitialized){
        release_queues_per_dev_lock();
        error("AssignQueueToDevice(%p, %d): cannot assign, structure is not initialized (call InitializeQueuesPerDevice() first)\n", queue, dev);
        return;
    }
    // "Host" device loc id used by CoCoPeLia is 0. See CoCoPeLiaSelectDevice
    int inner_dev_id = (dev == -1) ? 0: dev;

    queuesPerDevice[inner_dev_id].push_back(queue);
    release_queues_per_dev_lock();
}

void UnassignQueueFromDevice(CommandQueue * queue, int dev){
    get_queues_per_dev_lock();
    if(!queuesPerDeviceInitialized){
        release_queues_per_dev_lock();
        error("UnassignQueueFromDevice(%p, %d): cannot unassign, structure is not initialized\n", queue, dev);
        return;
    }
    // "Host" device loc id used by CoCoPeLia is 0. See CoCoPeLiaSelectDevice
    int inner_dev_id = (dev == -1) ? 0: dev;

	for(int i = 0; i < queuesPerDevice[inner_dev_id].size(); i++){
		if(queuesPerDevice[inner_dev_id][i] == queue) queuesPerDevice[inner_dev_id].erase(queuesPerDevice[inner_dev_id].begin()+i);
	}
    release_queues_per_dev_lock();
}

void DeviceSynchronize(){
    get_queues_per_dev_lock();
    if(!queuesPerDeviceInitialized){
        release_queues_per_dev_lock();
        // error("DeviceSynchronize(): cannot synchronize queues, structure is not initialized (call InitializeQueuesPerDevice() first)\n");
        return;
    }

    int dev = -1;
    cudaError_t err = cudaGetDevice(&dev);
    massert(cudaSuccess == err,"DeviceSynchronize(): cudaGetDevice failed - %s\n", cudaGetErrorString(err));

    for(int i = 0; i < queuesPerDevice[dev].size(); i++){
        queuesPerDevice[dev][i]->sync_barrier();
    }
    release_queues_per_dev_lock();
}
