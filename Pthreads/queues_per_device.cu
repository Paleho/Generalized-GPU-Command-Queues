#include <cuda.h>
#include "queues_per_device.hpp"

std::vector<CommandQueue *> * queuesPerDevice;
bool queuesPerDeviceInitialized = false;

void InitializeQueuesPerDevice(){
    if(queuesPerDeviceInitialized){
        warning("InitializeQueuesPerDevice(): already initialized\n");
        return;
    }

    // Get number of devices
    int dev_count;
    cudaError_t err = cudaGetDeviceCount(&dev_count);
    massert(cudaSuccess == err, "InitializeQueuesPerDevice(): cudaGetDeviceCount() failed - %s\n", cudaGetErrorString(err));

    queuesPerDevice = new std::vector<CommandQueue *>[dev_count]();
    if(!queuesPerDevice) error("InitializeQueuesPerDevice(): malloc failed\n");

    queuesPerDeviceInitialized = true;
}

void UninitializeQueuesPerDevice(){
    if(!queuesPerDeviceInitialized){
        error("UninitializeQueuesPerDevice(%): cannot destroy, structure is not initialized\n");
    }

    delete [] queuesPerDevice;

    queuesPerDeviceInitialized = false;
}

void AssignQueueToDevice(CommandQueue * queue, int dev){
#ifdef DEBUG
	lprintf(1, "AssignQueueToDevice(%p, %d)\n", queue, dev);
#endif
    if(!queuesPerDeviceInitialized){
        error("AssignQueueToDevice(%p, %d): cannot assign, structure is not initialized (call InitializeQueuesPerDevice() first)\n", queue, dev);
    }
    // "Host" device loc id used by CoCoPeLia is 0. See CoCoPeLiaSelectDevice
    int inner_dev_id = (dev == -1) ? 0: dev;

    queuesPerDevice[inner_dev_id].push_back(queue);
}

void UnassignQueueFromDevice(CommandQueue * queue, int dev){
    if(!queuesPerDeviceInitialized){
        error("UnassignQueueFromDevice(%p, %d): cannot unassign, structure is not initialized\n", queue, dev);
    }
    // "Host" device loc id used by CoCoPeLia is 0. See CoCoPeLiaSelectDevice
    int inner_dev_id = (dev == -1) ? 0: dev;

	for(int i = 0; i < queuesPerDevice[inner_dev_id].size(); i++){
		if(queuesPerDevice[inner_dev_id][i] == queue) queuesPerDevice[inner_dev_id].erase(queuesPerDevice[inner_dev_id].begin()+i);
	}
}

void DeviceSynchronize(){
    if(!queuesPerDeviceInitialized){
        // error("DeviceSynchronize(): cannot synchronize queues, structure is not initialized (call InitializeQueuesPerDevice() first)\n");
        return;
    }

    int dev = -1;
    cudaError_t err = cudaGetDevice(&dev);
    massert(cudaSuccess == err,"DeviceSynchronize(): cudaGetDevice failed - %s\n", cudaGetErrorString(err));

    for(int i = 0; i < queuesPerDevice[dev].size(); i++){
        queuesPerDevice[dev][i]->sync_barrier();
    }
}
