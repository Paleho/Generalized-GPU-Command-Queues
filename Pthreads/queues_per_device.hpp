#include <vector>
#include <unihelpers.hpp>

extern std::vector<CommandQueue *> * queuesPerDevice;

extern bool queuesPerDeviceInitialized;

void InitializeQueuesPerDevice();

void UninitializeQueuesPerDevice();

void AssignQueueToDevice(CommandQueue * queue, int dev);

void UnassignQueueFromDevice(CommandQueue * queue, int dev);

void DeviceSynchronize();
