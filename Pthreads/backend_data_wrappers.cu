///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief 
///

#include <cstdio>
#include <typeinfo>
#include <float.h>
#include <curand.h>

#include "pthreads_backend_wrappers.hpp"

long int CoCoGetMaxDimSqAsset2D(short Asset2DNum, short dsize, long int step, short loc){
	size_t free_cuda_mem, max_cuda_mem;
	int prev_loc; cudaGetDevice(&prev_loc);
    /// TODO: Can this ever happen in a healthy scenario?
    //if (prev_loc != loc) warning("CoCoMalloc: Malloc'ed memory in other device (Previous device: %d, Malloc in: %d)\n", prev_loc, loc);
    cudaSetDevice(loc);
	massert(cudaSuccess == cudaMemGetInfo(&free_cuda_mem, &max_cuda_mem), "backend_get_max_dim_sq_Asset2D: cudaMemGetInfo failed");

	// Define the max size of a benchmark kernel to run on this machine.
	long int maxDim = (( (long int) sqrt((free_cuda_mem*PROBLEM_GPU_PERCENTAGE/100.0)/(Asset2DNum*dsize))) / step) * step;
	cudaSetDevice(prev_loc);
	return maxDim;
}

long int CoCoGetMaxDimAsset1D(short Asset1DNum, short dsize, long int step, short loc){
	size_t free_cuda_mem, max_cuda_mem;
	int prev_loc; cudaGetDevice(&prev_loc);
    /// TODO: Can this ever happen in a healthy scenario?
    //if (prev_loc != loc) warning("CoCoMalloc: Malloc'ed memory in other device (Previous device: %d, Malloc in: %d)\n", prev_loc, loc);
    cudaSetDevice(loc);
	massert(cudaSuccess == cudaMemGetInfo(&free_cuda_mem, &max_cuda_mem), "backend_get_max_dim_Asset1D: cudaMemGetInfo failed");

	long int maxDim = (( (long int) (free_cuda_mem*PROBLEM_GPU_PERCENTAGE/100.0)/(Asset1DNum*dsize)) / step) * step;
	cudaSetDevice(prev_loc);
	return maxDim;
}

short CoCoGetPtrLoc(const void * in_ptr)
{
// This is legacy code for CUDA 9.2 <<. It should not be used due to CUDA ptr_att back-end struct changes in latest versions
#ifdef CUDA_9_WRAPPER_MESS
	short loc = -2;
	cudaPointerAttributes ptr_att;
	if (cudaSuccess != cudaPointerGetAttributes(&ptr_att, in_ptr)) warning("CoCoGetPtrLoc(9.2 version, ptr =%p):\
	Pointer not visible to CUDA, host alloc or error\n", in_ptr);
	if (ptr_att.memoryType == cudaMemoryTypeHost) loc = -1;
	else if (ptr_att.memoryType == cudaMemoryTypeDevice) loc = ptr_att.device;
	else if (ptr_att.isManaged) loc = ptr_att.device;
	else error("CoCoGetPtrLoc(9.2 version, ptr =%p): Invalid memory type", in_ptr);
	return loc;
#else
	short loc = -2;
	cudaPointerAttributes ptr_att;
	if (cudaSuccess != cudaPointerGetAttributes(&ptr_att, in_ptr)) warning("CoCoGetPtrLoc(11.0 version, ptr =%p):\
	Pointer not visible to CUDA, host alloc or error\n", in_ptr);
	if (ptr_att.type == cudaMemoryTypeHost) loc = -1;
	else if (ptr_att.type == cudaMemoryTypeDevice) loc = ptr_att.device;
	// TODO: Unified memory is considered available in the GPU as cuBLASXt ( not bad, not great)
	else if (ptr_att.type == cudaMemoryTypeManaged) loc = ptr_att.device;
	else error("CoCoGetPtrLoc(11.0 version, ptr =%p): Invalid memory type", in_ptr);
	return loc;
#endif
}

void *gpu_malloc(long long count) {
  void *ret;
  massert(cudaMalloc(&ret, count) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
  return ret;
}

void *pin_malloc(long long count) {
  void *ret;
  massert(cudaMallocHost(&ret, count) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
  return ret;
}

void* CoCoMalloc(long long bytes, short loc){
  int count = 42;
  massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "CoCoMalloc: cudaGetDeviceCount failed");
  void *ptr = NULL;

  if (-2 == loc) {
    //fprintf(stderr, "Allocating %lld bytes to host...\n", bytes);
	ptr = (void*) malloc(bytes);
  }
  else if (-1 == loc) {
    //fprintf(stderr, "Allocating %lld bytes to pinned host...\n", bytes);
	ptr = pin_malloc(bytes);

  } 
  else if (loc >= count || loc < 0)
    error("CoCoMalloc: Invalid device id/location\n");
  else {
    //fprintf(stderr, "Allocating %lld bytes to device(%d)...\n", bytes, loc);
    int prev_loc; cudaGetDevice(&prev_loc);
    /// TODO: Can this ever happen in a healthy scenario?
    //if (prev_loc != loc) warning("CoCoMalloc: Malloc'ed memory in other device (Previous device: %d, Malloc in: %d)\n", prev_loc, loc);
    cudaSetDevice(loc);
    ptr = gpu_malloc(bytes);

	cudaCheckErrors();
    	if (prev_loc != loc){
		//warning("CoCoMalloc: Reseting device to previous: %d\n", prev_loc);
		cudaSetDevice(prev_loc);
	}
  }
  cudaCheckErrors();
  return ptr;
}

void gpu_free(void *gpuptr) {
  massert(cudaFree(gpuptr) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
}

void pin_free(void *gpuptr) {
  massert(cudaFreeHost(gpuptr) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
}

void CoCoFree(void * ptr, short loc){
  int count = 42;
  massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "CoCoFree: cudaGetDeviceCount failed");

  if (-2 == loc) free(ptr);
  else if (-1 == loc) pin_free(ptr);
  else if (loc >= count || loc < 0) error("CoCoFree: Invalid device id/location\n");
  else {
	int prev_loc; cudaGetDevice(&prev_loc);
	//if (prev_loc != loc) warning("CoCoFree: Freed memory in other device (Previous device: %d, Free in: %d)\n", prev_loc, loc);
    	cudaSetDevice(loc);
	gpu_free(ptr);
	cudaCheckErrors();
    	if (prev_loc != loc){
		//warning("CoCoFree: Reseting device to previous: %d\n", prev_loc);
		cudaSetDevice(prev_loc);
	}
  }
  cudaCheckErrors();
}

void CoCoMemcpy(void* dest, void* src, long long bytes, short loc_dest, short loc_src)
{
	int count = 42;
	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "CoCoMemcpy: cudaGetDeviceCount failed");
	massert(-3 < loc_dest && loc_dest < count, "CoCoMemcpy: Invalid destination device: %d/n", loc_dest);
	massert(-3 < loc_src && loc_src < count, "CoCoMemcpy: Invalid source device: %d/n", loc_src);

	enum cudaMemcpyKind kind = cudaMemcpyHostToHost;
	if (loc_src < 0 && loc_dest < 0) memcpy(dest, src, bytes);
	else if (loc_dest < 0) kind = cudaMemcpyDeviceToHost;
	else if (loc_src < 0) kind = cudaMemcpyHostToDevice;
	else kind = cudaMemcpyDeviceToDevice;

#ifdef DEBUG
	if (loc_src == loc_dest) warning("CoCoMemcpy(dest=%p, src=%p, bytes=%lld, loc_dest=%d, loc_src=%d): Source location matches destination\n",
	dest, src, bytes, loc_dest, loc_src);
#endif
	massert(CUBLAS_STATUS_SUCCESS == cudaMemcpy(dest, src, bytes, kind), "CoCoMemcpy: cudaMemcpy from device src=%d to dest=%d failed\n", loc_src, loc_dest);
	cudaCheckErrors();
}

typedef struct CoCoMemcpy_data
{
	queue_data_p q_data;
	void* dest;
	void* src;
	long long bytes;
	short loc_dest;
	short loc_src;
}* CoCoMemcpy_data_p;

void _CoCoMemcpyAsync(void* input)
{
	CoCoMemcpy_data_p input_unwrapped = (CoCoMemcpy_data_p) input;

	queue_data_p queue_backend_data = input_unwrapped->q_data;

	get_lock_q(&queue_backend_data->queueLock);
		// Get stream and increase stream index
		int current_stream_ctr = queue_backend_data->stream_ctr;
		queue_backend_data->stream_ctr = (current_stream_ctr + 1) % STREAM_POOL_SZ;
	release_lock_q(&queue_backend_data->queueLock);

	int count = 42;
	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "_CoCoMemcpyAsync: cudaGetDeviceCount failed\n");
	massert(-2 < input_unwrapped->loc_dest && input_unwrapped->loc_dest < count, "_CoCoMemcpyAsync: Invalid destination device: %d\n", input_unwrapped->loc_dest);
	massert(-2 < input_unwrapped->loc_src && input_unwrapped->loc_src < count, "_CoCoMemcpyAsync: Invalid source device: %d\n", input_unwrapped->loc_src);

	enum cudaMemcpyKind kind;
	if (input_unwrapped->loc_src < 0 && input_unwrapped->loc_dest < 0) kind = cudaMemcpyHostToHost;
	else if (input_unwrapped->loc_dest < 0) kind = cudaMemcpyDeviceToHost;
	else if (input_unwrapped->loc_src < 0) kind = cudaMemcpyHostToDevice;
	else kind = cudaMemcpyDeviceToDevice;

	if (input_unwrapped->loc_src == input_unwrapped->loc_dest) warning("_CoCoMemcpyAsync(dest=%p, src=%p, bytes=%lld, loc_dest=%d, loc_src=%d): Source location matches destination\n",
	input_unwrapped->dest, input_unwrapped->src, input_unwrapped->bytes, input_unwrapped->loc_dest, input_unwrapped->loc_src);
	massert(cudaSuccess == cudaMemcpyAsync(input_unwrapped->dest, input_unwrapped->src, input_unwrapped->bytes, kind, queue_backend_data->stream_pool[current_stream_ctr]),
	"_CoCoMemcpyAsync: cudaMemcpyAsync failed\n");

	massert(cudaSuccess == cudaStreamSynchronize(queue_backend_data->stream_pool[current_stream_ctr]), "_CoCoMemcpyAsync: stream sync failed\n");

	delete input_unwrapped;

	return;
}

void CoCoMemcpyAsync(void* dest, void* src, long long bytes, short loc_dest, short loc_src, CQueue_p transfer_queue)
{
	CoCoMemcpy_data_p data = new CoCoMemcpy_data;
	data->q_data = (queue_data_p) transfer_queue->cqueue_backend_data;
	data->dest = dest;
	data->src = src;
	data->bytes = bytes;
	data->loc_dest = loc_dest;
	data->loc_src = loc_src;

	transfer_queue->add_host_func((void*) &_CoCoMemcpyAsync, (void*) data);
	return;
}

void CoCoMemcpy2D(void* dest, long int ldest, void* src, long int ldsrc, long int rows, long int cols, short elemSize, short loc_dest, short loc_src){
	short lvl = 6;
#ifdef DDEBUG
	lprintf(lvl, "CoCoMemcpy2D(dest=%p, ldest =%zu, src=%p, ldsrc = %zu, rows = %zu, cols = %zu, elemsize = %d, loc_dest = %d, loc_src = %d)\n",
		dest, ldest, src, ldsrc, rows, cols, elemSize, loc_dest, loc_src);
#endif
	int count = 42;
	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "CoCoMemcpy2D: cudaGetDeviceCount failed\n");
	massert(-3 < loc_dest && loc_dest < count, "CoCoMemcpy2D: Invalid destination device: %d\n", loc_dest);
	massert(-3 < loc_src && loc_src < count, "CoCoMemcpy2D: Invalid source device: %d\n", loc_src);

	enum cudaMemcpyKind kind;
	if (loc_src < 0 && loc_dest < 0) kind = cudaMemcpyHostToHost;
	else if (loc_dest < 0) kind = cudaMemcpyDeviceToHost;
	else if (loc_src < 0) kind = cudaMemcpyHostToDevice;
	else kind = cudaMemcpyDeviceToDevice;

	if (loc_src == loc_dest) warning("CoCoMemcpy2D(dest=%p, ldest =%zu, src=%p, ldsrc = %zu, rows=%zu, cols=%zu, elemSize =%d, loc_dest=%d, loc_src=%d): Source location matches destination\n",
	dest, ldest, src, ldsrc, rows, cols, elemSize, loc_dest, loc_src);
	massert(cudaSuccess == cudaMemcpy2D(dest, ldest*elemSize, src, ldsrc*elemSize, rows*elemSize, cols, kind),
	"CoCoMemcpy2D: cudaMemcpy2D failed\n");
	//if (loc_src == -1 && loc_dest >=0) massert(CUBLAS_STATUS_SUCCESS == cublasSetMatrix(rows, cols, elemSize, src, ldsrc, dest, ldest), "CoCoMemcpy2DAsync: cublasSetMatrix failed\n");
	//else if (loc_src >=0 && loc_dest == -1) massert(CUBLAS_STATUS_SUCCESS == cublasGetMatrix(rows, cols, elemSize, src, ldsrc, dest, ldest),  "CoCoMemcpy2DAsync: cublasGetMatrix failed");

	return;
}

void CoCMempy2DAsyncWrap3D(void* dest, long int ldest, void* src, long int ldsrc, long int rows, long int cols, short elemSize, short loc_dest, short loc_src, CQueue_p transfer_queue){
	// Convert 2d input (as CoCoMemcpy2DAsync) to 3D for ...reasons.
	enum cudaMemcpyKind kind = cudaMemcpyDefault;
	cudaStream_t stream = *((cudaStream_t*)transfer_queue->cqueue_backend_ptr);
	cudaMemcpy3DParms* cudaMemcpy3DParms_p = (cudaMemcpy3DParms*) calloc(1, sizeof(cudaMemcpy3DParms));
	cudaMemcpy3DParms_p->extent = make_cudaExtent(rows*elemSize, cols, 1);
	cudaMemcpy3DParms_p->srcPtr = make_cudaPitchedPtr (src, ldsrc*elemSize, rows, cols );
	cudaMemcpy3DParms_p->dstPtr = make_cudaPitchedPtr (dest, ldest*elemSize, rows, cols );
	massert(cudaSuccess == cudaMemcpy3DAsync ( cudaMemcpy3DParms_p, stream) , "cudaMemcpy3DAsync failed\n");
}

typedef struct CoCoMemcpy2D_data
{
	queue_data_p q_data;
	void* dest;
	long int ldest;
	void* src;
	long int ldsrc;
	long int rows; 
	long int cols;
	short elemSize;
	short loc_dest;
	short loc_src;
}* CoCoMemcpy2D_data_p;

void _CoCoMemcpy2DAsync(void* input)
{
	CoCoMemcpy2D_data_p input_unwrapped = (CoCoMemcpy2D_data_p) input;

	queue_data_p queue_backend_data = input_unwrapped->q_data;

	get_lock_q(&queue_backend_data->queueLock);
		// Get stream and increase stream index
		int current_stream_ctr = queue_backend_data->stream_ctr;
		queue_backend_data->stream_ctr = (current_stream_ctr + 1) % STREAM_POOL_SZ;
	release_lock_q(&queue_backend_data->queueLock);

	int count = 42;
	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "_CoCoMemcpy2DAsync: cudaGetDeviceCount failed\n");
	massert(-2 < input_unwrapped->loc_dest && input_unwrapped->loc_dest < count, "_CoCoMemcpy2DAsync: Invalid destination device: %d\n", input_unwrapped->loc_dest);
	massert(-2 < input_unwrapped->loc_src && input_unwrapped->loc_src < count, "_CoCoMemcpy2DAsync: Invalid source device: %d\n", input_unwrapped->loc_src);

	enum cudaMemcpyKind kind;
	if (input_unwrapped->loc_src < 0 && input_unwrapped->loc_dest < 0) kind = cudaMemcpyHostToHost;
	else if (input_unwrapped->loc_dest < 0) kind = cudaMemcpyDeviceToHost;
	else if (input_unwrapped->loc_src < 0) kind = cudaMemcpyHostToDevice;
	else kind = cudaMemcpyDeviceToDevice;

	if (input_unwrapped->loc_src == input_unwrapped->loc_dest) warning("_CoCoMemcpy2DAsync(dest=%p, ldest =%zu, src=%p, ldsrc = %zu, rows=%zu, cols=%zu, elemSize =%d, loc_dest=%d, loc_src=%d): Source location matches destination\n",
	input_unwrapped->dest, input_unwrapped->ldest, input_unwrapped->src, input_unwrapped->ldsrc, input_unwrapped->rows, input_unwrapped->cols, input_unwrapped->elemSize, input_unwrapped->loc_dest, input_unwrapped->loc_src);
	massert(cudaSuccess == cudaMemcpy2DAsync(input_unwrapped->dest, (input_unwrapped->ldest)*(input_unwrapped->elemSize), input_unwrapped->src, (input_unwrapped->ldsrc)*(input_unwrapped->elemSize),
		(input_unwrapped->rows)*(input_unwrapped->elemSize), input_unwrapped->cols, kind, queue_backend_data->stream_pool[current_stream_ctr]),  "_CoCoMemcpy2DAsync(dest=%p, ldest =%zu, src=%p, ldsrc = %zu,\
			\nrows = %zu, cols = %zu, elemsize = %d, loc_dest = %d, loc_src = %d): cudaMemcpy2DAsync failed\n",
			input_unwrapped->dest, input_unwrapped->ldest, input_unwrapped->src, input_unwrapped->ldsrc, input_unwrapped->rows, input_unwrapped->cols, input_unwrapped->elemSize, input_unwrapped->loc_dest, input_unwrapped->loc_src);

	massert(cudaSuccess == cudaStreamSynchronize(queue_backend_data->stream_pool[current_stream_ctr]), "_CoCoMemcpy2DAsync: stream sync failed\n");

	delete input_unwrapped;

	return;
}

void CoCoMemcpy2DAsync(void* dest, long int ldest, void* src, long int ldsrc, long int rows, long int cols, short elemSize, short loc_dest, short loc_src, CQueue_p transfer_queue){
	short lvl = 6;
#ifdef DDEBUG
	lprintf(lvl, "CoCoMemcpy2DAsync(dest=%p, ldest =%zu, src=%p, ldsrc = %zu, rows = %zu, cols = %zu, elemsize = %d, loc_dest = %d, loc_src = %d)\n",
		dest, ldest, src, ldsrc, rows, cols, elemSize, loc_dest, loc_src);
#endif

	CoCoMemcpy2D_data_p data = new CoCoMemcpy2D_data;
	data->q_data = (queue_data_p) transfer_queue->cqueue_backend_data;
	data->dest = dest;
	data->ldest = ldest;
	data->src = src;
	data->ldsrc = ldsrc;
	data->rows = rows;
	data->cols = cols;
	data->elemSize = elemSize;
	data->loc_dest = loc_dest;
	data->loc_src = loc_src;

	transfer_queue->add_host_func((void*) &_CoCoMemcpy2DAsync, (void*) data);

	return;
}

template<typename VALUETYPE>
void CoCoVecInit(VALUETYPE *vec, long long length, int seed, short loc)
{
  int count = 42;
  cudaGetDeviceCount(&count);
  if (!vec) error("CoCoVecInit: vec is not allocated (correctly)\n");
  if (-2 == loc || -1 == loc) CoCoParallelVecInitHost(vec, length, seed);
  else if (loc >= count || loc < 0) error("CoCoVecInit: Invalid device id/location\n");
  else {
	int prev_loc; cudaGetDevice(&prev_loc);

	//if (prev_loc != loc) warning("CoCoVecInit: Initialized vector in other device (Previous device: %d, init in: %d)\n", prev_loc, loc);
    	cudaSetDevice(loc);
	curandGenerator_t gen;
	/* Create pseudo-random number generator */
	massert(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
	/* Set seed */
	massert(curandSetPseudoRandomGeneratorSeed(gen, seed) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
	if (typeid(VALUETYPE) == typeid(float))
	  massert(curandGenerateUniform(gen, (float*) vec, length) == cudaSuccess,
            cudaGetErrorString(cudaGetLastError()));
	else if (typeid(VALUETYPE) == typeid(double))
	  massert(curandGenerateUniformDouble(gen, (double*) vec, length) == cudaSuccess,
            cudaGetErrorString(cudaGetLastError()));
	cudaCheckErrors();
    	if (prev_loc != loc){
		//warning("CoCoVecInit: Reseting device to previous: %d\n", prev_loc);
		cudaSetDevice(prev_loc);
	}
  }
  cudaCheckErrors();
}

template void CoCoVecInit<double>(double *vec, long long length, int seed, short loc);
template void CoCoVecInit<float>(float *vec, long long length, int seed, short loc);

template<typename VALUETYPE>
void CoCoParallelVecInitHost(VALUETYPE *vec, long long length, int seed)
{
	srand(seed);
	//#pragma omp parallel for
	for (long long i = 0; i < length; i++) vec[i] = (VALUETYPE) Drandom();
}

template void CoCoParallelVecInitHost<double>(double *vec, long long length, int seed);
template void CoCoParallelVecInitHost<float>(float *vec, long long length, int seed);

void CoCoEnableLinks(short target_dev_i, short num_devices){
	short lvl = 2;
#ifdef DEBUG
	lprintf(lvl-1, "|-----> CoCoPeLiaEnableGPUPeer(%d,%d)\n", target_dev_i, num_devices);
#endif
#ifdef TEST
	lprintf(lvl-1, "|-----> CoCoPeLiaEnableGPUPeer\n");
	double cpu_timer = csecond();
#endif
	int dev_id_target = deidxize(target_dev_i);
	CoCoPeLiaSelectDevice(dev_id_target);
	for(int j=0; j<num_devices;j++){
		int dev_id_current = deidxize(j);
		if (dev_id_target == dev_id_current || dev_id_target == -1 || dev_id_current == -1) continue;
		int can_access_peer;
		massert(cudaSuccess == cudaDeviceCanAccessPeer(&can_access_peer, dev_id_target, dev_id_current), "CoCopeLiaDgemm: cudaDeviceCanAccessPeer failed\n");
		if(can_access_peer){
			cudaError_t check_peer = cudaDeviceEnablePeerAccess(dev_id_current, 0);
			if(check_peer == cudaSuccess){ ;
#ifdef DEBUG
				lprintf(lvl, "Enabled Peer access for dev %d to dev %d\n", dev_id_target, dev_id_current);
#endif
			}
			else if (check_peer == cudaErrorPeerAccessAlreadyEnabled){
				cudaGetLastError();
#ifdef DEBUG
				lprintf(lvl, "Peer access already enabled for dev %d to dev %d\n", dev_id_target, dev_id_current);
#endif
			}
			else error("Enabling Peer access failed for %d to dev %d\n", dev_id_target, dev_id_current);
		}
	}
#ifdef TEST
	cpu_timer = csecond() - cpu_timer;
	lprintf(lvl, "Utiilizing Peer access for dev %d -> t_enable =%lf ms\n", dev_id_target, 1000*cpu_timer);
	cpu_timer = csecond();
	lprintf(lvl-1, "<-----|\n");
#endif
#ifdef DEBUG
	lprintf(lvl-1, "<-----|\n");
#endif
}
