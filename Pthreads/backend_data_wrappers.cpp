///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief 
///

#include <cstdio>
#include <typeinfo>
#include <float.h>

#include "pthreads_backend_wrappers.hpp"

// long int CoCoGetMaxDimSqAsset2D(short Asset2DNum, short dsize, long int step, short loc){
// 	size_t free_cuda_mem, max_cuda_mem;
// 	int prev_loc; cudaGetDevice(&prev_loc);
//     /// TODO: Can this ever happen in a healthy scenario?
//     //if (prev_loc != loc) warning("CoCoMalloc: Malloc'ed memory in other device (Previous device: %d, Malloc in: %d)\n", prev_loc, loc);
//     cudaSetDevice(loc);
// 	massert(cudaSuccess == cudaMemGetInfo(&free_cuda_mem, &max_cuda_mem), "backend_get_max_dim_sq_Asset2D: cudaMemGetInfo failed");

// 	// Define the max size of a benchmark kernel to run on this machine.
// 	long int maxDim = (( (long int) sqrt((free_cuda_mem*PROBLEM_GPU_PERCENTAGE/100.0)/(Asset2DNum*dsize))) / step) * step;
// 	cudaSetDevice(prev_loc);
// 	return maxDim;
// }

// long int CoCoGetMaxDimAsset1D(short Asset1DNum, short dsize, long int step, short loc){
// 	size_t free_cuda_mem, max_cuda_mem;
// 	int prev_loc; cudaGetDevice(&prev_loc);
//     /// TODO: Can this ever happen in a healthy scenario?
//     //if (prev_loc != loc) warning("CoCoMalloc: Malloc'ed memory in other device (Previous device: %d, Malloc in: %d)\n", prev_loc, loc);
//     cudaSetDevice(loc);
// 	massert(cudaSuccess == cudaMemGetInfo(&free_cuda_mem, &max_cuda_mem), "backend_get_max_dim_Asset1D: cudaMemGetInfo failed");

// 	long int maxDim = (( (long int) (free_cuda_mem*PROBLEM_GPU_PERCENTAGE/100.0)/(Asset1DNum*dsize)) / step) * step;
// 	cudaSetDevice(prev_loc);
// 	return maxDim;
// }

// short CoCoGetPtrLoc(const void * in_ptr)
// {
// #ifndef CUDA_VER
// #error CUDA_VER Undefined!
// #elif (CUDA_VER == 920)
// 	short loc = -2;
// 	cudaPointerAttributes ptr_att;
// 	if (cudaSuccess != cudaPointerGetAttributes(&ptr_att, in_ptr)) warning("CoCoGetPtrLoc(9.2 version, ptr =%p):\
// 	Pointer not visible to CUDA, host alloc or error\n", in_ptr);
// 	if (ptr_att.memoryType == cudaMemoryTypeHost) loc = -1;
// 	else if (ptr_att.memoryType == cudaMemoryTypeDevice) loc = ptr_att.device;
// 	else if (ptr_att.isManaged) loc = ptr_att.device;
// 	else error("CoCoGetPtrLoc(9.2 version, ptr =%p): Invalid memory type", in_ptr);
// 	return loc;
// #elif (CUDA_VER == 1100)
// 	short loc = -2;
// 	cudaPointerAttributes ptr_att;
// 	if (cudaSuccess != cudaPointerGetAttributes(&ptr_att, in_ptr)) warning("CoCoGetPtrLoc(11.0 version, ptr =%p):\
// 	Pointer not visible to CUDA, host alloc or error\n", in_ptr);
// 	if (ptr_att.type == cudaMemoryTypeHost) loc = -1;
// 	else if (ptr_att.type == cudaMemoryTypeDevice) loc = ptr_att.device;
// 	// TODO: Unified memory is considered available in the GPU as cuBLASXt ( not bad, not great)
// 	else if (ptr_att.type == cudaMemoryTypeManaged) loc = ptr_att.device;
// 	else error("CoCoGetPtrLoc(11.0 version, ptr =%p): Invalid memory type", in_ptr);
// 	return loc;
// #else
// #error Unknown CUDA_VER!
// #endif
// }

// void *gpu_malloc(long long count) {
//   void *ret;
//   massert(cudaMalloc(&ret, count) == cudaSuccess,
//           cudaGetErrorString(cudaGetLastError()));
//   return ret;
// }

// void *pin_malloc(long long count) {
//   void *ret;
//   massert(cudaMallocHost(&ret, count) == cudaSuccess,
//           cudaGetErrorString(cudaGetLastError()));
//   return ret;
// }

void* CoCoMalloc(long long bytes, short loc){
  void *ptr = NULL;

  if (-2 == loc || -1 == loc) {
    //fprintf(stderr, "Allocating %lld bytes to host...\n", bytes);
	ptr = (void*) malloc(bytes);

	if(!ptr) error("CoCoMalloc: malloc failed\n");
  }
  else {
    error("CoCoMalloc: Invalid device id/location (pthread implementation supports only host allocation [loc = -2 || loc = -1])\n");
  }
//   cudaCheckErrors();
  return ptr;
}

void CoCoFree(void * ptr, short loc){

  if (-2 == loc || -1 == loc) free(ptr);
  else {
	error("CoCoFree: Invalid device id/location (pthread implementation supports only host deallocation [loc = -2 || loc = -1])\n");
  }
//   cudaCheckErrors();
}

void CoCoMemcpy(void* dest, void* src, long long bytes, short loc_dest, short loc_src)
{
	massert(-3 < loc_dest && loc_dest < 0, "CoCoMemcpy: Invalid destination device: %d/n", loc_dest);
	massert(-3 < loc_src && loc_src < 0, "CoCoMemcpy: Invalid source device: %d/n", loc_src);

	if (loc_src < 0 && loc_dest < 0) memcpy(dest, src, bytes);
	// cudaCheckErrors();
}

typedef struct CoCoMemcpy_data
{
	void* dest;
	void* src;
	long long bytes;
	short loc_dest;
	short loc_src;
}* CoCoMemcpy_data_p;

void* _CoCoMemcpy(void* input)
{
	CoCoMemcpy_data_p input_unwrapped = (CoCoMemcpy_data_p) input;
	CoCoMemcpy(input_unwrapped->dest, input_unwrapped->src, input_unwrapped->bytes, input_unwrapped->loc_dest, input_unwrapped->loc_src);
	delete input_unwrapped;

	return 0;
}

void CoCoMemcpyAsync(void* dest, void* src, long long bytes, short loc_dest, short loc_src, CQueue_p transfer_queue)
{
	CoCoMemcpy_data_p data = new CoCoMemcpy_data;
	data->dest = dest;
	data->src = src;
	data->bytes = bytes;
	data->loc_dest = loc_dest;
	data->loc_src = loc_src;

	transfer_queue->add_host_func((void*) &_CoCoMemcpy, (void*) data);
}

// void CoCoMemcpy2D(void* dest, long int ldest, void* src, long int ldsrc, long int rows, long int cols, short elemSize, short loc_dest, short loc_src){
// 	short lvl = 6;
// #ifdef DDEBUG
// 	lprintf(lvl, "CoCoMemcpy2D(dest=%p, ldest =%zu, src=%p, ldsrc = %zu, rows = %zu, cols = %zu, elemsize = %d, loc_dest = %d, loc_src = %d)\n",
// 		dest, ldest, src, ldsrc, rows, cols, elemSize, loc_dest, loc_src);
// #endif
// 	int count = 42;
// 	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "CoCoMemcpy2D: cudaGetDeviceCount failed\n");
// 	massert(-3 < loc_dest && loc_dest < count, "CoCoMemcpy2D: Invalid destination device: %d\n", loc_dest);
// 	massert(-3 < loc_src && loc_src < count, "CoCoMemcpy2D: Invalid source device: %d\n", loc_src);

// 	enum cudaMemcpyKind kind;
// 	if (loc_src < 0 && loc_dest < 0) kind = cudaMemcpyHostToHost;
// 	else if (loc_dest < 0) kind = cudaMemcpyDeviceToHost;
// 	else if (loc_src < 0) kind = cudaMemcpyHostToDevice;
// 	else kind = cudaMemcpyDeviceToDevice;

// 	if (loc_src == loc_dest) warning("CoCoMemcpy2D(dest=%p, ldest =%zu, src=%p, ldsrc = %zu, rows=%zu, cols=%zu, elemSize =%d, loc_dest=%d, loc_src=%d): Source location matches destination\n",
// 	dest, ldest, src, ldsrc, rows, cols, elemSize, loc_dest, loc_src);
// 	massert(cudaSuccess == cudaMemcpy2D(dest, ldest*elemSize, src, ldsrc*elemSize, rows*elemSize, cols, kind),
// 	"CoCoMemcpy2D: cudaMemcpy2D failed\n");
// 	//if (loc_src == -1 && loc_dest >=0) massert(CUBLAS_STATUS_SUCCESS == cublasSetMatrix(rows, cols, elemSize, src, ldsrc, dest, ldest), "CoCoMemcpy2DAsync: cublasSetMatrix failed\n");
// 	//else if (loc_src >=0 && loc_dest == -1) massert(CUBLAS_STATUS_SUCCESS == cublasGetMatrix(rows, cols, elemSize, src, ldsrc, dest, ldest),  "CoCoMemcpy2DAsync: cublasGetMatrix failed");

// }
// void CoCMempy2DAsyncWrap3D(void* dest, long int ldest, void* src, long int ldsrc, long int rows, long int cols, short elemSize, short loc_dest, short loc_src, CQueue_p transfer_queue){
// 	// Convert 2d input (as CoCoMemcpy2DAsync) to 3D for ...reasons.
// 	enum cudaMemcpyKind kind = cudaMemcpyDefault;
// 	cudaStream_t stream = *((cudaStream_t*)transfer_queue->cqueue_backend_ptr);
// 	cudaMemcpy3DParms* cudaMemcpy3DParms_p = (cudaMemcpy3DParms*) calloc(1, sizeof(cudaMemcpy3DParms));
// 	cudaMemcpy3DParms_p->extent = make_cudaExtent(rows*elemSize, cols, 1);
// 	cudaMemcpy3DParms_p->srcPtr = make_cudaPitchedPtr (src, ldsrc*elemSize, rows, cols );
// 	cudaMemcpy3DParms_p->dstPtr = make_cudaPitchedPtr (dest, ldest*elemSize, rows, cols );
// 	massert(cudaSuccess == cudaMemcpy3DAsync ( cudaMemcpy3DParms_p, stream) , "cudaMemcpy3DAsync failed\n");
// }

// void CoCoMemcpy2DAsync(void* dest, long int ldest, void* src, long int ldsrc, long int rows, long int cols, short elemSize, short loc_dest, short loc_src, CQueue_p transfer_queue){
// 	short lvl = 6;
// #ifdef DDEBUG
// 	lprintf(lvl, "CoCoMemcpy2DAsync(dest=%p, ldest =%zu, src=%p, ldsrc = %zu, rows = %zu, cols = %zu, elemsize = %d, loc_dest = %d, loc_src = %d)\n",
// 		dest, ldest, src, ldsrc, rows, cols, elemSize, loc_dest, loc_src);
// #endif

// 	cudaStream_t stream = *((cudaStream_t*)transfer_queue->cqueue_backend_ptr);

// 	int count = 42;
// 	massert(CUBLAS_STATUS_SUCCESS == cudaGetDeviceCount(&count), "CoCoMemcpy2DAsync: cudaGetDeviceCount failed\n");
// 	massert(-2 < loc_dest && loc_dest < count, "CoCoMemcpyAsync2D: Invalid destination device: %d\n", loc_dest);
// 	massert(-2 < loc_src && loc_src < count, "CoCoMemcpyAsync2D: Invalid source device: %d\n", loc_src);

// 	enum cudaMemcpyKind kind;
// 	if (loc_src < 0 && loc_dest < 0) kind = cudaMemcpyHostToHost;
// 	else if (loc_dest < 0) kind = cudaMemcpyDeviceToHost;
// 	else if (loc_src < 0) kind = cudaMemcpyHostToDevice;
// 	else kind = cudaMemcpyDeviceToDevice;

// 	if (loc_src == loc_dest) warning("CoCoMemcpy2DAsync(dest=%p, ldest =%zu, src=%p, ldsrc = %zu, rows=%zu, cols=%zu, elemSize =%d, loc_dest=%d, loc_src=%d): Source location matches destination\n",
// 	dest, ldest, src, ldsrc, rows, cols, elemSize, loc_dest, loc_src);
// 	massert(cudaSuccess == cudaMemcpy2DAsync(dest, ldest*elemSize, src, ldsrc*elemSize,
// 		rows*elemSize, cols, kind, stream),  "CoCoMemcpy2DAsync(dest=%p, ldest =%zu, src=%p, ldsrc = %zu,\
// 			\nrows = %zu, cols = %zu, elemsize = %d, loc_dest = %d, loc_src = %d): cudaMemcpy2DAsync failed\n",
// 			dest, ldest, src, ldsrc, rows, cols, elemSize, loc_dest, loc_src);
// 	//if (loc_src == -1 && loc_dest >=0) massert(CUBLAS_STATUS_SUCCESS == cublasSetMatrixAsync(rows, cols, elemSize, src, ldsrc, dest, ldest, stream), "CoCoMemcpy2DAsync: cublasSetMatrixAsync failed\n");
// 	//else if (loc_src >=0 && loc_dest == -1) massert(CUBLAS_STATUS_SUCCESS == cublasGetMatrixAsync(rows, cols, elemSize, src, ldsrc, dest, ldest, stream),  "CoCoMemcpy2DAsync: cublasGetMatrixAsync failed");
// }

// template<typename VALUETYPE>
// void CoCoVecInit(VALUETYPE *vec, long long length, int seed, short loc)
// {
//   int count = 42;
//   cudaGetDeviceCount(&count);
//   if (!vec) error("CoCoVecInit: vec is not allocated (correctly)\n");
//   if (-2 == loc || -1 == loc) CoCoParallelVecInitHost(vec, length, seed);
//   else if (loc >= count || loc < 0) error("CoCoVecInit: Invalid device id/location\n");
//   else {
// 	int prev_loc; cudaGetDevice(&prev_loc);

// 	//if (prev_loc != loc) warning("CoCoVecInit: Initialized vector in other device (Previous device: %d, init in: %d)\n", prev_loc, loc);
//     	cudaSetDevice(loc);
// 	curandGenerator_t gen;
// 	/* Create pseudo-random number generator */
// 	massert(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) == cudaSuccess,
//           cudaGetErrorString(cudaGetLastError()));
// 	/* Set seed */
// 	massert(curandSetPseudoRandomGeneratorSeed(gen, seed) == cudaSuccess,
//           cudaGetErrorString(cudaGetLastError()));
// 	if (typeid(VALUETYPE) == typeid(float))
// 	  massert(curandGenerateUniform(gen, (float*) vec, length) == cudaSuccess,
//             cudaGetErrorString(cudaGetLastError()));
// 	else if (typeid(VALUETYPE) == typeid(double))
// 	  massert(curandGenerateUniformDouble(gen, (double*) vec, length) == cudaSuccess,
//             cudaGetErrorString(cudaGetLastError()));
// 	cudaCheckErrors();
//     	if (prev_loc != loc){
// 		//warning("CoCoVecInit: Reseting device to previous: %d\n", prev_loc);
// 		cudaSetDevice(prev_loc);
// 	}
//   }
//   cudaCheckErrors();
// }

// template void CoCoVecInit<double>(double *vec, long long length, int seed, short loc);
// template void CoCoVecInit<float>(float *vec, long long length, int seed, short loc);

// template<typename VALUETYPE>
// void CoCoParallelVecInitHost(VALUETYPE *vec, long long length, int seed)
// {
// 	srand(seed);
// 	//#pragma omp parallel for
// 	for (long long i = 0; i < length; i++) vec[i] = (VALUETYPE) Drandom();
// }

// template void CoCoParallelVecInitHost<double>(double *vec, long long length, int seed);
// template void CoCoParallelVecInitHost<float>(float *vec, long long length, int seed);

// void CoCoEnableLinks(short target_dev_i, short num_devices){
// 	short lvl = 2;
// #ifdef DEBUG
// 	lprintf(lvl-1, "|-----> CoCoPeLiaEnableGPUPeer(%d,%d)\n", target_dev_i, num_devices);
// #endif
// #ifdef TEST
// 	lprintf(lvl-1, "|-----> CoCoPeLiaEnableGPUPeer\n");
// 	double cpu_timer = csecond();
// #endif
// 	int dev_id_target = deidxize(target_dev_i);
// 	CoCoPeLiaSelectDevice(dev_id_target);
// 	for(int j=0; j<num_devices;j++){
// 		int dev_id_current = deidxize(j);
// 		if (dev_id_target == dev_id_current || dev_id_target == -1 || dev_id_current == -1) continue;
// 		int can_access_peer;
// 		massert(cudaSuccess == cudaDeviceCanAccessPeer(&can_access_peer, dev_id_target, dev_id_current), "CoCopeLiaDgemm: cudaDeviceCanAccessPeer failed\n");
// 		if(can_access_peer){
// 			cudaError_t check_peer = cudaDeviceEnablePeerAccess(dev_id_current, 0);
// 			if(check_peer == cudaSuccess){ ;
// #ifdef DEBUG
// 				lprintf(lvl, "Enabled Peer access for dev %d to dev %d\n", dev_id_target, dev_id_current);
// #endif
// 			}
// 			else if (check_peer == cudaErrorPeerAccessAlreadyEnabled){
// 				cudaGetLastError();
// #ifdef DEBUG
// 				lprintf(lvl, "Peer access already enabled for dev %d to dev %d\n", dev_id_target, dev_id_current);
// #endif
// 			}
// 			else error("Enabling Peer access failed for %d to dev %d\n", dev_id_target, dev_id_current);
// 		}
// 	}
// #ifdef TEST
// 	cpu_timer = csecond() - cpu_timer;
// 	lprintf(lvl, "Utiilizing Peer access for dev %d -> t_enable =%lf ms\n", dev_id_target, 1000*cpu_timer);
// 	cpu_timer = csecond();
// 	lprintf(lvl-1, "<-----|\n");
// #endif
// #ifdef DEBUG
// 	lprintf(lvl-1, "<-----|\n");
// #endif
// }