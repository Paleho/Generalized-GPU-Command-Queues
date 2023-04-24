///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief Some HIP function calls with added error-checking
///			  (HIPified version of original CUDA code)
///

#include <cstdio>
#include <typeinfo>
#include <float.h>

#include "backend_wrappers.hpp"

/*void print_devices() {
  hipDeviceProp_t properties;
  int nDevices = 0;
  massert(HIPBLAS_STATUS_SUCCESS == hipGetDeviceCount(&nDevices), "print_devices: hipGetDeviceCount failed");
  printf("Found %d Devices: \n\n", nDevices);
  for (int i = 0; i < nDevices; i++) {
    hipGetDeviceProperties(&properties, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", properties.name);
    printf("  Memory Clock Rate (MHz): %d\n",
           properties.memoryClockRate / 1024);
    printf("  Memory Bus Width (bits): %d\n", properties.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * properties.memoryClockRate * (properties.memoryBusWidth / 8) /
               1.0e6);
    if (properties.major >= 3)
      printf("  Unified Memory support: YES\n\n");
    else
      printf("  Unified Memory support: NO\n\n");
  }
}
*/

void CoCoSyncCheckErr(){
  hipError_t errSync = hipDeviceSynchronize();
  if (errSync != hipSuccess)
    printf("Sync kernel error: %s\n", hipGetErrorString(errSync));
}

void CoCoASyncCheckErr(){
  hipError_t errAsync = hipGetLastError();
  if (errAsync != hipSuccess)
    printf("Async kernel error: %s\n", hipGetErrorString(errAsync));
}

void cudaCheckErrors(){
	//CoCoASyncCheckErr();
	CoCoSyncCheckErr();
}

int CoCoPeLiaGetDevice(){
  int dev_id = -1;
  hipError_t err = hipGetDevice(&dev_id);
  massert(hipSuccess == err,
    "CoCoPeLiaGetDevice: hipGetDevice failed - %s\n", hipGetErrorString(err));
  return dev_id;
}

void CoCoPeLiaSelectDevice(short dev_id){
  int dev_count;
  hipError_t err = hipGetDeviceCount(&dev_count);
  if(dev_id >= 0 && dev_id < dev_count){
  hipError_t err = hipSetDevice(dev_id);
  massert(hipSuccess == err,
    "CoCoPeLiaSelectDevice(%d): hipSetDevice(%d) failed - %s\n", dev_id, dev_id, hipGetErrorString(err));
  }
  else if(dev_id == -1){  /// "Host" device loc id used by CoCoPeLia
    hipSetDevice(0);
  }
  else error("CoCoPeLiaSelectDevice(%d): invalid dev_id\n", dev_id);
}

void CoCoPeLiaDevGetMemInfo(long long* free_dev_mem, long long* max_dev_mem){
  size_t free_dev_mem_tmp, max_dev_mem_tmp;
    int tmp_dev_id;
    hipError_t err = hipGetDevice(&tmp_dev_id);
    // TODO: For the CPU this function returns device 0 memory availability. Its a feature not a bug.
    massert(hipSuccess == err,
      "CoCoPeLiaDevGetMemInfo: hipGetDevice failed - %s\n", hipGetErrorString(err));
    err = hipMemGetInfo(&free_dev_mem_tmp, &max_dev_mem_tmp);
  	massert(hipSuccess == err,
      "CoCoPeLiaDevGetMemInfo: hipMemGetInfo failed - %s\n", hipGetErrorString(err));
    *free_dev_mem = (long long) free_dev_mem_tmp;
    *max_dev_mem = (long long) max_dev_mem_tmp;
}

void TransposeTranslate(char TransChar, CBLAS_TRANSPOSE* cblasFlag, hipblasOperation_t* cuBLASFlag, long int* ldim, long int dim1, long int dim2){
	if (TransChar == 'N'){
 		*cblasFlag = CblasNoTrans;
 		*cuBLASFlag = HIPBLAS_OP_N;
		*ldim = dim1;
	}
	else if (TransChar == 'T'){
 		*cblasFlag = CblasTrans;
 		*cuBLASFlag = HIPBLAS_OP_T;
		*ldim = dim2;
	}
	else if (TransChar == 'C'){
 		*cblasFlag = CblasConjTrans;
 		*cuBLASFlag = HIPBLAS_OP_C;
		*ldim = dim2;
	}
	else error("TransposeTranslate: %c is an invalid Trans flag", TransChar);
}


hipblasOperation_t OpCblasToCublas(CBLAS_TRANSPOSE src)
{
	if(src == CblasNoTrans) return HIPBLAS_OP_N;
	else if(src == CblasTrans) return HIPBLAS_OP_T;
	else if(src == CblasConjTrans) return HIPBLAS_OP_C;
	else error("OpCblasToCublas: Invalid Op\n");
}

hipblasOperation_t OpCharToCublas(char src)
{
	if(src == 'N') return HIPBLAS_OP_N;
	else if(src == 'T') return HIPBLAS_OP_T;
	else if(src == 'C') return HIPBLAS_OP_C;
	else error("OpCharToCublas: Invalid Op: %c\n", src);
}

CBLAS_TRANSPOSE OpCharToCblas(char src)
{
	if(src == 'N') return CblasNoTrans;
	else if(src == 'T') return CblasTrans;
	else if(src == 'C') return CblasConjTrans;
	else error("OpCharToCblas: Invalid Op: %c\n", src);
}

CBLAS_TRANSPOSE OpCublasToCblas(hipblasOperation_t src)
{
	if(src == HIPBLAS_OP_N) return CblasNoTrans;
	else if(src == HIPBLAS_OP_T) return CblasTrans;
	else if(src == HIPBLAS_OP_C) return CblasConjTrans;
	else error("OpCublasToCblas: Invalid Op\n");
}

char PrintCublasOp(hipblasOperation_t src)
{

	if(src == HIPBLAS_OP_N) return 'N';
	else if(src == HIPBLAS_OP_T) return 'T';
	else if(src == HIPBLAS_OP_C) return 'C';
	else error("PrintCublasOp: Invalid Op\n");
}
