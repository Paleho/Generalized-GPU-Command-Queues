///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief 
///
#include <cstdio>
#include <typeinfo>
#include <float.h>

#include "pthreads_backend_wrappers.hpp"

void CoCoSyncCheckErr(){
	printf("CoCoSyncCheckErr: not implemented for pthreads\n");
}

void CoCoASyncCheckErr(){
	printf("CoCoASyncCheckErr: not implemented for pthreads\n");
}

void cudaCheckErrors(){
	//CoCoASyncCheckErr();
	CoCoSyncCheckErr();
}

int CoCoPeLiaGetDevice(){
  int dev_id = -1;
  return dev_id;
}

void CoCoPeLiaSelectDevice(short dev_id){
  // Does nothing for pthreads
}

void CoCoPeLiaDevGetMemInfo(long long* free_dev_mem, long long* max_dev_mem){
  // Not implemented for pthreads
}

void TransposeTranslate(char TransChar, CBLAS_TRANSPOSE* cblasFlag, cublasOperation_t* cuBLASFlag, long int* ldim, long int dim1, long int dim2){
	if (TransChar == 'N'){
 		*cblasFlag = CblasNoTrans;
 		*cuBLASFlag = CUBLAS_OP_N;
		*ldim = dim1;
	}
	else if (TransChar == 'T'){
 		*cblasFlag = CblasTrans;
 		*cuBLASFlag = CUBLAS_OP_T;
		*ldim = dim2;
	}
	else if (TransChar == 'C'){
 		*cblasFlag = CblasConjTrans;
 		*cuBLASFlag = CUBLAS_OP_C;
		*ldim = dim2;
	}
	else error("TransposeTranslate: %c is an invalid Trans flag", TransChar);
}


cublasOperation_t OpCblasToCublas(CBLAS_TRANSPOSE src)
{
	if(src == CblasNoTrans) return CUBLAS_OP_N;
	else if(src == CblasTrans) return CUBLAS_OP_T;
	else if(src == CblasConjTrans) return CUBLAS_OP_C;
	else error("OpCblasToCublas: Invalid Op\n");
}

cublasOperation_t OpCharToCublas(char src)
{
	if(src == 'N') return CUBLAS_OP_N;
	else if(src == 'T') return CUBLAS_OP_T;
	else if(src == 'C') return CUBLAS_OP_C;
	else error("OpCharToCublas: Invalid Op: %c\n", src);
}

CBLAS_TRANSPOSE OpCharToCblas(char src)
{
	if(src == 'N') return CblasNoTrans;
	else if(src == 'T') return CblasTrans;
	else if(src == 'C') return CblasConjTrans;
	else error("OpCharToCblas: Invalid Op: %c\n", src);
}

CBLAS_TRANSPOSE OpCublasToCblas(cublasOperation_t src)
{
	if(src == CUBLAS_OP_N) return CblasNoTrans;
	else if(src == CUBLAS_OP_T) return CblasTrans;
	else if(src == CUBLAS_OP_C) return CblasConjTrans;
	else error("OpCublasToCblas: Invalid Op\n");
}

char PrintCublasOp(cublasOperation_t src)
{

	if(src == CUBLAS_OP_N) return 'N';
	else if(src == CUBLAS_OP_T) return 'T';
	else if(src == CUBLAS_OP_C) return 'C';
	else error("PrintCublasOp: Invalid Op\n");
}
