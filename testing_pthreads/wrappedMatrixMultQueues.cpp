///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief 
#include "unihelpers.hpp"
#include "pthreads_backend_wrappers.hpp"

const double epsilon = 0.00001;

#define IDX2F(i,j,ld) (((j)*(ld)) + (i))

// Column - major matrix initialization
void matrixInit(double * M, int rows, int cols)
{
	for(int i = 0; i < rows; i++)
		for(int j = 0; j < cols; j++)
			M[IDX2F(i,j, rows)] = (double) (rand() % 10) - 5;
}

void _matrixMultiply(double* A, double* B, int m, int k, int n, double alpha, double* Res)
{
	int rows = m;
	int cols = n;
	for(int i = 0; i < rows; i++)
		for(int j = 0; j < cols; j++){
			double sum = 0;
			for(int x = 0; x < k; x++)
				sum += A[IDX2F(i,x,m)]*B[IDX2F(x,j,k)];

			Res[IDX2F(i,j,m)] = alpha * sum;
		}
}

int verifyRes(double* serial_Res, double* parallel_Res, int m, int n)
{
	for(int i = 0; i < m; i++)
		for(int j = 0; j < n; j++){
			double correctResult = serial_Res[IDX2F(i,j, m)];
			double dif = parallel_Res[IDX2F(i,j, m)] - correctResult;
			bool expr1 = dif > epsilon;
			bool expr2 = dif < (-1)*epsilon;
			if(expr1 || expr2){
				printf("verifyRes: incorrect at (%d, %d) -- should be = %0.5lf BUT Res[i, j] = %0.5lf -- Error = %0.5lf\n", i, j, correctResult, parallel_Res[IDX2F(i,j, m)], dif);
				return 0;
			} 
		}
	return 1;
}

int main(int argc, char ** argv){
	if(argc < 2){
		std::cout << "Invalid number of command line arguments\n";
		std::cout << "Proper usage: " << argv[0] << " [N] [synched/desynched]\n";
		return 2;
	}

	bool synched;
	if(!strcmp(argv[2], "synched")) synched = true;
	else if(!strcmp(argv[2], "desynched")) synched = false;
	else{
		std::cout << "Invalid argv[2]\n";
		std::cout << "Proper usage: " << argv[0] << " [N] [synched/desynched]\n";
		return 2;
	}

	int returnFlag = 0;
	const int N = atoi(argv[1]);
 	size_t size_n_by_n = N * N * sizeof(double);

	if(N <= 0){
		std::cout << "Invalid command line arg (N)\n";
		return 2;
	}

	// Allocate input matrices 
	double *h_A = new double[size_n_by_n];
	double *h_B = new double[size_n_by_n];
	double *h_AB = new double[size_n_by_n];
	double *h_C = new double[size_n_by_n];
	double *h_ABC = new double[size_n_by_n];
	double *h_D = new double[size_n_by_n];
	double *h_ABCD = new double[size_n_by_n];
	double *h_M = new double[size_n_by_n];
	double *h_MABCD = new double[size_n_by_n];

	double *verify_MA = new double[size_n_by_n];
	double *verify_MAB = new double[size_n_by_n];
	double *verify_MABC = new double[size_n_by_n];
	double *verify_Res = new double[size_n_by_n];

	// Initialize input matrices
	matrixInit(h_A, N, N);
	matrixInit(h_B, N, N);
	matrixInit(h_C, N, N);
	matrixInit(h_D, N, N);
	matrixInit(h_M, N, N);

	CQueue_p Q1_p = new CommandQueue(-1);
	CQueue_p Q2_p = new CommandQueue(-1);
	Event_p mult1_complete = new Event(-1);

	gemm_backend_in<double>* input_1 = new gemm_backend_in<double>;
	gemm_backend_in<double>* input_2 = new gemm_backend_in<double>;
	gemm_backend_in<double>* input_3 = new gemm_backend_in<double>;
	gemm_backend_in<double>* input_4 = new gemm_backend_in<double>;

	input_1->TransA = 'N';	// normal matrix A
	input_1->TransB = 'N';	// normal matrix B
	input_1->M = N;
	input_1->N = N;
	input_1->K = N;
	input_1->ldA = N;
	input_1->ldB = N;
	input_1->ldC = N;
	input_1->alpha = 1.0;
	input_1->beta = (double) 0.0;
	input_1->A = (void **) &h_A;
	input_1->B = (void **) &h_B;
	input_1->C = (void **) &h_AB;
	input_1->dev_id = -1;
	backend_run_operation(input_1, "Dgemm", Q1_p);

	input_2->TransA = 'N';	// normal matrix A
	input_2->TransB = 'N';	// normal matrix B
	input_2->M = N;
	input_2->N = N;
	input_2->K = N;
	input_2->ldA = N;
	input_2->ldB = N;
	input_2->ldC = N;
	input_2->alpha = 1.0;
	input_2->beta = (double) 0.0;
	input_2->A = (void **) &h_AB;
	input_2->B = (void **) &h_C;
	input_2->C = (void **) &h_ABC;
	input_2->dev_id = -1;
	backend_run_operation(input_2, "Dgemm", Q1_p);

	input_3->TransA = 'N';	// normal matrix A
	input_3->TransB = 'N';	// normal matrix B
	input_3->M = N;
	input_3->N = N;
	input_3->K = N;
	input_3->ldA = N;
	input_3->ldB = N;
	input_3->ldC = N;
	input_3->alpha = 1.0;
	input_3->beta = (double) 0.0;
	input_3->A = (void **) &h_ABC;
	input_3->B = (void **) &h_D;
	input_3->C = (void **) &h_ABCD;
	input_3->dev_id = -1;
	backend_run_operation(input_3, "Dgemm", Q1_p);

	mult1_complete->record_to_queue(Q1_p);

	if(synched)
		Q2_p->wait_for_event(mult1_complete);

	input_4->TransA = 'N';	// normal matrix A
	input_4->TransB = 'N';	// normal matrix B
	input_4->M = N;
	input_4->N = N;
	input_4->K = N;
	input_4->ldA = N;
	input_4->ldB = N;
	input_4->ldC = N;
	input_4->alpha = 1.0;
	input_4->beta = (double) 0.0;
	input_4->A = (void **) &h_M;
	input_4->B = (void **) &h_ABCD;
	input_4->C = (void **) &h_MABCD;
	input_4->dev_id = -1;
	backend_run_operation(input_4, "Dgemm", Q2_p);

	_matrixMultiply(h_M, h_A, N, N, N, 1.0, verify_MA);
	_matrixMultiply(verify_MA, h_B, N, N, N, 1.0, verify_MAB);
	_matrixMultiply(verify_MAB, h_C, N, N, N, 1.0, verify_MABC);
	_matrixMultiply(verify_MABC, h_D, N, N, N, 1.0, verify_Res);

	Q2_p->sync_barrier();

	// Verify result
	if(!verifyRes(verify_Res, h_MABCD, N, N)){
		std::cout << "Fail: The result of " << argv[2] << " computations is incorrect!\n";
		returnFlag = 1;

		std::cout << "Some more elements: \n";

		printf("verify_Res[0, 1] = %0.5lf BUT h_MABCD[0, 1] = %0.5lf\n", verify_Res[IDX2F(0, 1, N)], h_MABCD[IDX2F(0, 1, N)]);
		printf("verify_Res[1, 0] = %0.5lf BUT h_MABCD[1, 0] = %0.5lf\n", verify_Res[IDX2F(1, 0, N)], h_MABCD[IDX2F(1, 0, N)]);
		printf("verify_Res[2, 1] = %0.5lf BUT h_MABCD[2, 1] = %0.5lf\n", verify_Res[IDX2F(2, 1, N)], h_MABCD[IDX2F(2, 1, N)]);
	}
	else{
		std::cout << "Success: The result of " << argv[2] << " computations is correct!\n";
		returnFlag = 0;
	}

	// Free device 0 mem
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
	delete[] h_D;
	delete[] h_M;
	delete[] h_AB;
	delete[] h_ABC;
	delete[] h_ABCD;
	delete[] h_MABCD;

	delete[] verify_MA;
	delete[] verify_MAB;
	delete[] verify_MABC;
	delete[] verify_Res;

	delete Q1_p;
	delete Q2_p;
	delete mult1_complete;
	delete input_1;
	delete input_2;
	delete input_3;
	delete input_4;

	return returnFlag;
}