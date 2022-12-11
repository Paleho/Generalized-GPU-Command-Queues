#include "unihelpers.hpp"
#include "pthreads_backend_wrappers.hpp"

// Some dummy task function
void* taskFun(void* input){
    int x = * (int*) input;
    cout << "taskFun: input = " << x << endl;

    int sum = 0;
    for(int i = 0; i < x; i++){
        sum += i;
    }

    return 0;
}

int main(){
    CQueue_p MyQueue_p = new CommandQueue(-1);

    int input_1 = 1001;
    MyQueue_p->add_host_func((void*) &taskFun, (void*) &input_1);
    int input_2 = 1002;
    MyQueue_p->add_host_func((void*) &taskFun, (void*) &input_2);
    int input_3 = 1003;
    MyQueue_p->add_host_func((void*) &taskFun, (void*) &input_3);
    int input_4 = 1004;
    MyQueue_p->add_host_func((void*) &taskFun, (void*) &input_4);

    MyQueue_p->sync_barrier();
    cout << "After sync barrier" << endl;
    int input_5 = 1005;
    MyQueue_p->add_host_func((void*) &taskFun, (void*) &input_5);
    int input_6 = 1006;
    MyQueue_p->add_host_func((void*) &taskFun, (void*) &input_6);

    delete(MyQueue_p);
    return 0;
}