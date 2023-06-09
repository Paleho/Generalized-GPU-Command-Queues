#include "unihelpers.hpp"
#include "backend_wrappers.hpp"

// Some dummy task function
void* taskFun(void* input){
    int * x = (int*) input;
    cout << "taskFun: input = " << *x << endl;

    *x += 1000;

    return 0;
}

int main(){
    CQueue_p MyQueue_p = new CommandQueue(-1);

    int input = 1000;
    MyQueue_p->add_host_func((void*) &taskFun, (void*) &input);
    MyQueue_p->add_host_func((void*) &taskFun, (void*) &input);
    MyQueue_p->add_host_func((void*) &taskFun, (void*) &input);
    MyQueue_p->add_host_func((void*) &taskFun, (void*) &input);

    MyQueue_p->sync_barrier();
    cout << "After sync barrier" << endl;
    MyQueue_p->add_host_func((void*) &taskFun, (void*) &input);
    MyQueue_p->add_host_func((void*) &taskFun, (void*) &input);
    MyQueue_p->sync_barrier();

    delete(MyQueue_p);
    cout << "Main: final result = " << input << endl;
    return 0;
}