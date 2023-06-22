#include "unihelpers.hpp"
#include "backend_wrappers.hpp"

// Some dummy task function
void* taskFun(void* input){
    int * x = (int*) input;
    std::cout << "taskFun: input = " << *x << std::endl;

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
    std::cout << "After sync barrier" << std::endl;
    MyQueue_p->add_host_func((void*) &taskFun, (void*) &input);
    MyQueue_p->add_host_func((void*) &taskFun, (void*) &input);
    MyQueue_p->sync_barrier();

    delete(MyQueue_p);
    std::cout << "Main: final result = " << input << std::endl;
    return 0;
}