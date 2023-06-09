#include "unihelpers.hpp"
#include "backend_wrappers.hpp"

// Some dummy task function
void* taskFun(void* input){
    int * x = (int*) input;
    cout << "taskFun: input = " << *x << endl;

    *x += 1000;

    return 0;
}

void* taskFunDouble(void* input){
    int * x = (int*) input;
    cout << "taskFun: input = " << *x << endl;

    *x *= 2;

    return 0;
}

int main(){
    CQueue_p Q1_p = new CommandQueue(-1);
    CQueue_p Q2_p = new CommandQueue(-1);

    int input = 1000;
    Q1_p->add_host_func((void*) &taskFun, (void*) &input);
    Q1_p->add_host_func((void*) &taskFun, (void*) &input);
    Q1_p->add_host_func((void*) &taskFun, (void*) &input);
    Q1_p->add_host_func((void*) &taskFun, (void*) &input);

    // Q1_p->sync_barrier();

    Q2_p->add_host_func((void*) &taskFunDouble, (void*) &input);
    Q2_p->add_host_func((void*) &taskFunDouble, (void*) &input);
    Q2_p->add_host_func((void*) &taskFunDouble, (void*) &input);
    Q2_p->add_host_func((void*) &taskFunDouble, (void*) &input);

    Q1_p->sync_barrier();
    Q2_p->sync_barrier();
    cout << "Main: After sync barrier" << endl;


    Q1_p->add_host_func((void*) &taskFun, (void*) &input);
    Q1_p->add_host_func((void*) &taskFun, (void*) &input);
    Q1_p->sync_barrier();

    delete(Q1_p);
    delete(Q2_p);
    cout << "Main: final result = " << input << endl;
    return 0;
}