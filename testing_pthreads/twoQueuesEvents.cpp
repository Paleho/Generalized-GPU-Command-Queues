#include "unihelpers.hpp"
#include "backend_wrappers.hpp"

void* taskFun_Add(void* input){
    int * x = (int*) input;

    *x += 10;

    return 0;
}

void* taskFun_Mult(void* input){
    int * x = (int*) input;

    *x *= 10;

    return 0;
}

int main(){
    CQueue_p Q_adds_p = new CommandQueue(-1);
    CQueue_p Q_mults_p = new CommandQueue(-1);
    Event_p add1_complete = new Event(-1);
    Event_p mul1_complete = new Event(-1);
    Event_p add2_complete = new Event(-1);
    Event_p mul2_complete = new Event(-1);
    Event_p add3_complete = new Event(-1);

    int input = 5;
    Q_adds_p->add_host_func((void*) &taskFun_Add, (void*) &input);
    add1_complete->record_to_queue(Q_adds_p);

    Q_mults_p->wait_for_event(add1_complete);
    Q_mults_p->add_host_func((void*) &taskFun_Mult, (void*) &input);
    mul1_complete->record_to_queue(Q_mults_p);

    Q_adds_p->wait_for_event(mul1_complete);
    Q_adds_p->add_host_func((void*) &taskFun_Add, (void*) &input);
    Q_adds_p->add_host_func((void*) &taskFun_Add, (void*) &input);
    add2_complete->record_to_queue(Q_adds_p);

    Q_mults_p->wait_for_event(add2_complete);
    Q_mults_p->add_host_func((void*) &taskFun_Mult, (void*) &input);
    mul2_complete->record_to_queue(Q_mults_p);

    Q_adds_p->wait_for_event(mul2_complete);
    Q_adds_p->add_host_func((void*) &taskFun_Add, (void*) &input);
    add3_complete->record_to_queue(Q_adds_p);

    Q_mults_p->wait_for_event(add3_complete);
    Q_mults_p->add_host_func((void*) &taskFun_Mult, (void*) &input);

    std::cout << "MAIN: all tasks added" << std::endl;

    Q_mults_p->sync_barrier();
    std::cout << "MAIN: synched" << std::endl;

    delete(Q_adds_p);
    delete(Q_mults_p);
    delete(add1_complete);
    delete(mul1_complete);
    delete(add2_complete);
    delete(mul2_complete);
    delete(add3_complete);
    std::cout << "Main: final result = " << input << std::endl;
    return 0;
}