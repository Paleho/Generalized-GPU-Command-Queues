///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief 
///

#include <queue>
#include <unihelpers.hpp>
#include <pthreads_backend_wrappers.hpp>

int lvl = 1;

int Event_num_device[128] = {0};
#ifndef UNIHELPER_LOCKFREE_ENABLE
int unihelper_lock = 0;
#endif

inline void get_lock(){
#ifndef UNIHELPER_LOCKFREE_ENABLE
	while(__sync_lock_test_and_set (&unihelper_lock, 1));
#endif
	;
}
inline void release_lock(){
#ifndef UNIHELPER_LOCKFREE_ENABLE
	__sync_lock_release(&unihelper_lock);
#endif
	;
}

// Queue lock
inline void get_lock_q(int * lock){
	while(__sync_lock_test_and_set (lock, 1));
}
inline void release_lock_q(int * lock){
	__sync_lock_release(lock);
}

/*****************************************************/
/// Event Status-related functions

const char* print_event_status(event_status in_status){
	switch(in_status){
		case(UNRECORDED):
			return "UNRECORDED";
		case(RECORDED):
			return "RECORDED";
		case(COMPLETE):
			return "COMPLETE";
		case(CHECKED):
			return "CHECKED";
		case(GHOST):
			return "GHOST";
		default:
			error("print_event_status: Unknown state\n");
	}
}

void* taskExecLoop(void * args)
{
	// extract queue and lock from data
	queue_data_p thread_data = (queue_data_p) args;
	queue<pthread_task_p>* task_queue_p = (queue<pthread_task_p>* )thread_data->taskQueue;

	while(1){
		// cout << "taskExecLoop: Thread " << thread_data->threadId << " waiting for queueLock" << endl;
		get_lock_q(&thread_data->queueLock);
		// cout << "taskExecLoop: Thread " << thread_data->threadId << " got queueLock" << endl;
		if(thread_data->terminate){
			release_lock_q(&thread_data->queueLock);
			break;
		} 
		if(task_queue_p->size() > 0){
			// get next task
			pthread_task_p curr_task_p = task_queue_p->front();
			task_queue_p->pop();

			// cout << "taskExecLoop: Thread " << thread_data->threadId << " -- Got task = " << curr_task_p << endl;

			// cout << "taskExecLoop: about to release lock with value = " << thread_data->queueLock << endl;

			release_lock_q(&thread_data->queueLock);

			if(curr_task_p){
				// execute task
				void* (*curr_func) (void*);
				curr_func = (void* (*)(void*))curr_task_p->func;
				curr_func(curr_task_p->data);

				// cout << "taskExecLoop: Thread " << thread_data->threadId << " -- Executed task = " << curr_task_p << endl;

				// delete task
				delete(curr_task_p);
				// cout << "taskExecLoop: Thread " << thread_data->threadId << " -- Deleted task = " << endl;
			}
			else{
				// This should not happen
				cout << "taskExecLoop: Error: Thread " << thread_data->threadId << " -- task = " << curr_task_p << endl;
				cout << "taskExecLoop: Shouldn't reach this point " << endl;
			}
		}
		else{
			release_lock_q(&thread_data->queueLock);
		}
	}

	return 0;
}

/*****************************************************/
/// Command queue class functions
CommandQueue::CommandQueue(int dev_id_in)
{
	// TODO: bring back CoCoPeLiaGetDevice CoCoPeLiaSelectDevice
	// int prev_dev_id = CoCoPeLiaGetDevice();
	dev_id = dev_id_in;
	// CoCoPeLiaSelectDevice(dev_id);
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] |-----> CommandQueue::CommandQueue()\n", dev_id_in);
// #endif
// 	if(prev_dev_id != dev_id){;
// #ifdef UDEBUG
// 		lprintf(lvl, "[dev_id=%3d] ------- CommandQueue::CommandQueue(): Called for other dev_id = %d\n",
// 			dev_id, prev_dev_id);
// #endif
// 	}
	
#ifdef UDEBUG
		lprintf(lvl, "[dev_id=%3d] ------- CommandQueue::CommandQueue(%d): Initializing simple queue\n", dev_id);
#endif
	queue<pthread_task_p>* task_queue = new queue<pthread_task_p>;
	cqueue_backend_ptr = (void *) task_queue;
	queue_data_p data = new queue_data;
	
	data->taskQueue = (void *) task_queue;
	data->queueLock = 0; // initialize queue lock
	data->terminate = false;
	cqueue_backend_data = (void*) data;

	// Spawn thread that loops over queue and executes tasks
	if(pthread_create(&(data->threadId), NULL, taskExecLoop, data)) cout << "Error: CommandQueue::CommandQueue: pthread_create failed" << endl;

	// cout << "CommandQueue::CommandQueue: Queue constructor complete. Thread id = " << data->threadId << endl;

	// CoCoPeLiaSelectDevice(prev_dev_id);
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| CommandQueue::CommandQueue()\n", dev_id);
#endif
}

CommandQueue::~CommandQueue()
{
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> CommandQueue::~CommandQueue()\n", dev_id);
#endif
	// cout << "CommandQueue::~CommandQueue: Enter Queue destructor" << endl;
	sync_barrier();
	// CoCoPeLiaSelectDevice(dev_id);

	queue_data_p backend_d = (queue_data_p) cqueue_backend_data;
	// cout << "CommandQueue::~CommandQueue  waiting for queueLock" << endl;
	get_lock_q(&backend_d->queueLock);
	backend_d->terminate = true;
	release_lock_q(&backend_d->queueLock);

	if(pthread_join(backend_d->threadId, NULL)) cout << "Error: CommandQueue::~CommandQueue: pthread_join failed" << endl;

	queue<pthread_task_p> * task_queue_p = (queue<pthread_task_p> *)cqueue_backend_ptr;
	delete(task_queue_p);
	delete(backend_d);

	// cout << "CommandQueue::~CommandQueue: Queue destructor complete" << endl;

#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| CommandQueue::~CommandQueue()\n", dev_id);
#endif
	return;
}

void CommandQueue::sync_barrier()
{
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> CommandQueue::sync_barrier()\n", dev_id);
#endif

	// cout << "CommandQueue::sync_barrier: Enter sync_barrier" << endl;

	queue<pthread_task_p> * task_queue_p = (queue<pthread_task_p> *)cqueue_backend_ptr;
	queue_data_p backend_d = (queue_data_p) cqueue_backend_data;

	bool queueIsBusy = true;
	// busy wait until task queue is empty
	while(queueIsBusy){
		get_lock_q(&backend_d->queueLock);
		queueIsBusy = task_queue_p->size() > 0;
		release_lock_q(&backend_d->queueLock);
	}

	// cout << "CommandQueue::sync_barrier: sync_barrier complete" << endl;

#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| CommandQueue::sync_barrier()\n", dev_id);
#endif
}

void CommandQueue::add_host_func(void* func, void* data){
	// cout << "CommandQueue::add_host_func: Enter add_host_func -- waiting for unihelpersLock" << endl;
	get_lock();
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> CommandQueue::add_host_func()\n", dev_id);
#endif

	queue<pthread_task_p> * task_queue_p = (queue<pthread_task_p> *)cqueue_backend_ptr;
	pthread_task_p task_p = new pthread_task;
	task_p->func = func;
	task_p->data = data;

	queue_data_p backend_d = (queue_data_p) cqueue_backend_data;

	// cout << "CommandQueue::add_host_func: waiting for queueLock" << endl;
	get_lock_q(&backend_d->queueLock);
	task_queue_p->push(task_p);
	release_lock_q(&backend_d->queueLock);

	release_lock();

	// cout << "CommandQueue::add_host_func: add_host_func complete" << endl;

#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| CommandQueue::add_host_func()\n", dev_id);
#endif
}

void * blockQueue(void * data){
	// cout << "blockQueue: Start" << endl;
	Event_p Wevent = (Event_p) data;

	while(Wevent->query_status() < COMPLETE){
		;
		// cout << "blockQueue: waiting for event = "	<< (pthread_event_p) Wevent->event_backend_ptr << endl;
	}

	return 0;
}

void CommandQueue::wait_for_event(Event_p Wevent)
{
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> CommandQueue::wait_for_event(Event(%d))\n", dev_id, Wevent->id);
#endif
	if (Wevent->query_status() == CHECKED);
	else{
		// TODO: New addition (?)
		if (Wevent->query_status() == UNRECORDED) error("CommandQueue::wait_for_event:: UNRECORDED event\n");

		// cout << "CommandQueue::wait_for_event: waiting for unihelpersLock" << endl;
		get_lock();

		queue_data_p backend_d = (queue_data_p) cqueue_backend_data;
		pthread_event_p event_p = (pthread_event_p) Wevent->event_backend_ptr;

		release_lock();
		add_host_func((void*) &blockQueue, (void*) Wevent);
	}
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| CommandQueue::wait_for_event(Event(%d))\n", dev_id, Wevent->id);
#endif
	return;
}

void* eventFunc(void* event_data){
	pthread_event_p event_p = (pthread_event_p) event_data;
	event_p->estate = COMPLETE;

	return 0;
}


/*****************************************************/
/// Event class functions. TODO: Do status = .. commands need lock?
Event::Event(int dev_id_in)
{
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::Event()\n", dev_id_in, Event_num_device[idxize(dev_id_in)]);
#endif
	// cout << "Event::Event: waiting for unihelpersLock" << endl;
	get_lock();
	pthread_event_p event_p = new pthread_event;
	event_p->estate = UNRECORDED;
	event_backend_ptr = (void*) event_p;
	status = UNRECORDED;
	release_lock();
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::Event()\n", dev_id, id);
#endif
}

Event::~Event()
{
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::~Event()\n", dev_id, id);
#endif
	sync_barrier();
	// cout << "Event::~Event: waiting for unihelpersLock" << endl;
	get_lock();

	pthread_event_p event_p = (pthread_event_p) event_backend_ptr;
	delete(event_p);
	release_lock();
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::~Event()\n", dev_id, id);
#endif
}

void Event::sync_barrier()
{
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::sync_barrier()\n", dev_id, id);
#endif
	//get_lock();
	if (status != CHECKED){
		if (status == UNRECORDED){;
#ifdef UDEBUG
			warning("[dev_id=%3d] |-----> Event(%d)::sync_barrier() - Tried to sync unrecorded event\n", dev_id, id);
#endif
		}
		else{
			pthread_event_p event_p = (pthread_event_p) event_backend_ptr;
			while(event_p->estate < COMPLETE);

			status = COMPLETE;
		}
	}
	//release_lock();
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::sync_barrier()\n", dev_id, id);
#endif
	return;
}

void Event::record_to_queue(CQueue_p Rr){
	// cout << "Event::record_to_queue: waiting for unihelpersLock" << endl;
	get_lock();
	if (Rr == NULL){
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----> Event(%d)::record_to_queue(NULL)\n", dev_id, id);
#endif
		status = CHECKED;
		release_lock();
		return;
	}
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::record_to_queue(Queue(dev_id=%d))\n", dev_id, id, Rr->dev_id);
#endif
	int prev_dev_id;
	// cudaGetDevice(&prev_dev_id);
// 	if (Rr->dev_id != prev_dev_id){
// 		CoCoPeLiaSelectDevice(Rr->dev_id);
// #ifdef UDEBUG
// 		warning("Event(%d,dev_id = %d)::record_to_queue(%d): caller prev_dev_id=%d, changing to %d\n",
// 		id, dev_id, Rr->dev_id, prev_dev_id, Rr->dev_id);
// #endif
// 	}
	if (status != UNRECORDED){
		;
#ifdef UDEBUG
		warning("Event(%d,dev_id = %d)::record_to_queue(%d): Recording %s event\n",
			id, dev_id, Rr->dev_id, print_event_status(status));
#endif
#ifdef ENABLE_LAZY_EVENTS
		if(Rr->dev_id != dev_id)
			error("(Lazy)Event(%d,dev_id = %d)::record_to_queue(%d): Recording %s event in iligal dev\n",
				id, dev_id, Rr->dev_id, print_event_status(status));
#endif
	}
#ifdef ENABLE_LAZY_EVENTS
	else if (status == UNRECORDED){
		if(dev_id > -1) /// TODO: This used to be an error, but with soft reset it was problematic...is it ok?
			;//warning("(Lazy)Event(%d,dev_id = %d)::record_to_queue(%d) - UNRECORDED event suspicious dev_id\n",
			//	id, dev_id, Rr->dev_id);
		// dev_id = Rr->dev_id;
		// cudaError_t err = cudaEventCreate(( cudaEvent_t*) event_backend_ptr);
		// massert(cudaSuccess == err, "(Lazy)Event(%d,dev_id = %d)::record_to_queue(%d): - %s\n",
		// 	id, dev_id, Rr->dev_id, cudaGetErrorString(err));
	}
#endif
	pthread_event_p event_p = (pthread_event_p) event_backend_ptr;
	if(event_p->estate != UNRECORDED) 
		warning("Event(%d,dev_id = %d)::record_to_queue(%d): Recording %s event\n",
			id, dev_id, Rr->dev_id, print_event_status(status));

	event_p->estate = RECORDED;
	status = RECORDED;
	release_lock();

	// cout << "Event::record_to_queue: event " << event_p << "recorded" << endl;
	Rr->add_host_func((void*) &eventFunc, (void*) event_p);

	// if (Rr->dev_id != prev_dev_id){
	// 	cudaSetDevice(prev_dev_id);
	// }
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::record_to_queue(Queue(dev_id=%d))\n", dev_id, id, Rr->dev_id);
#endif
}

event_status Event::query_status(){
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::query_status()\n", dev_id, id);
#endif
	// cout << "Event::query_status: waiting for unihelpersLock" << endl;
	get_lock();
	enum event_status local_status = status;
	if (local_status != CHECKED){
#ifdef ENABLE_LAZY_EVENTS
		if (local_status == UNRECORDED){
			release_lock();
			return UNRECORDED;
		}
#endif
		pthread_event_p event_p = (pthread_event_p) event_backend_ptr;
		
		if(status == RECORDED && event_p->estate == COMPLETE) status = COMPLETE;
		local_status = event_p->estate;
	}
	release_lock();
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::query_status() = %s\n", dev_id, id, print_event_status(status));
#endif
	return local_status;
}

// void Event::checked(){
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::checked()\n", dev_id, id);
// #endif
// 	get_lock();
// 	if (status == COMPLETE) status = CHECKED;
// 	else error("Event::checked(): error event was %s,  not COMPLETE()\n", print_event_status(status));
// 	release_lock();
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::checked()\n", dev_id, id);
// #endif
// }

// void Event::soft_reset(){
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::soft_reset()\n", dev_id, id);
// #endif
// 	//sync_barrier();
// 	get_lock();
// 	//event_status prev_status = status;
// 	status = UNRECORDED;
// 	release_lock();
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::soft_reset()\n", dev_id, id);
// #endif
// }

// void Event::reset(){
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::reset()\n", dev_id, id);
// #endif
// 	sync_barrier();
// 	get_lock();
// 	event_status prev_status = status;
// 	status = UNRECORDED;
// #ifdef ENABLE_LAZY_EVENTS
// 	if(dev_id >= -1){
// 		dev_id = dev_id - 42;
// 		cudaError_t err = cudaEventDestroy(*(( cudaEvent_t*) event_backend_ptr));
// 		massert(cudaSuccess == err, "[dev_id=%3d] (Lazy)Event(%d)::reset - %s\n", dev_id + 42, id, cudaGetErrorString(err));
// 	}
// #endif
// 	release_lock();
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::reset()\n", dev_id, id);
// #endif
// }

// /*****************************************************/
// /// Event-based timer class functions

// Event_timer::Event_timer(int dev_id) {
//   Event_start = new Event(dev_id);
//   Event_stop = new Event(dev_id);
//   time_ms = 0;
// }

// void Event_timer::start_point(CQueue_p start_queue)
// {
// 	Event_start->record_to_queue(start_queue);
// }

// void Event_timer::stop_point(CQueue_p stop_queue)
// {
// 	Event_stop->record_to_queue(stop_queue);
// }

// double Event_timer::sync_get_time()
// {
// 	float temp_t;
// 	if(Event_start->query_status() != UNRECORDED){
// 		Event_start->sync_barrier();
// 		if(Event_stop->query_status() != UNRECORDED) Event_stop->sync_barrier();
// 		else error("Event_timer::sync_get_time: Event_start is %s but Event_stop still UNRECORDED\n",
// 			print_event_status(Event_start->query_status()));
// 		cudaEvent_t cuda_event_start = *(cudaEvent_t*) Event_start->event_backend_ptr;
// 		cudaEvent_t cuda_event_stop = *(cudaEvent_t*) Event_stop->event_backend_ptr;
// 		cudaEventElapsedTime(&temp_t, cuda_event_start, cuda_event_stop);
// 	}
// 	else temp_t = 0;
// 	time_ms = (double) temp_t;
// 	return time_ms;
// }

// /*****************************************************/