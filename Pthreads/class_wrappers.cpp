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
	backend_data_p thread_data = (backend_data_p) args;
	queue<pthread_task_p>* task_queue_p = (queue<pthread_task_p>* )thread_data->taskQueue;

	while(1){
		get_lock_q(&thread_data->queueLock);
		if(thread_data->terminate){
			release_lock_q(&thread_data->queueLock);
			break;
		} 
		if(task_queue_p->size() > 0){
			// get next task
			pthread_task_p curr_task_p = task_queue_p->front();
			task_queue_p->pop();

			cout << "taskExecLoop: Thread " << thread_data->threadId << " -- Got task = " << curr_task_p << endl;

			release_lock_q(&thread_data->queueLock);

			if(curr_task_p){
				// execute task
				void* (*curr_func) (void*);
				curr_func = (void* (*)(void*))curr_task_p->func;
				curr_func(curr_task_p->data);

				// delete task
				delete(curr_task_p);
			}
			else{
				// This should not happen
				cout << "taskExecLoop: Error: Thread " << thread_data->threadId << " -- task = " << curr_task_p << " -- executed successfully!" << endl;
				cout << "taskExecLoop: Shouldn't reach this point " << endl;
			}
		}
		release_lock_q(&thread_data->queueLock);
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
	backend_data_p data = new backend_data;
	
	cout << "CommandQueue::CommandQueue: Declared thread with id = " << data->threadId << endl;

	data->taskQueue = (void *) task_queue;
	cout << "CommandQueue::CommandQueue: task queue initialized" << endl;
	data->queueLock = 0; // initialize queue lock
	cout << "CommandQueue::CommandQueue: queue lock initialized" << endl;
	data->terminate = false;
	cout << "CommandQueue::CommandQueue: terminate initialized" << endl;
	cqueue_backend_data = (void*) data;
	cout << "CommandQueue::CommandQueue: cqueue_backend_data initialized" << endl;

	// TODO: spawn thread that loops over queue and executes tasks
	pthread_create(&(data->threadId), NULL, taskExecLoop, data);
	cout << "CommandQueue::CommandQueue: pthread_create changed id to tid = " << data->threadId << endl;

	cout << "CommandQueue::CommandQueue: Queue constructor complete" << endl;

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
	cout << "CommandQueue::~CommandQueue: Enter Queue destructor" << endl;
	sync_barrier();
	// CoCoPeLiaSelectDevice(dev_id);

	backend_data_p backend_d = (backend_data_p) cqueue_backend_data;
	get_lock_q(&backend_d->queueLock);
	backend_d->terminate = true;
	release_lock_q(&backend_d->queueLock);

	if(pthread_join(backend_d->threadId, NULL) != 0) cout << "CommandQueue::~CommandQueue: Error: pthread_join" << endl;

	queue<pthread_task_p> * task_queue_p = (queue<pthread_task_p> *)cqueue_backend_ptr;
	delete(task_queue_p);
	delete(backend_d);

	cout << "CommandQueue::~CommandQueue: Queue destructor complete" << endl;

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

	cout << "CommandQueue::sync_barrier: Enter sync_barrier" << endl;

	queue<pthread_task_p> * task_queue_p = (queue<pthread_task_p> *)cqueue_backend_ptr;

	// busy wait until task queue is empty
	while(task_queue_p->size() > 0){
		cout << "CommandQueue::sync_barrier: task_queue_p->size() = "	<< task_queue_p->size() << endl;
	}

	cout << "CommandQueue::sync_barrier: sync_barrier complete" << endl;

#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| CommandQueue::sync_barrier()\n", dev_id);
#endif
}

void CommandQueue::add_host_func(void* func, void* data){
	cout << "CommandQueue::add_host_func: Enter add_host_func" << endl;
	get_lock();
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> CommandQueue::add_host_func()\n", dev_id);
#endif

	queue<pthread_task_p> * task_queue_p = (queue<pthread_task_p> *)cqueue_backend_ptr;
	pthread_task_p task_p = new pthread_task;
	task_p->func = func;
	task_p->data = data;

	backend_data_p backend_d = (backend_data_p) cqueue_backend_data;

	get_lock_q(&backend_d->queueLock);
	task_queue_p->push(task_p);
	release_lock_q(&backend_d->queueLock);

	release_lock();

	cout << "CommandQueue::add_host_func: add_host_func complete" << endl;

#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| CommandQueue::add_host_func()\n", dev_id);
#endif
}

// void CommandQueue::wait_for_event(Event_p Wevent)
// {
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] |-----> CommandQueue::wait_for_event(Event(%d))\n", dev_id, Wevent->id);
// #endif
// 	if (Wevent->query_status() == CHECKED);
// 	else{
// 		// TODO: New addition (?)
// 		if (Wevent->query_status() == UNRECORDED) error("CommandQueue::wait_for_event:: UNRECORDED event\n");
// 		get_lock();

// 		cudaStream_t stream = *((cudaStream_t*) cqueue_backend_ptr);

// 		cudaEvent_t cuda_event= *(cudaEvent_t*) Wevent->event_backend_ptr;
// 		release_lock();
// 		cudaError_t err = cudaStreamWaitEvent(stream, cuda_event, 0); // 0-only parameter = future NVIDIA masterplan?
// 		massert(cudaSuccess == err, "CommandQueue::wait_for_event - %s\n", cudaGetErrorString(err));
// 	}
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] <-----| CommandQueue::wait_for_event(Event(%d))\n", dev_id, Wevent->id);
// #endif
// 	return;
// }


// /*****************************************************/
// /// Event class functions. TODO: Do status = .. commands need lock?
// Event::Event(int dev_id_in)
// {
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::Event()\n", dev_id_in, Event_num_device[idxize(dev_id_in)]);
// #endif
// 	get_lock();
// 	event_backend_ptr = malloc(sizeof(cudaEvent_t));
// 	id = Event_num_device[idxize(dev_id_in)];
// 	Event_num_device[idxize(dev_id_in)]++;
// #ifndef ENABLE_LAZY_EVENTS
// 	dev_id = dev_id_in;
// 	cudaError_t err = cudaEventCreate(( cudaEvent_t*) event_backend_ptr);
// 	massert(cudaSuccess == err, "Event::Event() - %s\n", cudaGetErrorString(err));
// #else
// 	dev_id = dev_id_in - 42;
// #endif
// 	status = UNRECORDED;
// 	release_lock();
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::Event()\n", dev_id, id);
// #endif
// }

// Event::~Event()
// {
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::~Event()\n", dev_id, id);
// #endif
// 	sync_barrier();
// 	get_lock();
// #ifndef ENABLE_LAZY_EVENTS
// 	Event_num_device[idxize(dev_id)]--;
// 	cudaError_t err = cudaEventDestroy(*(( cudaEvent_t*) event_backend_ptr));
// 	massert(cudaSuccess == err, "Event(%d)::~Event() - %s\n", id, cudaGetErrorString(err));
// #else
// 	if (dev_id < -1) 	Event_num_device[idxize(dev_id+42)]--;
// 	else{
// 			Event_num_device[idxize(dev_id)]--;
// 			cudaError_t err = cudaEventDestroy(*(( cudaEvent_t*) event_backend_ptr));
// 			massert(cudaSuccess == err, "Event(%d)::~Event() - %s\n", id, cudaGetErrorString(err));
// 	}
// #endif
// 	free(event_backend_ptr);
// 	release_lock();
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::~Event()\n", dev_id, id);
// #endif
// }

// void Event::sync_barrier()
// {
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::sync_barrier()\n", dev_id, id);
// #endif
// 	//get_lock();
// 	if (status != CHECKED){
// 		if (status == UNRECORDED){;
// #ifdef UDEBUG
// 			warning("[dev_id=%3d] |-----> Event(%d)::sync_barrier() - Tried to sync unrecorded event\n", dev_id, id);
// #endif
// 		}
// 		else{
// 			cudaEvent_t cuda_event= *(cudaEvent_t*) event_backend_ptr;
// 			cudaError_t err = cudaEventSynchronize(cuda_event);
// 			if (status == RECORDED) status = CHECKED;
// 			massert(cudaSuccess == err, "Event::sync_barrier() - %s\n", cudaGetErrorString(err));
// 		}
// 	}
// 	//release_lock();
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::sync_barrier()\n", dev_id, id);
// #endif
// 	return;
// }

// void Event::record_to_queue(CQueue_p Rr){
// 	get_lock();
// 	if (Rr == NULL){
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] <-----> Event(%d)::record_to_queue(NULL)\n", dev_id, id);
// #endif
// 		status = CHECKED;
// 		release_lock();
// 		return;
// 	}
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::record_to_queue(Queue(dev_id=%d))\n", dev_id, id, Rr->dev_id);
// #endif
// 	int prev_dev_id;
// 	cudaGetDevice(&prev_dev_id);
// 	if (Rr->dev_id != prev_dev_id){
// 		CoCoPeLiaSelectDevice(Rr->dev_id);
// #ifdef UDEBUG
// 		warning("Event(%d,dev_id = %d)::record_to_queue(%d): caller prev_dev_id=%d, changing to %d\n",
// 		id, dev_id, Rr->dev_id, prev_dev_id, Rr->dev_id);
// #endif
// 	}
// 	if (status != UNRECORDED){
// 		;
// #ifdef UDEBUG
// 		warning("Event(%d,dev_id = %d)::record_to_queue(%d): Recording %s event\n",
// 			id, dev_id, Rr->dev_id, print_event_status(status));
// #endif
// #ifdef ENABLE_LAZY_EVENTS
// 		if(Rr->dev_id != dev_id)
// 			error("(Lazy)Event(%d,dev_id = %d)::record_to_queue(%d): Recording %s event in iligal dev\n",
// 				id, dev_id, Rr->dev_id, print_event_status(status));
// #endif
// 	}
// #ifdef ENABLE_LAZY_EVENTS
// 	else if (status == UNRECORDED){
// 		if(dev_id > -1) /// TODO: This used to be an error, but with soft reset it was problematic...is it ok?
// 			;//warning("(Lazy)Event(%d,dev_id = %d)::record_to_queue(%d) - UNRECORDED event suspicious dev_id\n",
// 			//	id, dev_id, Rr->dev_id);
// 		dev_id = Rr->dev_id;
// 		cudaError_t err = cudaEventCreate(( cudaEvent_t*) event_backend_ptr);
// 		massert(cudaSuccess == err, "(Lazy)Event(%d,dev_id = %d)::record_to_queue(%d): - %s\n",
// 			id, dev_id, Rr->dev_id, cudaGetErrorString(err));
// 	}
// #endif
// 	cudaEvent_t cuda_event= *(cudaEvent_t*) event_backend_ptr;

// 	cudaStream_t stream = *((cudaStream_t*) Rr->cqueue_backend_ptr);
// 	cudaError_t err = cudaEventRecord(cuda_event, stream);

// 	status = RECORDED;
// 	massert(cudaSuccess == err, "Event(%d,dev_id = %d)::record_to_queue(%d) - %s\n",  id, dev_id, Rr->dev_id, cudaGetErrorString(err));
// 	if (Rr->dev_id != prev_dev_id){
// 		cudaSetDevice(prev_dev_id);
// 	}
// 	release_lock();
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::record_to_queue(Queue(dev_id=%d))\n", dev_id, id, Rr->dev_id);
// #endif
// }

// event_status Event::query_status(){
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::query_status()\n", dev_id, id);
// #endif
// 	get_lock();
// 	enum event_status local_status = status;
// 	if (local_status != CHECKED){
// #ifdef ENABLE_LAZY_EVENTS
// 		if (local_status == UNRECORDED){
// 			release_lock();
// 			return UNRECORDED;
// 		}
// #endif
// 		cudaEvent_t cuda_event= *(cudaEvent_t*) event_backend_ptr;
// 		cudaError_t err = cudaEventQuery(cuda_event);

// 		if (err == cudaSuccess && (local_status == UNRECORDED ||  local_status == COMPLETE));
// 		else if (err == cudaSuccess && local_status == RECORDED) local_status = status = COMPLETE;
// 		else if (err == cudaErrorNotReady && local_status == RECORDED);
// 		else if (err == cudaErrorNotReady && local_status == UNRECORDED){
// #ifdef UDEBUG
// 			// this should not happen in a healthy locked update scenario.
// 			warning("Event::query_status(): cudaErrorNotReady with status == UNRECORDED should not happen\n");
// #endif
// 			local_status = status = RECORDED;
// 		}
// 		else if (err == cudaSuccess &&  local_status == CHECKED){
// 			;
// 			// TODO: This should not happen in a healthy locked update scenario.
// 			// But it does since no locking yet. Not sure of its effects.
// #ifdef UDEBUG
// 			warning("[dev_id=%3d] |-----> Event(%d)::query_status(): cudaSuccess with local_status == CHECKED should not happen\n", dev_id, id);
// #endif
// 		}
// 		else error("[dev_id=%3d] |-----> Event(%d)::query_status() - %s, local_status=%s, status = %s\n", dev_id, id,
// 		cudaGetErrorString(err), print_event_status(local_status), print_event_status(status));
// 	}
// 	release_lock();
// #ifdef UDDEBUG
// 	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::query_status() = %s\n", dev_id, id, print_event_status(status));
// #endif
// 	return local_status;
// }

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
