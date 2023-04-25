///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief Some HIP function calls with added error-checking
///			(HIPified version of original CUDA code)
///

#include <cstdio>
#include <typeinfo>
#include <float.h>

#include "backend_wrappers.hpp"

inline static hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void* userData) {
    return hipCUDAErrorTohipError(cudaLaunchHostFunc(stream, fn, userData));
}

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

/*****************************************************/
/// Command queue class functions
CommandQueue::CommandQueue(int dev_id_in)
{
	int prev_dev_id = CoCoPeLiaGetDevice();
	dev_id = dev_id_in;
	CoCoPeLiaSelectDevice(dev_id);
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> CommandQueue::CommandQueue()\n", dev_id_in);
#endif
	if(prev_dev_id != dev_id){;
#ifdef UDEBUG
		lprintf(lvl, "[dev_id=%3d] ------- CommandQueue::CommandQueue(): Called for other dev_id = %d\n",
			dev_id, prev_dev_id);
#endif
	}
#ifdef ENABLE_PARALLEL_BACKEND
#ifdef UDEBUG
		lprintf(lvl, "[dev_id=%3d] ------- CommandQueue::CommandQueue(): Initializing parallel queue with %d Backend workers\n",
		dev_id, MAX_BACKEND_L);
#endif
	backend_ctr = 0;
	for (int par_idx = 0; par_idx < MAX_BACKEND_L; par_idx++ ){
		cqueue_backend_ptr[par_idx] = malloc(sizeof(hipStream_t));
		hipError_t err = hipStreamCreate((hipStream_t*) cqueue_backend_ptr[par_idx]);
		massert(hipSuccess == err, "CommandQueue::CommandQueue(%d) - %s\n", dev_id, hipGetErrorString(err));
		hipStream_t stream = *((hipStream_t*) cqueue_backend_ptr[par_idx]);

		cqueue_backend_data[par_idx] = malloc(sizeof(hipblasHandle_t));
		massert(HIPBLAS_STATUS_SUCCESS == hipblasCreate((hipblasHandle_t*) cqueue_backend_data[par_idx]),
			"CommandQueue::CommandQueue(%d): hipblasCreate failed\n", dev_id);
		massert(HIPBLAS_STATUS_SUCCESS == hipblasSetStream(*((hipblasHandle_t*) cqueue_backend_data[par_idx]), stream),
			"CommandQueue::CommandQueue(%d): hipblasSetStream failed\n", dev_id);
		//warning("FIXME: Running on limited SMs, custom stuff, beware this in not a drill\n");
		//massert(HIPBLAS_STATUS_SUCCESS == cublasSetSmCountTarget(*((hipblasHandle_t*) cqueue_backend_data[par_idx]), 1),
		//	"CommandQueue::CommandQueue(%d): cublasSetSmCountTarget failed\n", dev_id);
	}
#else
#ifdef UDEBUG
		lprintf(lvl, "[dev_id=%3d] ------- CommandQueue::CommandQueue(%d): Initializing simple queue\n", dev_id);
#endif
	cqueue_backend_ptr = malloc(sizeof(hipStream_t));
	hipError_t err = hipStreamCreate((hipStream_t*) cqueue_backend_ptr);
	massert(hipSuccess == err, "CommandQueue::CommandQueue(%d) - %s\n", dev_id, hipGetErrorString(err));
	hipStream_t stream = *((hipStream_t*) cqueue_backend_ptr);

	cqueue_backend_data = malloc(sizeof(hipblasHandle_t));
	massert(HIPBLAS_STATUS_SUCCESS == hipblasCreate((hipblasHandle_t*) cqueue_backend_data),
		"CommandQueue::CommandQueue(%d): hipblasCreate failed\n", dev_id);
	massert(HIPBLAS_STATUS_SUCCESS == hipblasSetStream(*((hipblasHandle_t*) cqueue_backend_data), stream),
		"CommandQueue::CommandQueue(%d): hipblasSetStream failed\n", dev_id);
#endif
	CoCoPeLiaSelectDevice(prev_dev_id);
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| CommandQueue::CommandQueue()\n", dev_id);
#endif
}

CommandQueue::~CommandQueue()
{
	#ifdef UDDEBUG
		lprintf(lvl, "[dev_id=%3d] |-----> CommandQueue::~CommandQueue()\n", dev_id);
	#endif
		sync_barrier();
		CoCoPeLiaSelectDevice(dev_id);
#ifdef ENABLE_PARALLEL_BACKEND
	for (int par_idx = 0; par_idx < MAX_BACKEND_L; par_idx++ ){
		hipStream_t stream = *((hipStream_t*) cqueue_backend_ptr[par_idx]);
		hipError_t err = hipStreamSynchronize(stream);
		massert(hipSuccess == err, "CommandQueue::CommandQueue - hipStreamSynchronize: %s\n", hipGetErrorString(err));
		err = hipStreamDestroy(stream);
		massert(hipSuccess == err, "CommandQueue::CommandQueue - hipStreamDestroy: %s\n", hipGetErrorString(err));
		free(cqueue_backend_ptr[par_idx]);
		hipblasHandle_t handle = *((hipblasHandle_t*) cqueue_backend_data[par_idx]);
		massert(HIPBLAS_STATUS_SUCCESS == hipblasDestroy(handle),
			"CommandQueue::CommandQueue - hipblasDestroy(handle) failed\n");
	}
#else
	hipStream_t stream = *((hipStream_t*) cqueue_backend_ptr);
	hipError_t err = hipStreamSynchronize(stream);
	massert(hipSuccess == err, "CommandQueue::CommandQueue - hipStreamSynchronize: %s\n", hipGetErrorString(err));
	err = hipStreamDestroy(stream);
	massert(hipSuccess == err, "CommandQueue::CommandQueue - hipStreamDestroy: %s\n", hipGetErrorString(err));
	free(cqueue_backend_ptr);
	hipblasHandle_t handle = *((hipblasHandle_t*) cqueue_backend_data);
	massert(HIPBLAS_STATUS_SUCCESS == hipblasDestroy(handle),
		"CommandQueue::CommandQueue - hipblasDestroy(handle) failed\n");
#endif
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
#ifdef ENABLE_PARALLEL_BACKEND
	for (int par_idx = 0; par_idx < MAX_BACKEND_L; par_idx++ ){
		hipStream_t stream = *((hipStream_t*) cqueue_backend_ptr[par_idx]);
		hipError_t err = hipStreamSynchronize(stream);
		massert(hipSuccess == err, "CommandQueue::sync_barrier - %s\n", hipGetErrorString(err));
	}
#else
	hipStream_t stream = *((hipStream_t*) cqueue_backend_ptr);
	hipError_t err = hipStreamSynchronize(stream);
	massert(hipSuccess == err, "CommandQueue::sync_barrier - %s\n", hipGetErrorString(err));
#endif
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| CommandQueue::sync_barrier()\n", dev_id);
#endif
}

void CommandQueue::add_host_func(void* func, void* data){
	get_lock();
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> CommandQueue::add_host_func()\n", dev_id);
#endif
#ifdef ENABLE_PARALLEL_BACKEND
	hipStream_t stream = *((hipStream_t*) cqueue_backend_ptr[backend_ctr]);
	hipError_t err = hipLaunchHostFunc(stream, (hipHostFn_t) func, data);
	massert(hipSuccess == err, "CommandQueue::add_host_func - %s\n", hipGetErrorString(err));
#else
	hipStream_t stream = *((hipStream_t*) cqueue_backend_ptr);
	hipError_t err = hipLaunchHostFunc(stream, (hipHostFn_t) func, data);
	massert(hipSuccess == err, "CommandQueue::add_host_func - %s\n", hipGetErrorString(err));
#endif
	release_lock();
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| CommandQueue::add_host_func()\n", dev_id);
#endif
}

void CommandQueue::wait_for_event(Event_p Wevent)
{
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> CommandQueue::wait_for_event(Event(%d))\n", dev_id, Wevent->id);
#endif
	if (Wevent->query_status() == CHECKED);
	else{
		// TODO: New addition (?)
		if (Wevent->query_status() == UNRECORDED) {
			warning("CommandQueue::wait_for_event():: UNRECORDED event\n");
			return;
		}
		get_lock();
#ifdef ENABLE_PARALLEL_BACKEND
		hipStream_t stream = *((hipStream_t*) cqueue_backend_ptr[backend_ctr]);
#else
		hipStream_t stream = *((hipStream_t*) cqueue_backend_ptr);
#endif
		hipEvent_t cuda_event= *(hipEvent_t*) Wevent->event_backend_ptr;
		release_lock();
		hipError_t err = hipStreamWaitEvent(stream, cuda_event, 0); // 0-only parameter = future NVIDIA masterplan?
		massert(hipSuccess == err, "CommandQueue::wait_for_event - %s\n", hipGetErrorString(err));
	}
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| CommandQueue::wait_for_event(Event(%d))\n", dev_id, Wevent->id);
#endif
	return;
}

#ifdef ENABLE_PARALLEL_BACKEND
int CommandQueue::request_parallel_backend()
{
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> CommandQueue::request_parallel_backend()\n", dev_id);
#endif
	get_lock();
	if (backend_ctr == MAX_BACKEND_L - 1) backend_ctr = 0;
	else backend_ctr++;
	int tmp_backend_ctr = backend_ctr;
	release_lock();
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| CommandQueue::request_parallel_backend() = %d\n", dev_id, tmp_backend_ctr);
#endif
	return tmp_backend_ctr;
}

void CommandQueue::set_parallel_backend(int backend_ctr_in)
{
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> CommandQueue::set_parallel_backend(%d)\n", dev_id, backend_ctr_in);
#endif
	get_lock();
	backend_ctr = backend_ctr_in;
	release_lock();
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| CommandQueue::set_parallel_backend(%d)\n", dev_id, backend_ctr);
#endif
	return;
}

#endif

/*****************************************************/
/// Event class functions. TODO: Do status = .. commands need lock?
Event::Event(int dev_id_in)
{
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::Event()\n", dev_id_in, Event_num_device[idxize(dev_id_in)]);
#endif
	get_lock();
	event_backend_ptr = malloc(sizeof(hipEvent_t));
	id = Event_num_device[idxize(dev_id_in)];
	Event_num_device[idxize(dev_id_in)]++;
#ifndef ENABLE_LAZY_EVENTS
	dev_id = dev_id_in;
	hipError_t err = hipEventCreate(( hipEvent_t*) event_backend_ptr);
	massert(hipSuccess == err, "Event::Event() - %s\n", hipGetErrorString(err));
#else
	dev_id = dev_id_in - 42;
#endif
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
	get_lock();
#ifndef ENABLE_LAZY_EVENTS
	Event_num_device[idxize(dev_id)]--;
	hipError_t err = hipEventDestroy(*(( hipEvent_t*) event_backend_ptr));
	massert(hipSuccess == err, "Event(%d)::~Event() - %s\n", id, hipGetErrorString(err));
#else
	if (dev_id < -1) 	Event_num_device[idxize(dev_id+42)]--;
	else{
			Event_num_device[idxize(dev_id)]--;
			hipError_t err = hipEventDestroy(*(( hipEvent_t*) event_backend_ptr));
			massert(hipSuccess == err, "Event(%d)::~Event() - %s\n", id, hipGetErrorString(err));
	}
#endif
	free(event_backend_ptr);
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
			hipEvent_t cuda_event= *(hipEvent_t*) event_backend_ptr;
			hipError_t err = hipEventSynchronize(cuda_event);
			if (status == RECORDED) status = CHECKED;
			massert(hipSuccess == err, "Event::sync_barrier() - %s\n", hipGetErrorString(err));
		}
	}
	//release_lock();
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::sync_barrier()\n", dev_id, id);
#endif
	return;
}

void Event::record_to_queue(CQueue_p Rr){
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
	hipGetDevice(&prev_dev_id);
	if (Rr->dev_id != prev_dev_id){
		CoCoPeLiaSelectDevice(Rr->dev_id);
#ifdef UDEBUG
		warning("Event(%d,dev_id = %d)::record_to_queue(%d): caller prev_dev_id=%d, changing to %d\n",
		id, dev_id, Rr->dev_id, prev_dev_id, Rr->dev_id);
#endif
	}
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
		dev_id = Rr->dev_id;
		hipError_t err = hipEventCreate(( hipEvent_t*) event_backend_ptr);
		massert(hipSuccess == err, "(Lazy)Event(%d,dev_id = %d)::record_to_queue(%d): - %s\n",
			id, dev_id, Rr->dev_id, hipGetErrorString(err));
	}
#endif
	hipEvent_t cuda_event= *(hipEvent_t*) event_backend_ptr;
#ifdef ENABLE_PARALLEL_BACKEND
	hipStream_t stream = *((hipStream_t*) Rr->cqueue_backend_ptr[Rr->backend_ctr]);
	hipError_t err = hipEventRecord(cuda_event, stream);
#else
	hipStream_t stream = *((hipStream_t*) Rr->cqueue_backend_ptr);
	hipError_t err = hipEventRecord(cuda_event, stream);
#endif
	status = RECORDED;
	massert(hipSuccess == err, "Event(%d,dev_id = %d)::record_to_queue(%d) - %s\n",  id, dev_id, Rr->dev_id, hipGetErrorString(err));
	if (Rr->dev_id != prev_dev_id){
		hipSetDevice(prev_dev_id);
	}
	release_lock();
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::record_to_queue(Queue(dev_id=%d))\n", dev_id, id, Rr->dev_id);
#endif
}

event_status Event::query_status(){
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::query_status()\n", dev_id, id);
#endif
	get_lock();
	enum event_status local_status = status;
	if (local_status != CHECKED){
#ifdef ENABLE_LAZY_EVENTS
		if (local_status == UNRECORDED){
			release_lock();
			return UNRECORDED;
		}
#endif
		hipEvent_t cuda_event= *(hipEvent_t*) event_backend_ptr;
		hipError_t err = hipEventQuery(cuda_event);

		if (err == hipSuccess && (local_status == UNRECORDED ||  local_status == COMPLETE));
		else if (err == hipSuccess && local_status == RECORDED) local_status = status = COMPLETE;
		else if (err == hipErrorNotReady && local_status == RECORDED);
		else if (err == hipErrorNotReady && local_status == UNRECORDED){
#ifdef UDEBUG
			// this should not happen in a healthy locked update scenario.
			warning("Event::query_status(): hipErrorNotReady with status == UNRECORDED should not happen\n");
#endif
			local_status = status = RECORDED;
		}
		else if (err == hipSuccess &&  local_status == CHECKED){
			;
			// TODO: This should not happen in a healthy locked update scenario.
			// But it does since no locking yet. Not sure of its effects.
#ifdef UDEBUG
			warning("[dev_id=%3d] |-----> Event(%d)::query_status(): hipSuccess with local_status == CHECKED should not happen\n", dev_id, id);
#endif
		}
		else error("[dev_id=%3d] |-----> Event(%d)::query_status() - %s, local_status=%s, status = %s\n", dev_id, id,
		hipGetErrorString(err), print_event_status(local_status), print_event_status(status));
	}
	release_lock();
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::query_status() = %s\n", dev_id, id, print_event_status(status));
#endif
	return local_status;
}

void Event::checked(){
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::checked()\n", dev_id, id);
#endif
	get_lock();
	if (status == COMPLETE) status = CHECKED;
	else error("Event::checked(): error event was %s,  not COMPLETE()\n", print_event_status(status));
	release_lock();
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::checked()\n", dev_id, id);
#endif
}

void Event::soft_reset(){
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::soft_reset()\n", dev_id, id);
#endif
	// sync_barrier();
	get_lock();
	// event_status prev_status = status;
	status = UNRECORDED;
#ifdef ENABLE_LAZY_EVENTS
	if(dev_id >= -1){
		dev_id = dev_id - 42;
		event_backend_ptr = malloc(sizeof(hipEvent_t));
	}
#endif
	release_lock();
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::soft_reset()\n", dev_id, id);
#endif
}

void Event::reset(){
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] |-----> Event(%d)::reset()\n", dev_id, id);
#endif
	sync_barrier();
	get_lock();
	event_status prev_status = status;
	status = UNRECORDED;
#ifdef ENABLE_LAZY_EVENTS
	if(dev_id >= -1){
		dev_id = dev_id - 42;
		hipError_t err = hipEventDestroy(*(( hipEvent_t*) event_backend_ptr));
		massert(hipSuccess == err, "[dev_id=%3d] (Lazy)Event(%d)::reset - %s\n", dev_id + 42, id, hipGetErrorString(err));
	}
#endif
	release_lock();
#ifdef UDDEBUG
	lprintf(lvl, "[dev_id=%3d] <-----| Event(%d)::reset()\n", dev_id, id);
#endif
}

/*****************************************************/
/// Event-based timer class functions

Event_timer::Event_timer(int dev_id) {
  Event_start = new Event(dev_id);
  Event_stop = new Event(dev_id);
  time_ms = 0;
}

void Event_timer::start_point(CQueue_p start_queue)
{
	Event_start->record_to_queue(start_queue);
}

void Event_timer::stop_point(CQueue_p stop_queue)
{
	Event_stop->record_to_queue(stop_queue);
}

double Event_timer::sync_get_time()
{
	float temp_t;
	if(Event_start->query_status() != UNRECORDED){
		Event_start->sync_barrier();
		if(Event_stop->query_status() != UNRECORDED) Event_stop->sync_barrier();
		else error("Event_timer::sync_get_time: Event_start is %s but Event_stop still UNRECORDED\n",
			print_event_status(Event_start->query_status()));
		hipEvent_t cuda_event_start = *(hipEvent_t*) Event_start->event_backend_ptr;
		hipEvent_t cuda_event_stop = *(hipEvent_t*) Event_stop->event_backend_ptr;
		hipEventElapsedTime(&temp_t, cuda_event_start, cuda_event_stop);
	}
	else temp_t = 0;
	time_ms = (double) temp_t;
	return time_ms;
}

/*****************************************************/