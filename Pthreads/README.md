# Pthread implementation

In this implementation CommandQueue and Event classes are running pthreads in their backend.

## CommandQueue

CommandQueue uses a **task queue** (`std::queue<pthread_task_p>`). A thread is created in the constructor, which will loop over `taskExecLoop` to execute all given tasks.

Whenever a function or method uses queue data (`struct queue_data`) the queue's **lock** must be used (`queueLock`). Some examples of such cases are:
* pop next task 
* push new task 
* any queue method (e.g. `queue::size()`)
* check terminate flag

In order to terminate a CommandQueue (destructor) it has to wait for all tasks in the queue to execute (`sync_barrier()`) and then the destructor sets the `terminate` flag to `true`. The looping thread will read this flag and return from `taskExecLoop` function. The destructor method waits for the thread to return with `pthread_join()`.

## Events

Events are implemented **as tasks** that run the `eventFunc` function, which just sets the event's inner status (`estate`) to COMPLETE. In that way when the given task gets executed in some queue, the event state will be updated to COMPLETE.

In order for a CommandQueue to wait for some event (`CommandQueue::wait_for_event()`), a task with associated function `blockQueue` is pushed in the queue. This function gets the event object as input and keeps queue thread busy while event status is UNRECORDED or RECORDED.