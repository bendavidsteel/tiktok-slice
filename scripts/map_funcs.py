import asyncio
import multiprocessing
from typing import Coroutine, Iterable, List, Optional, Callable

import tqdm
from tqdm.asyncio import tqdm as atqdm

async def aworker(
    coroutine: Coroutine,
    tasks_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    timeout: float = 1,
    callback: Optional[Callable] = None
) -> None:
    """
    A worker coroutine to process tasks from a queue.

    Args:
        coroutine: The coroutine to be applied to each task.
        tasks_queue: The queue containing the tasks to be processed.
        result_queue: The queue to put the results of each processed task.
        stop_event: An event to signal when all tasks have been added to the tasks queue.
        timeout: The timeout value for getting a task from the tasks queue.
        callback: A function that can be called at the end of each coroutine.
    """
    # Continue looping until stop_event is set and the tasks queue is empty
    while not stop_event.is_set() or not tasks_queue.empty():
        try:
            # Try to get a task from the tasks queue with a timeout
            idx, arg = await asyncio.wait_for(tasks_queue.get(), timeout)
        except asyncio.TimeoutError:
            # If no task is available, continue the loop
            continue
        try:
            # Try to execute the coroutine with the argument from the task
            result = await coroutine(arg)
            # If successful, add the result to the result queue
            result_queue.put_nowait((idx, result))

        finally:
            # Mark the task as done in the tasks queue
            tasks_queue.task_done()
            # callback for progress update
            if callback is not None:
                callback(idx, arg)


async def _amap(
    coroutine: Coroutine,
    data: Iterable,
    max_concurrent_tasks: int = 10,
    max_queue_size: int = -1,  # infinite
    callback: Optional[Callable] = None,
) -> List:
    """
    An async function to map a coroutine over a list of arguments.

    Args:
        coroutine: The coroutine to be applied to each argument.
        data: The list of arguments to be passed to the coroutine.
        max_concurrent_tasks: The maximum number of concurrent tasks.
        max_queue_size: The maximum number of tasks in the workers queue.
        callback: A function to be called at the end of each coroutine.
    """
    # Initialize the tasks queue and results queue
    # The queue size is infinite if max_queue_size is 0 or less.
    # Setting it to finite number will save some resources,
    # but will risk that an exception will be thrown too late.
    # Should be higher than the max_concurrent_tasks.
    tasks_queue = asyncio.Queue(max_queue_size)
    result_queue = asyncio.PriorityQueue()

    # Create an event to signal when all tasks have been added to the tasks queue
    stop_event = asyncio.Event()
    # Create workers
    workers = [
        asyncio.create_task(aworker(
            coroutine, tasks_queue, result_queue, stop_event, callback=callback
        ))
        for _ in range(max_concurrent_tasks)
    ]

    # Add inputs to the tasks queue
    for arg in enumerate(data):
        await tasks_queue.put(arg)
    # Set the stop_event to signal that all tasks have been added to the tasks queue
    stop_event.set()

    # Wait for all workers to complete
    # raise the earliest exception raised by a coroutine (if any)
    await asyncio.gather(*workers)
    # Ensure all tasks have been processed
    await tasks_queue.join()

    # Gather all results
    results = []
    while not result_queue.empty():
        # Get the result from the results queue and discard the index
        # Given that the results queue is a PriorityQueue, the index
        # plays a role to ensure that the results are in the same order
        # like the original list.
        _, res = result_queue.get_nowait()
        results.append(res)
    return results

async def async_amap(coroutine, data, num_workers=8, pbar_desc=None):
    pbar = atqdm(total=len(data), desc=pbar_desc)  # track progress tqdm

    def callback(*_):
        pbar.update()

    res = await _amap(coroutine, data, num_workers, callback=callback)
    pbar.close()
    return res

def process_amap(function, data, num_workers=8, pbar_desc=None):

    with multiprocessing.Pool(processes=num_workers) as pool:
        res = list(tqdm.tqdm(pool.imap(function, data), total=len(data), desc=pbar_desc))

    return res
