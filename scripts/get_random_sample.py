import asyncio
import collections
import datetime
import itertools
import json
import multiprocessing
import os
import random
import urllib.parse as url_parsers

import httpx
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm

import asyncio
from typing import Callable, Coroutine, Iterable, List, Optional


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


async def amap(
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

async def amap_all(coroutine, data, num_workers=10):
    pbar = atqdm(total=len(data))  # track progress tqdm

    def callback(*_):
        pbar.update()

    res = await amap(coroutine, data, num_workers, callback=callback)
    pbar.close()
    return res

def map_pool(function, data):
    with multiprocessing.Pool(10) as pool:
        res = list(tqdm(pool.imap(function, data), total=len(data)))  # track progress tqdm
    pool.join()
    return res

class InvalidResponseException(Exception):
    pass

class NotFoundException(Exception):
    pass

async def get_video(url):
    headers = {
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/121.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Chromium";v="121", "Not A(Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Linux"'
    }

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=headers)
    except Exception:
        raise InvalidResponseException(
            "TikTok returned an invalid response."
        )
    if r.status_code != 200:
        raise InvalidResponseException(
            r.text, "TikTok returned an invalid response."
        )

    # Try SIGI_STATE first
    # extract tag <script id="SIGI_STATE" type="application/json">{..}</script>
    # extract json in the middle

    # start = r.text.find('<script id="SIGI_STATE" type="application/json">')
    # if start != -1:
    #     start += len('<script id="SIGI_STATE" type="application/json">')
    #     end = r.text.find("</script>", start)

    #     if end == -1:
    #         raise InvalidResponseException(
    #             r.text, "TikTok returned an invalid response.", error_code=r.status_code
    #         )

    #     data = json.loads(r.text[start:end])
    #     video_info = data["ItemModule"][self.id]
    # else:
    # Try __UNIVERSAL_DATA_FOR_REHYDRATION__ next

    # extract tag <script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">{..}</script>
    # extract json in the middle

    start = r.text.find('<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">')
    if start == -1:
        raise InvalidResponseException(
            r.text, "TikTok returned an invalid response."
        )

    start += len('<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">')
    end = r.text.find("</script>", start)

    if end == -1:
        raise InvalidResponseException(
            r.text, "TikTok returned an invalid response."
        )

    data = json.loads(r.text[start:end])
    default_scope = data.get("__DEFAULT_SCOPE__", {})
    video_detail = default_scope.get("webapp.video-detail", {})
    if video_detail.get("statusCode", 0) != 0: # assume 0 if not present
        # TODO move this further up to optimize for fast fail
        if video_detail.get("statusCode", 0) == 10204:
            raise NotFoundException(
                r.text, "TikTok indicated that the content does not exist."
            )
        else:
            raise InvalidResponseException(
                r.text, "TikTok returned an invalid response structure."
            )
    video_info = video_detail.get("itemInfo", {}).get("itemStruct")
    if video_info is None:
        raise InvalidResponseException(
            r.text, "TikTok returned an invalid response structure."
        )
        
    return video_info

async def is_get_existing_video(video_id):

    if not isinstance(video_id, list):
        video_ids = [video_id]
    else:
        video_ids = video_id
        preds = []

    ms_token = None
    for valid_video_id in video_ids:
        url = f"https://www.tiktok.com/@therock/video/{valid_video_id}"
        try:
            video_data = await get_video(url)
            predicted_truth = 'exists'
        except NotFoundException:
            predicted_truth = 'not exists'
            video_data = None
        except InvalidResponseException:
            predicted_truth = 'invalid response'
            video_data = None
        if isinstance(video_id, list):
            preds.append((predicted_truth, video_data))
        else:
            return predicted_truth, video_data
    if isinstance(video_id, list):
        return preds
   
        
async def test_real_video(video_id):
    does_exist, video_data = await is_get_existing_video(video_id)
    if does_exist == 'exists':
        return 1, 1, video_data
    elif does_exist == 'invalid response':
        return 0, 0, video_data
    else:
        return 0, 1, video_data

def test_1_month_ago(i):
    timestamp_1year_time = int((datetime.datetime.now() - datetime.timedelta(days=30)).timestamp())
    # convert to binary
    timestamp_binary = format(timestamp_1year_time, '032b')
    # create random 32 bit number
    random_32bit = format(random.getrandbits(32), '032b')
    # concatenate into 64 bit number
    random_video_id = int(timestamp_binary + random_32bit, 2)
    does_exist = is_existing_video(random_video_id)
    if does_exist == 'exists':
        return 1, 1, None
    elif does_exist == 'invalid response':
        return 0, 0, None
    else:
        video = get_existing_video(random_video_id)
        return 0, 1, video

def test_1_year_ahead(i):
    timestamp_1year_time = int((datetime.datetime.now() + datetime.timedelta(days=365)).timestamp())
    # convert to binary
    timestamp_binary = format(timestamp_1year_time, '032b')
    # create random 32 bit number
    random_32bit = format(random.getrandbits(32), '032b')
    # concatenate into 64 bit number
    random_video_id = int(timestamp_binary + random_32bit, 2)
    does_exist = is_existing_video(random_video_id)
    if does_exist == 'exists':
        return 1, 1, None
    elif does_exist == 'invalid response':
        return 0, 0, None
    else:
        video = get_existing_video(random_video_id)
        return 0, 1, video

def iterate_binary(b):
    pass

async def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    if False:
        # test real
        
        data_dir_path = os.path.join(this_dir_path, "..", "data", "germany")
        with open(os.path.join(data_dir_path, 'videos', 'all_010324.json'), 'r') as file:
            videos = json.load(file)

        num_test = 1000

        r = await amap_all(test_real_video, [videos[i]['id'] for i in range(num_test)])
        score = sum([x[0] for x in r if x[1] == 1])
        num_valid = sum([x[1] for x in r])

        assert score == num_valid
        print(f"Score for real video IDs: {score / num_valid}")
        print(f"Number of valid responses: {num_valid / num_test}")

        # # test random 1 month ago timestamps
        r = await amap_all(test_1_month_ago, range(num_test))
        score = sum([x[0] for x in r if x[1] == 1])
        num_valid = sum([x[1] for x in r])

        print(f"Score for real video IDs 1 month ago: {score / num_valid}")
        print(f"Number of valid responses: {num_valid / num_test}")

        # test random 1 year ahead timestamps
        r = await amap_all(test_1_year_ahead, range(num_test))
        score = sum([x[0] for x in r if x[1] == 1])
        num_valid = sum([x[1] for x in r])

        print(f"Score for real video IDs 1 year ahead: {score / num_valid}")
        print(f"Number of valid responses: {num_valid / num_test}")

    if True:
        with open(os.path.join(this_dir_path, '..', 'figs', '0_999_found_segments_combinations.json'), 'r') as file:
            data = json.load(file)

        # get bits of non timestamp sections of ID
        # order dict according to interval
        data = [(tuple(map(int, interval.strip('()').split(', '))), vals) for interval, vals in data.items()]
        data = sorted(data, key=lambda x: x[0][0])
        # get rid of millisecond bits
        data = data[1:]
        interval_bits = []
        for interval, vals in data:
            # format ints to binary
            num_bits = interval[1] - interval[0] + 1
            bits = [format(i, f'0{num_bits}b') for i in vals]
            interval_bits.append(bits)
        other_bit_sequences = itertools.product(*interval_bits)
        other_bit_sequences = [''.join(bits) for bits in other_bit_sequences]

        # get all videos in 1 millisecond
        start_time = datetime.datetime(2024, 3, 1, 20, 0, 0)
        end_time = start_time + datetime.timedelta(milliseconds=10)
        start_timestamp = start_time.timestamp()
        end_timestamp = end_time.timestamp()
        c_time = start_timestamp
        all_timestamp_bits = []
        while c_time < end_timestamp:
            unix_timestamp_bits = format(int(c_time), '032b')
            milliseconds = int(format(c_time, '.3f').split('.')[1])
            milliseconds_bits = format(milliseconds, '010b')
            timestamp_bits = unix_timestamp_bits + milliseconds_bits
            all_timestamp_bits.append(timestamp_bits)
            c_time += 0.001

        potential_video_bits = itertools.product(all_timestamp_bits, other_bit_sequences)
        potential_video_bits = [''.join(bits) for bits in potential_video_bits]
        potential_video_ids = [int(bits, 2) for bits in potential_video_bits]
        r = await amap_all(test_real_video, potential_video_ids, num_workers=64)
        num_hits = sum([x[0] for x in r if x[1] == 1])
        num_valid = sum([x[1] for x in r])
        print(f"Num hits: {num_hits}, Num valid: {num_valid}, Num potential video IDs: {len(potential_video_ids)}")
        print(f"Fraction hits: {num_hits / num_valid}")
        print(f"Fraction valid: {num_valid / len(potential_video_ids)}")

if __name__ == '__main__':
    asyncio.run(main())