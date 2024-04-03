import datetime
import itertools
import json
import os

import httpx
import tqdm

class InvalidResponseException(Exception):
    pass

class NotFoundException(Exception):
    pass

def get_headers():
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-CA',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
    }
    return headers

def process_response(r):
    if r.status_code != 200:
        raise InvalidResponseException(
            r.text, f"TikTok returned a {r.status_code} status code."
        )

    start = r.text.find('<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">')
    if start == -1:
        raise InvalidResponseException(
            r.text, "Could not find normal JSON section in returned HTML."
        )

    start += len('<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">')
    end = r.text.find("</script>", start)

    if end == -1:
        raise InvalidResponseException(
            r.text, "Could not find normal JSON section in returned HTML."
        )

    data = json.loads(r.text[start:end])
    default_scope = data.get("__DEFAULT_SCOPE__", {})
    video_detail = default_scope.get("webapp.video-detail", {})
    if video_detail.get("statusCode", 0) != 0: # assume 0 if not present
        # TODO move this further up to optimize for fast fail
        if video_detail.get("statusCode", 0) == 10204:
            return None
        else:
            raise InvalidResponseException(
                r.text, "TikTok JSON had an unrecognised status code."
            )
    video_info = video_detail.get("itemInfo", {}).get("itemStruct")
    if video_info is None:
        raise InvalidResponseException(
            r.text, "TikTok JSON did not contain expected JSON."
        )
        
    return video_info

async def async_get_video(url):
    headers = get_headers()

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=headers)
    except Exception:
        raise InvalidResponseException(
            "TikTok returned an invalid response."
        )
    
    return process_response(r)

def get_video(video_id):
    url = f"https://www.tiktok.com/@therock/video/{video_id}"
    headers = get_headers()

    try:
        with httpx.Client() as client:
            r = client.get(url, headers=headers)
    except Exception:
        raise InvalidResponseException(
            "TikTok returned an invalid response."
        )
    
    return process_response(r)

def optimized_get_video(video_id):
    url = f"https://www.tiktok.com/@/video/{video_id}"
    headers = get_headers()
    
    try:
        with httpx.Client() as client:
            with client.stream("GET", url, headers=headers) as r:
                if r.status_code != 200:
                    raise InvalidResponseException(
                        r.text, f"TikTok returned a {r.status_code} status code."
                    )
                text = ""
                start = -1
                json_start = '"webapp.video-detail":'
                # json_start = '<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">'
                json_start_len = len(json_start)
                end = -1
                # json_end = '</script>'
                json_end = ',"webapp.a-b":'

                for text_chunk in r.iter_text():
                    text += text_chunk
                    if len(text) < json_start_len:
                        continue
                    if start == -1:
                        start = text.find(json_start)
                        if start != -1:
                            text = text[start + json_start_len:]
                            start = 0
                    if start != -1:
                        end = text.find(json_end)
                        if end != -1:
                            text = text[:end]
                            break

                if start == -1 or end == -1:
                    raise InvalidResponseException(
                        text, "Could not find normal JSON section in returned HTML."
                    )
                video_detail = json.loads(text)
                # default_scope = data.get("__DEFAULT_SCOPE__", {})
                # video_detail = default_scope.get("webapp.video-detail", {})
                if video_detail.get("statusCode", 0) != 0: # assume 0 if not present
                    # TODO move this further up to optimize for fast fail
                    return video_detail
                video_info = video_detail.get("itemInfo", {}).get("itemStruct")
                if video_info is None:
                    raise InvalidResponseException(
                        r.text, "TikTok JSON did not contain expected JSON."
                    )
                return video_info
    except Exception as ex:
        raise InvalidResponseException(
            f"TikTok returned an invalid response: {ex}"
        )

class FuncWrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, arg):
        
        pre_time = datetime.datetime.now()
        try:
            res = self.func(arg)
            exception = None
        except Exception as e:
            res = None
            exception = e
        post_time = datetime.datetime.now()

        return {
            'res': res,
            'exception': exception,
            'pre_time': pre_time,
            'post_time': post_time,
        }

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    
    with open(os.path.join(this_dir_path, '..', 'figs', 'all_videos', 'all_found_segments_combinations.json'), 'r') as file:
        data = json.load(file)

    # get bits of non timestamp sections of ID
    # order dict according to interval
    data = [(tuple(map(int, interval.strip('()').split(', '))), vals) for interval, vals in data.items()]
    data = sorted(data, key=lambda x: x[0][0])
    # get rid of millisecond bits
    data = [t for t in data if t[0] != (0,9)]
    interval_bits = []
    for interval, vals in data:
        # format ints to binary
        num_bits = interval[1] - interval[0] + 1
        bits = [format(i, f'0{num_bits}b') for i in vals]
        interval_bits.append(bits)
    other_bit_sequences = itertools.product(*interval_bits)
    other_bit_sequences = [''.join(bits) for bits in other_bit_sequences]

    # get all videos in 1 millisecond
    num_time = 1
    time_unit = 'ms'
    unit_map = {
        'ms': 'milliseconds',
        's': 'seconds',
    }
    time_delta = datetime.timedelta(**{unit_map[time_unit]: num_time})
    start_time = datetime.datetime(2024, 3, 1, 20, 0, 0)
    end_time = start_time + time_delta
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
    num_workers = 1
    reqs_per_ip = -1
    task_batch_size = 1
    task_nthreads = 1
    worker_cpu = 256
    worker_mem = 512
    cluster_type = 'local'
    # r = await async_map(test_real_video, potential_video_ids, num_workers=64)
    results = []
    func = FuncWrapper(optimized_get_video)
    for video_id in tqdm.tqdm(potential_video_ids):
        results.append(func(video_id))

    num_hits = len([r for r in results if r.result and r.result['res'] is not None])
    num_valid = len([r for r in results if r.completed])
    print(f"Num hits: {num_hits}, Num valid: {num_valid}, Num potential video IDs: {len(potential_video_ids)}")
    print(f"Fraction hits: {num_hits / num_valid}")
    print(f"Fraction valid: {num_valid / len(potential_video_ids)}")
    # convert to jsonable format
    json_results = [
        {
            'args': r.args, 
            'exceptions': [{
                    'exception': str(e['exception']),
                    'pre_time': e['pre_time'].isoformat(),
                    'post_time': e['post_time'].isoformat()
                }
                for e in r.exceptions
            ], 
            'result': {
                'return': r.result['res'] if r.result is not None else None,
                'pre_time': r.result['pre_time'].isoformat() if r.result is not None else None,
                'post_time': r.result['post_time'].isoformat() if r.result is not None else None
            },
            'completed': r.completed
        }
        for r in results
    ]

    results_dir_path = os.path.join(this_dir_path, '..', 'data', 'results')
    results_dirs = [dir_name for dir_name in os.listdir(results_dir_path)]
    new_result_dir = str(max([int(d) for d in results_dirs]) + 1) if results_dirs else '0'
    os.mkdir(os.path.join(results_dir_path, new_result_dir))

    params = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'num_time': num_time,
        'time_unit': time_unit,
        'num_workers': num_workers,
        'reqs_per_ip': reqs_per_ip,
        'task_batch_size': task_batch_size,
        'worker_nthreads': task_nthreads,
        'worker_cpu': worker_cpu,
        'worker_mem': worker_mem,
        'cluster_type': cluster_type,
    }

    with open(os.path.join(this_dir_path, '..', 'data', 'results', new_result_dir, 'parameters.json'), 'w') as f:
        json.dump(params, f)

    with open(os.path.join(this_dir_path, '..', 'data', 'results', new_result_dir, 'results.json'), 'w') as f:
        json.dump(json_results, f)

if __name__ == "__main__":
    main()