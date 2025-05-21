import concurrent.futures
import datetime
import time

import httpx
import pycurl

import get_random_sample

def using_httpx(ids):
    video_id = "7248300636498890011"
    with httpx.Client() as client:
        res = get_random_sample.get_video(video_id, client)

    assert res['id'] == video_id

def using_threaded_curl(ids, max_workers=8):
    results = get_random_sample.thread_map(ids, function=use_single_curl, num_workers=max_workers)
    assert len(results) == len(ids)
    num_errors = 0
    for result, video_id in zip(results, ids):
        if 'video' in result:
            video = result['video']
            if 'id' in video:
                assert video['id'] == video_id
        elif 'error' in result:
            num_errors += 1
    print(f"Error percentage: {num_errors / len(ids) * 100:.3f}%")

def use_single_curl(video_id):
    try:
        url = f"https://www.tiktok.com/@/video/{video_id}"
        headers = get_random_sample.get_headers()
        
        network_interface = None
        client = get_random_sample.PyCurlClient(network_interface=network_interface)
        resp = client.get(url, headers=headers)
        client.close()

        if resp.status_code >= 300:
            raise get_random_sample.InvalidResponseException(f"Status code: {resp.status_code}")

        resp_html = resp.text

        video = get_random_sample.process_video(resp_html, headers=resp.headers)
    except Exception as e:
        return {'error': str(e), 'id': video_id}
    else:
        return {'video': video, 'id': video_id}

def using_multi_curl(ids, max_workers=8):

    # Create a CurlMulti object
    multi = pycurl.CurlMulti()
    multi.setopt(pycurl.M_MAX_TOTAL_CONNECTIONS, max_workers)

    # Create a CurlShare object
    share = pycurl.CurlShare()
    share.setopt(pycurl.SH_SHARE, pycurl.LOCK_DATA_COOKIE)  # Share cookies
    share.setopt(pycurl.SH_SHARE, pycurl.LOCK_DATA_DNS)     # Share DNS cache
    share.setopt(pycurl.SH_SHARE, pycurl.LOCK_DATA_SSL_SESSION)  # Share SSL session IDs

    # Create a list to hold Curl objects and their buffers
    client_map = {}
    id_map = {}
    headers = get_random_sample.get_headers()

    def add_transfer(video_id):
        url = f"https://www.tiktok.com/@/video/{video_id}"
        client = get_random_sample.PyCurlClient(share=share)
        client._setup(url, headers=headers)
        client_map[id(client.c)] = client
        id_map[id(client.c)] = video_id
        multi.add_handle(client.c)

    def process_video(video_id, text, headers):
        try:
            video = get_random_sample.process_video(text, headers)
        except Exception as e:
            return {'error': str(e), 'id': video_id}
        else:
            return {'video': video, 'id': video_id}
    
    def process_error(video_id, error):
        return {'error': error, 'id': video_id}

    # Start the initial batch of transfers
    for video_id in ids:
        add_transfer(video_id)

    futures = []
    curl_processed_count = 0
    curl_total = len(ids)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while curl_processed_count < curl_total:
            # Perform the transfers
            while True:
                ret, num_handles = multi.perform()
                if ret != pycurl.E_CALL_MULTI_PERFORM:
                    break

            # Wait 1s...
            ret = multi.select(0.1)

            # Check for completed transfers and remove them
            while True:
                num_q, ok_list, err_list = multi.info_read()
                for c, error_num, error_message in err_list:
                    future = executor.submit(process_error, id_map[id(c)], error_message)
                    futures.append(future)
                    curl_processed_count += 1

                for c in ok_list:
                    resp = client_map[id(c)]._get_response()
                    video_id = id_map[id(c)]
                    future = executor.submit(process_video, video_id, resp.text, headers=resp.headers)
                    futures.append(future)
                    multi.remove_handle(c)
                    curl_processed_count += 1

                if num_q == 0:
                    break

        results = [future.result() for future in futures]

    # Cleanup
    multi.close()
    share.close()
    
    # reorder based on original ids order
    results = sorted(results, key=lambda x: ids.index(x['id']))
    assert len(results) == len(ids)
    num_errors = 0
    for result, video_id in zip(results, ids):
        if 'video' in result:
            video = result['video']
            if 'id' in video:
                assert video['id'] == video_id
        elif 'error' in result:
            num_errors += 1
    print(f"Error percentage: {num_errors / len(ids) * 100:.3f}%")



def main():
    first_id = 7248300636498890011
    video_ids = [str(first_id + i) for i in range(50)]
    # time both requests
    # start_httpx = datetime.datetime.now()
    # using_httpx()
    # end_httpx = datetime.datetime.now()
    # print(f"HTTPX: {end_httpx - start_httpx}")
    max_workers = 8

    start_multi_curl = datetime.datetime.now()
    using_multi_curl(video_ids, max_workers=max_workers)
    end_multi_curl = datetime.datetime.now()
    print(f"MULTI CURL: {end_multi_curl - start_multi_curl}")

    start_curl = datetime.datetime.now()
    using_threaded_curl(video_ids, max_workers=max_workers)
    end_curl = datetime.datetime.now()
    print(f"CURL: {end_curl - start_curl}")


if __name__ == "__main__":
    main()