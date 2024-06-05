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

def using_threaded_curl(ids):
    get_random_sample.thread_map(ids, function=use_single_curl, num_workers=2)

def use_single_curl(video_id):
    url = f"https://www.tiktok.com/@/video/{video_id}"
    headers = get_random_sample.get_headers()
    
    network_interface = None
    client = get_random_sample.PyCurlClient(network_interface=network_interface)
    resp = client.get(url, headers=headers)
    client.close()

    if resp.status_code >= 300:
        raise get_random_sample.InvalidResponseException(f"Status code: {resp.status_code}")

    resp_html = resp.text

    video_processor = get_random_sample.ProcessVideo()
    video_processor.process_chunk(resp_html)
    video = video_processor.process_response()
    assert video['id'] == video_id

def using_multi_curl(ids):
    # Limit the number of concurrent requests
    max_concurrent_requests = 2

    # Create a CurlMulti object
    multi = pycurl.CurlMulti()

    # Create a CurlShare object
    share = pycurl.CurlShare()
    share.setopt(pycurl.SH_SHARE, pycurl.LOCK_DATA_COOKIE)  # Share cookies
    share.setopt(pycurl.SH_SHARE, pycurl.LOCK_DATA_DNS)     # Share DNS cache
    share.setopt(pycurl.SH_SHARE, pycurl.LOCK_DATA_SSL_SESSION)  # Share SSL session IDs

    # Create a list to hold Curl objects and their buffers
    curl_objects = []
    clients = []
    urls = [f"https://www.tiktok.com/@/video/{video_id}" for video_id in ids]
    headers = get_random_sample.get_headers()
    url_queue = urls.copy()
    active_transfers = set()

    def add_transfer(url):
        client = get_random_sample.PyCurlClient(share=share)
        client._setup(url, headers=headers)
        curl_objects.append(client.c)
        clients.append(client)
        multi.add_handle(client.c)
        active_transfers.add(client.c)

    # Start the initial batch of transfers
    for _ in range(min(max_concurrent_requests, len(url_queue))):
        add_transfer(url_queue.pop(0))

    # Perform the transfers
    while active_transfers:
        while True:
            ret, num_handles = multi.perform()
            print(f'multi perform = {ret} {num_handles}')
            if ret != pycurl.E_CALL_MULTI_PERFORM:
                break

        # Wait 1s...
        ret = multi.select(1)
        print(f'multi select = {ret}')

        # Check for completed transfers and remove them
        while True:
            num_q, ok_list, err_list = multi.info_read()
            print(f'multi read = {num_q} {ok_list} {err_list}')
            for err in err_list:
                print(f"Error: {err.last_error}")

            for c in ok_list:
                index = curl_objects.index(c)
                print(clients[index]._get_response())
                multi.remove_handle(c)
                active_transfers.remove(c)

                # Start new transfers if there are more URLs in the queue
                if url_queue:
                    add_transfer(url_queue.pop(0))

            if num_q == 0:
                break

    # Cleanup
    multi.close()
    share.close()



def main():
    video_ids = ["7248300636498890011", "7248300636498890012", "7248300636498890013", "7248300636498890014"]
    # time both requests
    # start_httpx = datetime.datetime.now()
    # using_httpx()
    # end_httpx = datetime.datetime.now()
    # print(f"HTTPX: {end_httpx - start_httpx}")
    start_curl = datetime.datetime.now()
    using_threaded_curl(video_ids)
    end_curl = datetime.datetime.now()
    print(f"CURL: {end_curl - start_curl}")

    start_multi_curl = datetime.datetime.now()
    using_multi_curl(video_ids)
    end_multi_curl = datetime.datetime.now()
    print(f"MULTI CURL: {end_multi_curl - start_multi_curl}")

if __name__ == "__main__":
    main()