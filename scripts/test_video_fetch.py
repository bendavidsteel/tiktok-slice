import datetime
import io

import brotli
import certifi
import httpx
import pycurl

import get_random_sample

def using_httpx(ids):
    video_id = "7248300636498890011"
    with httpx.Client() as client:
        res = get_random_sample.get_video(video_id, client)

    assert res['id'] == video_id

def using_threaded_curl(ids):
    get_random_sample.thread_map(ids, use_single_curl, 5)

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
    max_concurrent_requests = 3

    # Create a CurlMulti object
    multi = pycurl.CurlMulti()

    # Create a CurlShare object
    share = pycurl.CurlShare()
    share.setopt(pycurl.SH_SHARE, pycurl.LOCK_DATA_COOKIE)  # Share cookies
    share.setopt(pycurl.SH_SHARE, pycurl.LOCK_DATA_DNS)     # Share DNS cache
    share.setopt(pycurl.SH_SHARE, pycurl.LOCK_DATA_SSL_SESSION)  # Share SSL session IDs

    # Create a list to hold Curl objects and their buffers
    curl_objects = []
    buffers = []
    url_queue = urls.copy()
    active_transfers = set()

    def add_transfer(url):
        buffer = BytesIO()
        c = pycurl.Curl()
        c.setopt(pycurl.URL, url)
        c.setopt(pycurl.WRITEFUNCTION, buffer.write)
        c.setopt(pycurl.CAINFO, certifi.where())  # Ensure SSL certificates are verified
        c.setopt(pycurl.SHARE, share)  # Share session data
        curl_objects.append(c)
        buffers.append(buffer)
        multi.add_handle(c)
        active_transfers.add(c)

    # Start the initial batch of transfers
    for _ in range(min(max_concurrent_requests, len(url_queue))):
        add_transfer(url_queue.pop(0))

    # Perform the transfers
    while active_transfers:
        while True:
            ret, num_handles = multi.perform()
            if num_handles == 0:
                break
            multi.select(1.0)

        # Check for completed transfers and remove them
        while True:
            num_q, ok_list, err_list = multi.info_read()
            if num_q == 0:
                break

            for c in ok_list:
                index = curl_objects.index(c)
                print(buffers[index].getvalue().decode('utf-8'))
                buffers[index].close()
                multi.remove_handle(c)
                active_transfers.remove(c)

                # Start new transfers if there are more URLs in the queue
                if url_queue:
                    add_transfer(url_queue.pop(0))

        # Sleep briefly to avoid busy-waiting
        time.sleep(0.1)

    # Cleanup
    multi.close()
    share.close()



def main():
    video_ids = ["7248300636498890011"]
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