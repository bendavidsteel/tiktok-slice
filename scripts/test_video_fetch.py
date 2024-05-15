import datetime
import io

import brotli
import certifi
import httpx
import pycurl

import get_random_sample

def using_httpx():
    video_id = "7248300636498890011"
    with httpx.Client() as client:
        res = get_random_sample.get_video(video_id, client)

    assert res['id'] == video_id

class ResponseHeaders:
    def __init__(self):
        self.headers = {}

    def header_function(self, header_line):
        header_line = header_line.decode('iso-8859-1')

        if ':' not in header_line:
            return

        name, value = header_line.split(':', 1)

        name = name.strip()
        value = value.strip()

        name = name.lower()

        self.headers[name] = value
    

def using_curl():
    video_id = "7248300636498890011"
    url = f"https://www.tiktok.com/@/video/{video_id}"
    c = pycurl.Curl()
    buffer = io.BytesIO()
    response_headers = ResponseHeaders()
    c.setopt(pycurl.URL, url)
    c.setopt(pycurl.HTTPHEADER, [f"{key}: {value}" for key, value in get_random_sample.get_headers().items()])
    c.setopt(pycurl.TIMEOUT, 10)
    c.setopt(pycurl.WRITEFUNCTION, buffer.write)
    c.setopt(pycurl.HEADERFUNCTION, response_headers.header_function)
    c.setopt(pycurl.CAINFO, certifi.where())
    network_interface = None
    if network_interface:
        c.setopt(pycurl.INTERFACE, network_interface)
    c.perform()

    # Json response
    resp_bytes = buffer.getvalue()

    if response_headers.headers['content-encoding'] == 'br':
        resp = brotli.decompress(resp_bytes).decode()
    else:
        raise get_random_sample.InvalidResponseException("Content encoding not supported")

    buffer.close()
    c.close()

    # if res.statuscode != 0:
    #     raise get_random_sample.InvalidResponseException(res.stdout.decode())
    video_processor = get_random_sample.ProcessVideo()
    do = video_processor.process_chunk(resp)
    video = video_processor.process_response()
    assert video['id'] == video_id

def main():
    # time both requests
    # start_httpx = datetime.datetime.now()
    # using_httpx()
    # end_httpx = datetime.datetime.now()
    # print(f"HTTPX: {end_httpx - start_httpx}")
    start_curl = datetime.datetime.now()
    using_curl()
    end_curl = datetime.datetime.now()
    print(f"CURL: {end_curl - start_curl}")

if __name__ == "__main__":
    main()