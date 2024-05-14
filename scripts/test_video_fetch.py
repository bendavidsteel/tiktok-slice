import datetime
import subprocess

import httpx

import get_random_sample

def using_httpx():
    video_id = "7248300636498890011"
    with httpx.Client() as client:
        res = get_random_sample.get_video(video_id, client)

    assert res['id'] == video_id

def using_curl():
    video_id = "7248300636498890011"
    res = subprocess.run(["curl", "--interface", "wlan0", f"https://www.tiktok.com/@/video/{video_id}"], capture_output=True)
    if res.statuscode != 0:
        raise get_random_sample.InvalidResponseException(res.stdout.decode())
    video_processor = get_random_sample.ProcessVideo()
    do = video_processor.process_chunk(res.stdout.decode())
    video = video_processor.process_response()
    assert video['id'] == video_id

def main():
    # time both requests
    start_httpx = datetime.datetime.now()
    using_httpx()
    end_httpx = datetime.datetime.now()
    print(f"HTTPX: {end_httpx - start_httpx}")
    start_curl = datetime.datetime.now()
    using_curl()
    end_curl = datetime.datetime.now()
    print(f"CURL: {end_curl - start_curl}")

if __name__ == "__main__":
    main()