import httpx

import get_random_sample

def main():
    video_id = "7248300636498890011"
    with httpx.Client() as client:
        res = get_random_sample.get_video(video_id, client)

    assert res.status_code == 200

if __name__ == "__main__":
    main()