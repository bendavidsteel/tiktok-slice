import datetime

import httpx
import tqdm

from get_random_sample import get_video

def main():
    to_try = {
        '00110100011110': [
            10, 11, 12, 13, 14, 15, 16, 17, 19
        ],
        '00110100101010': [
            12, 13, 16
        ]
    }
    # TODO how principled should we be about this?

    start_time = datetime.datetime(2024, 4, 10, 19, 0, 0)
    end_time = datetime.datetime(2024, 4, 10, 19, 30, 0)
    times = []
    ctime = start_time
    while ctime < end_time:
        times.append(ctime)
        ctime += datetime.timedelta(milliseconds=1)

    # with httpx.Client() as client:
        # test missing counts
    for geo_bits in to_try:

        for missing_val in to_try[geo_bits]:
            for time in tqdm.tqdm(times, desc=f"Testing missing {missing_val}, geo {geo_bits}"):
                # create a tiktok id from these missing vals
                missing_bits = format(missing_val, '08b')
                timestamp = int(time.timestamp())
                timestamp_bits = format(timestamp, '032b')
                milliseconds = time.microsecond // 1000
                milliseconds_bits = format(milliseconds, '010b')
                missing_id = int(timestamp_bits + milliseconds_bits + missing_bits + geo_bits, 2)
                try:
                    res = get_video(missing_id, None)
                    if 'statusCode' not in res:
                        raise ValueError(f"Missing id {missing_id} is valid")
                except Exception as e:
                    print(e)
                    continue

if __name__ == '__main__':
    main()