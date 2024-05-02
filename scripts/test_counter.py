import json
import math
import multiprocessing
import os

import httpx
import pandas as pd
import matplotlib.pyplot as plt

from get_random_sample import get_video
from map_funcs import process_amap

def read_result_path(result_path):
    with open(result_path, 'r') as f:
        try:
            results = json.load(f)
        except:
            return []
    return [r for r in results]

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")

    all_videos = []
    fetched_videos = []
    result_paths = []
    for dir_path, dir_names, filenames in os.walk(os.path.join(data_dir_path, 'results')):
        for filename in filenames:
            if filename == 'results.json':
                result_paths.append(os.path.join(dir_path, filename))
    # for result_path in tqdm.tqdm(result_paths, desc="Reading result files"):
    
    all_results = process_amap(read_result_path, result_paths, num_workers=multiprocessing.cpu_count() - 1, pbar_desc="Reading result files")
    all_results = [v for res in all_results for v in res]
    fetched_results = [r for r in all_results if r['result']['return']]
    fetched_video_ids = [r['args'] for r in fetched_results]
    video_bits = [format(v, '064b') for v in fetched_video_ids]

    # get distribution of number of videos per millisecond
    bit_sections = [{'time_bits': int(b[:32], 2) + int(b[32:42], 2) / 1000, 'counter_bits': b[32+10:32+18], 'geo_bits': b[32+18:]} for b in video_bits]
    df = pd.DataFrame(bit_sections)
    df['result'] = fetched_results
    df['id'] = df['result'].apply(lambda x: x['args'])
    df = df.drop_duplicates(subset='id')
    df['statusMsg'] = df['result'].apply(lambda x: x['result']['return']['statusMsg'] if 'statusMsg' in x['result']['return'] else 'success')
    df['counter_vals'] = df['counter_bits'].apply(lambda x: int(x, 2))
    

    def get_groups(df):
        geo_groups_df = df.groupby('geo_bits')
        geo_groups = {}
        for geo_group in geo_groups_df.groups:
            geo_df = geo_groups_df.get_group(geo_group)
            geo_groups[geo_group] = geo_df[['time_bits', 'counter_vals']].groupby('time_bits').agg(list)['counter_vals'].to_dict()
        return geo_groups
    
    request_groups = get_groups(df)
    # success_groups = get_groups(df[df['statusMsg'] != "item doesn't exist"])

    # for geo_bits in request_groups:
    #     request_group = request_groups[geo_bits]
    #     success_group = success_groups[geo_bits]
    #     num_milliseconds_requests = len(request_group)
    #     num_milliseconds_success = len(success_group)
    #     percent_milliseconds_with_success = num_milliseconds_success / num_milliseconds_requests
    #     # if we did get successful requests, where all of the requests successful?
    #     if num_milliseconds_success > 0:
    #         all_successful = all([len(request_group[time_group]) == len(success_group[time_group]) for time_group in success_group])
    #         if not all_successful:
    #             for time_group in success_group:
                    

    with httpx.Client() as client:
        # test missing counts
        for geo_bits in request_groups:
            # check if we find a sequence of vals with some missing vals
            time_groups = request_groups[geo_bits]
            for time in time_groups:
                vals = time_groups[time]
                min_val = min(vals)
                max_val = max(vals)
                missing_vals = [val for val in range(min_val, max_val) if val not in vals]

                # create a tiktok id from these missing vals
                missing_bits = [format(val, '08b') for val in missing_vals]
                timestamp = math.floor(time_group[0])
                timestamp_bits = format(timestamp, '032b')
                milliseconds = math.floor((time_group[0] - timestamp) * 1000)
                milliseconds_bits = format(milliseconds, '010b')
                geo_bits = time_group[1]
                for missing_bit in missing_bits:
                    missing_id = int(timestamp_bits + milliseconds_bits + missing_bit + geo_bits, 2)
                    # res = get_video(missing_id, client)
                    pass

if __name__ == '__main__':
    main()