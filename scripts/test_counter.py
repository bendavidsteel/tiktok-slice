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
    
    success_groups = get_groups(df[df['statusMsg'] != "item doesn't exist"])

    with open(os.path.join(this_dir_path, '..', 'figs', 'all_videos', 'all_two_segments_combinations.json'), 'r') as f:
        all_two_segments_combinations = json.load(f)

    requested_ids = all_two_segments_combinations['(10, 31)']

    # TODO look to see if likely workers in the same datacenter have the same sequence system

    valid_ids = []
    with httpx.Client() as client:
        # test missing counts
        for geo_bits in success_groups:
            # check if we find a sequence of successful requests with missing vals, where we didn't make the request
            time_groups = success_groups[geo_bits]
            missing_vals_top_contendors = set()
            for time in time_groups:
                success_vals = time_groups[time]
                min_val = min(success_vals)
                max_val = max(success_vals)
                missing_vals = [val for val in range(min_val, max_val) if val not in success_vals]
                if len(missing_vals) / len(success_vals) < 0.2: # suspicious if less than of vals are missing
                    for missing_val in missing_vals:
                        missing_bits = format(missing_val, '08b')
                        all_bits = missing_bits + geo_bits
                        missing_id = int(all_bits, 2)
                        if missing_id not in requested_ids:
                            missing_vals_top_contendors.add(missing_val)

            for missing_val in missing_vals_top_contendors:
                for time in time_groups:
                    success_vals = time_groups[time]
                    min_val = min(success_vals)
                    max_val = max(success_vals)
                    time_missing_vals = [val for val in range(min_val, max_val) if val not in success_vals]
                    if missing_val not in time_missing_vals:
                        continue
                    # create a tiktok id from these missing vals
                    missing_bits = format(missing_val, '08b')
                    timestamp = math.floor(time)
                    timestamp_bits = format(timestamp, '032b')
                    milliseconds = math.floor((time - timestamp) * 1000)
                    milliseconds_bits = format(milliseconds, '010b')
                    missing_id = int(timestamp_bits + milliseconds_bits + missing_bits + geo_bits, 2)
                    res = get_video(missing_id, client)
                    if 'statusCode' not in res:
                        valid_ids.append(int(missing_bits + geo_bits, 2))

if __name__ == '__main__':
    main()