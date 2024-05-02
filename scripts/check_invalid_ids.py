import collections
import json
import os
import multiprocessing

from map_funcs import process_amap

def invalid_item_id(result):
    ret = result['result']['return']
    if not ret:
        return False
    # only return false if definite confirmation that it doesn't exist or is cannot have existed
    if 'statusCode' in ret:
        if ret['statusCode'] == 100002 and ret['statusMsg'] == 'invalid item id':
            return True

    return False

def read_result_path(result_path):
    with open(os.path.join(result_path, 'results.json'), 'r') as f:
        try:
            results = json.load(f)
        except:
            return [], []
    # if len([r['args'] for r in results if invalid_item_id(r)]) > 0:
    #     with open(os.path.join(result_path, 'parameters.json'), 'r') as f:
    #         return json.load(f)
    return [r['args'] for r in results if invalid_item_id(r)], [r['args'] for r in results if not invalid_item_id(r)]

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")

    all_videos = []
    fetched_videos = []
    result_paths = []
    for dir_path, dir_names, filenames in os.walk(os.path.join(data_dir_path, 'results')):
        for filename in filenames:
            if filename == 'results.json':
                result_paths.append(dir_path)
    # for result_path in tqdm.tqdm(result_paths, desc="Reading result files"):
    
    results = process_amap(read_result_path, result_paths, num_workers=multiprocessing.cpu_count() - 1, pbar_desc="Reading result files")
    invalid_ids = [result[0] for result in results]
    other_ids = [result[1] for result in results]
    num_invalid_ids = sum([len(invalid_id) for invalid_id in invalid_ids])
    num_other_ids = sum([len(other_id) for other_id in other_ids])
    percent_invalid_ids = num_invalid_ids / (num_invalid_ids + num_other_ids) * 100
    print(f"Percent invalid ids: {percent_invalid_ids:.2f}%")

    with open(os.path.join(this_dir_path, '..', 'figs', 'all_videos', 'all_two_segments_combinations.json'), 'r') as file:
        combinations = json.load(file)

    invalid_ids = [item for sublist in invalid_ids for item in sublist]
    invalid_bits = [format(invalid_id, '064b') for invalid_id in invalid_ids]
    timestamps = [int(invalid_bit[:32], 2) for invalid_bit in invalid_bits]
    milliseconds = [int(invalid_bit[32:32+10], 2) for invalid_bit in invalid_bits]
    counters = [int(invalid_bit[32+10:32+18], 2) for invalid_bit in invalid_bits]
    geo_bits = [invalid_bit[32+18:] for invalid_bit in invalid_bits]
    # print(collections.Counter(timestamps))
    # print(collections.Counter(milliseconds))
    # print(collections.Counter(counters))
    # print(collections.Counter(geo_bits))
    for interval, ids in combinations.items():
        interval = tuple(map(int, interval.strip('()').split(', ')))
        invalid_interval_bits = [invalid_bit[32+interval[0]:32+interval[1]+1] for invalid_bit in invalid_bits]
        invalid_interval_ids = [int(invalid_interval_bit, 2) for invalid_interval_bit in invalid_interval_bits]
        # check if invalid interval bits are in the found bits
        assert all([invalid_interval_id in ids for invalid_interval_id in invalid_interval_ids]), f"Invalid interval bits not in found bits: {invalid_interval_bits}"

if __name__ == '__main__':
    main()