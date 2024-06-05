import concurrent.futures
import json
import os

import pandas as pd
import tqdm


def get_result_paths(data_dir_path):

    for dir_path, dir_names, filenames in os.walk(os.path.join(data_dir_path, 'results')):
        for filename in filenames:
            if 'results' in filename:
                result_path = os.path.join(dir_path, filename)
                yield result_path

def write_parquet(results, result_path):
    df = pd.DataFrame(results)
    parquet_path = result_path.replace('results.json', 'results.parquet.gzip')
    df.to_parquet(parquet_path, compression='gzip')
    return parquet_path

def check_results(result_path):
    if result_path.endswith('.parquet.gzip'):
        results = pd.read_parquet(result_path).to_dict('records')
    elif result_path.endswith('.json'):
        with open(result_path, 'r') as f:
            try:
                results = json.load(f)
            except:
                return
    else:
        raise ValueError(f"Unknown file type: {result_path}")
    
    valid = True
    # check if any video ids don't match up with args
    def check_result(result):
        if result['result']['return'] and 'id' in result['result']['return'] and result['result']['return']['id']:
            if result['args'] != int(result['result']['return']['id']):
                return False
            else:
                return True
        else:
            return True
            
    valid = all(check_result(result) for result in results)

    return valid

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")
    result_paths = list(get_result_paths(data_dir_path))

    for result_path in tqdm.tqdm(result_paths):
        result_dir_path = os.path.dirname(result_path)
        valid_path = os.path.join(result_dir_path, 'valid.json')
        if os.path.exists(valid_path):
            continue
        
        valid = check_results(result_path)
        if not valid:
            print(f"Invalid results: {result_path}")
        
        with open(valid_path, 'w') as f:
            json.dump({'valid': valid}, f)



if __name__ == "__main__":
    main()