import asyncio
import json
import os

import pandas as pd
import tqdm


def get_result_paths(data_dir_path):

    for dir_path, dir_names, filenames in os.walk(os.path.join(data_dir_path, 'results')):
        for filename in filenames:
            if filename == 'results.json':
                result_path = os.path.join(dir_path, filename)
                yield result_path

def cleanup(result):
    if result['result']['return'] and 'item_control' in result['result']['return'] and len(result['result']['return']['item_control']) == 0:
        result['result']['return']['item_control'] = None
    return result

def write_parquet(results, result_path):
    pq_results = [cleanup(r) for r in results]
    df = pd.DataFrame(pq_results)
    parquet_path = result_path.replace('results.json', 'results.parquet.gzip')
    df.to_parquet(parquet_path, compression='gzip')
    return parquet_path

def convert_json_to_parquet(result_path):
    
    
    try:
        with open(result_path, 'r') as f:
            results = json.load(f)
    except json.JSONDecodeError:
        with open(result_path, 'r') as f:
            json_string = f.read()
            while True:
                if not json_string:
                    raise ValueError("Couldn't fix JSON")
                try:
                    results = json.loads(json_string + "]")
                except json.decoder.JSONDecodeError:
                    last_right_curly = json_string.rfind("}")
                    if last_right_curly == -1:
                        raise ValueError("Couldn't fix JSON")
                    elif last_right_curly == len(json_string) - 1:
                        json_string = json_string[:last_right_curly]
                    json_string = json_string[:last_right_curly+1]
                    continue
                break
    parquet_path = write_parquet(results, result_path)
    
    # df = pd.read_parquet(parquet_path)
    # # check contents same as results
    # df_results = df.to_dict('records')
    # assert len(df_results) == len(results)
    # for df_result, result in zip(df_results, results):
    #     assert df_result['args'] == result['args']
    #     if 'id' in result:
    #         assert df_result['id'] == result['id']
    #         assert df_result['video'] == result['video']
    #     elif 'statusCode' in result:
    #         assert int(df_result['statusCode']) == int(result['statusCode'])
    #     assert len(df_result['exceptions']) == len(result['exceptions'])
    #     assert df_result['completed'] == result['completed']
    os.remove(result_path)

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")
    result_paths = list(get_result_paths(data_dir_path))

    for result_path in tqdm.tqdm(result_paths):
        convert_json_to_parquet(result_path)



if __name__ == "__main__":
    main()