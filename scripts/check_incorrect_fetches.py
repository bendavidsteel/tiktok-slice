import asyncio
import json
import os

import asyncssh
import pandas as pd
import tqdm

from convert_json_to_parquet import get_result_paths, get_remote_result_paths


def check_results(result_path):
    result_df = pd.read_parquet(result_path)
    result_df = result_df['result'].apply(pd.Series)
    result_df = result_df['return'].apply(pd.Series)
    
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

async def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")
    remote = True
    if remote:
        local_result_dir = os.path.join(data_dir_path, 'results')
        conn = await asyncssh.connect(os.environ['PRODESK_HOST'], username=os.environ['USERNAME'], password=os.environ['PASSWORD'])
        result_paths = await get_remote_result_paths(conn)
    else:
        result_paths = list(get_result_paths(data_dir_path))

    for result_path in tqdm.tqdm(result_paths):
        if remote:
            
        else:
            result_dir_path = os.path.dirname(result_path)
            valid_path = os.path.join(result_dir_path, 'valid.json')
            if os.path.exists(valid_path):
                continue
        
        valid = check_results(result_path)
        
        with open(valid_path, 'w') as f:
            json.dump({'valid': valid}, f)

        if remote:
            remote_valid_path = result_path.replace('results.parquet.gzip', 'valid.json')
            await asyncssh.scp(valid_path, (conn, remote_valid_path))

            


async def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")
    remote = True
    if remote:
        local_result_dir = os.path.join(data_dir_path, 'results')
        conn = await asyncssh.connect(os.environ['PRODESK_HOST'], username=os.environ['USERNAME'], password=os.environ['PASSWORD'])
        result_paths = await get_remote_result_paths(conn)
    else:
        result_paths = list(get_result_paths(data_dir_path))

    for found_result_path in tqdm.tqdm(result_paths):
        if remote:
            local_result_path = os.path.join(local_result_dir, found_result_path.split('results/')[1])
            if not os.path.exists(os.path.dirname(local_result_path)):
                os.makedirs(os.path.dirname(local_result_path))
            await asyncssh.scp((conn, found_result_path), local_result_path)
            result_path = local_result_path
        else:
            result_path = found_result_path
        parquet_path = convert_json_to_parquet(result_path)
        if remote:
            remote_parquet_path = found_result_path.replace('results.json', 'results.parquet.gzip')
            await asyncssh.scp(parquet_path, (conn, remote_parquet_path))
            # delete remote json file
            await conn.run(f'rm {found_result_path}')



if __name__ == "__main__":
    asyncio.run(main())