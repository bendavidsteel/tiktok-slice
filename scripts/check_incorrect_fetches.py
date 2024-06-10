import asyncio
import json
import os

import asyncssh
import pandas as pd
import tqdm

from convert_json_to_parquet import get_result_paths, get_remote_result_paths


def check_results(result_path):
    result_df = pd.read_parquet(result_path, columns=['result', 'args'])
    result_df['return'] = result_df['result'].map(lambda r: r['return'])
    result_df['id'] = result_df['return'].map(lambda r: r['id'] if r and 'id' in r else None)
    video_df = result_df[result_df['id'].map(lambda i: i is not None)]
    video_df.loc[:, 'id'] = video_df['id'].map(int)
    mismatched_df = video_df[video_df['id'] != video_df['args']]
    valid = len(mismatched_df) == 0
    
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
    result_paths = sorted(result_paths)

    for result_path in tqdm.tqdm(result_paths):
        if remote:
            local_result_path = os.path.join(local_result_dir, result_path.split('results/')[1])
            if not os.path.exists(os.path.dirname(local_result_path)):
                os.makedirs(os.path.dirname(local_result_path))
            await asyncssh.scp((conn, result_path), local_result_path)
            result_path = local_result_path
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


if __name__ == "__main__":
    asyncio.run(main())