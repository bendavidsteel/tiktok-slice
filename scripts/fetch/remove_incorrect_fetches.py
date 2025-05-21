import asyncio
import json
import os

import asyncssh
import pandas as pd
import tqdm

from convert_json_to_parquet import get_result_paths, get_remote_result_paths


def remove_incorrect_results(result_path):
    result_df = pd.read_parquet(result_path)
    result_df['return'] = result_df['result'].map(lambda r: r['return'])
    result_df['id'] = result_df['return'].map(lambda r: r['id'] if r and 'id' in r else None)
    video_df = result_df[result_df['id'].map(lambda i: i is not None)]
    video_df.loc[:, 'id'] = video_df['id'].map(int)
    correct_video_df = video_df[video_df['id'] == video_df['args']]
    
    correct_video_df = correct_video_df.drop(columns=['return', 'id'])
    result_df.to_parquet(result_path.replace('.parquet.gzip', '_incorrect.parquet.gzip'), compression='gzip')
    correct_video_df.to_parquet(result_path, compression='gzip')


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
            remote_result_path = result_path
            local_result_path = os.path.join(local_result_dir, result_path.split('results/')[1])
            if not os.path.exists(local_result_path):
                if not os.path.exists(os.path.dirname(local_result_path)):
                    os.makedirs(os.path.dirname(local_result_path))
                await asyncssh.scp((conn, result_path), local_result_path)
            result_path = local_result_path
            remote_valid_path = remote_result_path.replace('results.parquet.gzip', 'valid.json')
            valid_path = local_result_path.replace('results.parquet.gzip', 'valid.json')

        else:
            result_dir_path = os.path.dirname(result_path)
            valid_path = os.path.join(result_dir_path, 'valid.json')

        if not os.path.exists(valid_path):
            continue

        with open(valid_path) as f:
            valid = json.load(f)['valid']

        if valid:
            continue
        
        remove_incorrect_results(result_path)
        
        with open(valid_path, 'w') as f:
            json.dump({'valid': True}, f)

        if remote:
            await asyncssh.scp(result_path, (conn, remote_result_path))
            await asyncssh.scp(valid_path, (conn, remote_valid_path))


if __name__ == "__main__":
    asyncio.run(main())