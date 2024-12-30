import json
import os

import polars as pl
import tqdm

def main():
    comment_cols = ['aweme_id', 'cid', 'uid', 'create_time', 'text']

    comments_dir_path = os.path.join('.', 'data', 'comments')
    comment_pbar = tqdm.tqdm(total=len(list(os.listdir(comments_dir_path))), desc='Reading comments')
    for dir_name in os.listdir(comments_dir_path):
        comment_pbar.update(1)
        if dir_name.endswith('.zip') or dir_name.endswith('.sh'):
            continue
        df = None
        comment_path = os.path.join(comments_dir_path, dir_name, 'comments.parquet.zstd')
        if os.path.exists(comment_path):
            comment_df = pl.read_parquet(comment_path)
            if all(col in comment_df.columns for col in comment_cols):
                continue
        if os.path.exists(os.path.join(comments_dir_path, dir_name, dir_name)):
            comment_dir_path = os.path.join(comments_dir_path, dir_name, dir_name)
        else:
            comment_dir_path = os.path.join(comments_dir_path, dir_name)
        for file_name in os.listdir(comment_dir_path):
            
            if file_name.endswith('.json'):
                file_path = os.path.join(comment_dir_path, file_name)
                with open(file_path, 'r') as f:
                    t = f.read()
                try:
                    d = json.loads(t)
                    file_df = pl.from_dicts(d['data']['comments'])
                except:
                    ds = [json.loads(l) for l in t.split('\n') if l]
                    file_dfs = [pl.from_dicts(d['data']['comments']) for d in ds if d['data'] and 'comments' in d['data'] and d['data']['comments']]
                    if len(file_dfs) == 0:
                        continue
                    file_df = pl.concat(file_dfs, how='diagonal_relaxed')
                file_df = file_df.with_columns(pl.col('user').struct.field('uid').cast(pl.UInt64).alias('uid'))
                file_df = file_df.select(comment_cols)
                if df is None:
                    df = file_df
                else:
                    df = pl.concat([df, file_df])
        df.write_parquet(comment_path, compression='zstd')


if __name__ == '__main__':
    main()