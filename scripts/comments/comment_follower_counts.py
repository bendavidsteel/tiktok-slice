import json
import os
import numpy as np
import polars as pl
import tqdm
from pathlib import Path

def main():
    schema = {'data': 
        pl.Struct({
            'comments': pl.List(pl.Struct({'user': pl.Struct({'unique_id': pl.String, 'follower_count': pl.Int64, 'following_count': pl.Int64, 'follower_status': pl.Int64})}))
        })
    }

    # Collect all DataFrames in a list instead of progressive concatenation
    all_dfs = []
    
    comments_dir_path = Path('./data/comments')
    dir_names = [d for d in os.listdir(comments_dir_path) 
                 if not d.endswith(('.zip', '.sh'))]
    
    comment_pbar = tqdm.tqdm(total=len(dir_names), desc='Reading comments')
    
    for dir_name in dir_names:
        comment_pbar.update(1)
        
        # Determine correct path more efficiently
        nested_path = comments_dir_path / dir_name / dir_name
        comment_dir_path = nested_path if nested_path.exists() else comments_dir_path / dir_name
        
        # Get all JSON files at once
        json_files = [f for f in os.listdir(comment_dir_path) if f.endswith('.json')]
        
        for file_name in json_files:
            file_path = comment_dir_path / file_name
            
            # More specific error handling
            try:
                # Try parsing as single JSON first
                file_df = pl.read_json(file_path, schema=schema)
            except pl.exceptions.ComputeError:
                # Handle line-delimited JSON
                file_df = pl.read_ndjson(file_path, schema=schema)

            # if isinstance(file_df.schema['data'], pl.Null):
            #     continue

            # if 'comments' not in [f.name for f in file_df.schema['data'].fields]:
            #     continue

            file_df = file_df.select(pl.col('data').struct.field('comments'))\
                            .explode('comments')\
                            .select(pl.col('comments').struct.unnest())\
                            .select(pl.col('user').struct.unnest())

            # if 'uid' not in file_df.columns or 'follower_count' not in file_df.columns:
            #     continue

            all_dfs.append(file_df)
    
    comment_pbar.close()
    
    # Single concatenation and unique operation at the end
    if all_dfs:
        df = pl.concat(all_dfs)
        df = df.unique(subset=['uid'], maintain_order=True)
    else:
        df = pl.DataFrame({'uid': [], 'follower_count': []}, 
                         schema={'uid': pl.UInt64, 'follower_count': pl.UInt64})
    
    return df

if __name__ == '__main__':
    result = main()