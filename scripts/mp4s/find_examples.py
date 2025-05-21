import os

import polars as pl

def main():
    bytes_dir_path = os.path.join('/', 'media', 'bsteel', 'TT_DAY', 'TikTok_Hour', 'mp4s')
    dir_path = os.path.join('.', 'data', 'stats', '24hour')
    video_df = pl.read_parquet(os.path.join(dir_path, 'video_class_prob_test.parquet.gzip'))

    cats = ['child_prob', 'porn_prob', 'violence_prob']
    for cat in cats:
        cat_df = video_df.sort(cat, descending=True).head(5)
        video_ids = cat_df['id'].to_list()
        video_paths = []
        num_not_found = 0
        for video_id in video_ids:
            id_bits = format(int(video_id), '064b')
            timestamp_bits = id_bits[:32]
            timestamp = int(timestamp_bits, 2)
            
            timestamp_dir = os.path.join(bytes_dir_path, str(timestamp))
            video_path = os.path.join(timestamp_dir, f'{video_id}.mp4')
            if not os.path.exists(video_path):
                num_not_found += 1
            video_paths.append(video_path)
        print(f'video_paths: {video_paths}, pct_not_found: {num_not_found / len(video_paths)} cat: {cat}')

if __name__ == '__main__':
    main()