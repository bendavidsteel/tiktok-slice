import os

import polars as pl
from tqdm import tqdm

def write_stats(df: pl.DataFrame, path):
    os.makedirs(path, exist_ok=True)

    # write engagement distributions
    engagement_types = ['share', 'digg', 'comment', 'play']
    for engagement_type in engagement_types:
        df[f'{engagement_type}Count'].value_counts()\
            .sort(f'{engagement_type}Count')\
            .write_csv(os.path.join(path, f'{engagement_type}_count.csv'))

    pl.from_epoch(df['createTime']).value_counts().sort('createTime').write_csv(os.path.join(path, 'create_time_count.csv'))

    df['locationCreated'].value_counts().write_csv(os.path.join(path, 'location_created_count.csv'))

    df.select(['createTime', 'locationCreated']).write_csv(os.path.join(path, 'createtime_location_created.csv'))

    # breakdown by country
    location_dir = os.path.join(path, 'location_created')
    os.makedirs(location_dir, exist_ok=True)
    for location_created in df['locationCreated'].unique().drop_nulls():
        location_path = os.path.join(location_dir, location_created)
        os.makedirs(location_path, exist_ok=True)
        df.filter(pl.col('locationCreated') == location_created)[f'{engagement_type}Count'].value_counts()\
            .sort(f'{engagement_type}Count', descending=True)\
            .write_csv(os.path.join(location_path, f'{engagement_type}_count.csv'))

def write_topic_data(topic_desc_df: pl.DataFrame, hour_video_df: pl.DataFrame, hour_dir):
    topic_dir = os.path.join(hour_dir, 'topics')
    os.makedirs(topic_dir, exist_ok=True)

    # write topic description
    topic_desc_df.drop('Count')\
        .join(
            hour_video_df.filter(pl.col('topic').is_not_null())['topic'].value_counts(), 
            left_on='Topic', 
            right_on='topic', 
            maintain_order='left'
        )\
        .rename({'count': 'Count'})\
        .write_parquet(os.path.join(topic_dir, 'topic_desc.parquet'))
    
    for topic in topic_desc_df['Topic']:
        this_topic_dir = os.path.join(topic_dir, str(topic))
        os.makedirs(this_topic_dir, exist_ok=True)
        hour_video_df.filter(pl.col('topic') == topic)['playCount'].value_counts().sort('playCount').write_csv(os.path.join(this_topic_dir, 'play_count_dist.csv'))
        hour_video_df.filter(pl.col('topic') == topic)['locationCreated'].value_counts().write_csv(os.path.join(this_topic_dir, 'location_created_dist.csv'))

def write_child_data(df: pl.DataFrame, dir):
    child_dir = os.path.join(dir, 'children')
    os.makedirs(child_dir, exist_ok=True)

    df.filter(pl.col('child_present') == 1)['playCount'].value_counts().sort('playCount').write_csv(os.path.join(child_dir, 'play_count_dist.csv'))
    df.group_by('locationCreated')\
        .agg(pl.col('child_present').sum().alias('num_child_videos'), pl.col('child_present').count().alias('num_videos'))\
        .with_columns((pl.col('num_child_videos') / pl.col('num_videos')).alias('child_share'))\
        .write_csv(os.path.join(child_dir, 'country_share_videos_with_children.csv'))

    for location_created in df['locationCreated'].unique().drop_nulls():
        location_dir = os.path.join(child_dir, location_created)
        os.makedirs(location_dir, exist_ok=True)
        df.filter(pl.col('locationCreated') == location_created)\
            .filter(pl.col('child_present') == 1)['playCount'].value_counts()\
            .sort('playCount')\
            .write_csv(os.path.join(location_dir, 'play_count_dist.csv'))
    
def write_comment_stats(df: pl.DataFrame, comment_df: pl.DataFrame, dir):
    comment_dir = os.path.join(dir, 'comments')
    os.makedirs(comment_dir, exist_ok=True)

    comment_df = comment_df.join(df.select(['video_id']), left_on='aweme_id', right_on='video_id', how='right')

    comment_df['comment_language'].value_counts().write_csv(os.path.join(comment_dir, 'comment_language_counts.csv'))
    pl.from_epoch(comment_df['create_time']).value_counts().sort('create_time').write_csv(os.path.join(comment_dir, 'create_time_dist.csv'))


def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", '..', "data", 'topic_model_videos')

    hour_video_df = pl.DataFrame()
    day_video_df = pl.DataFrame()
    video_dir_path = os.path.join('.', 'data', 'results', '2024_04_10', 'hours')
    video_pbar = tqdm(total=60*60 + 23 * 60, desc='Reading videos')
    for root, dirs, files in os.walk(video_dir_path):
        for file in files:
            if file == 'videos.parquet.zstd':
                video_pbar.update(1)
                root_sections = root.split('/')
                hour, minute, second = root_sections[-3], root_sections[-2], root_sections[-1]
                result_path = os.path.join(root, file)
                batch_video_df = pl.read_parquet(result_path)
                batch_video_df = batch_video_df.select([
                    pl.col('video_id'),
                    pl.col('createTime'),
                    pl.col('authorVerified'),
                    pl.col('musicOriginal'),
                    pl.col('videoDuration'),
                    pl.col('videoQuality'),
                    pl.col('locationCreated'),
                    pl.col('desc'),
                    pl.col('shareCount'),
                    pl.col('diggCount'),
                    pl.col('commentCount'),
                    pl.col('playCount'),
                    pl.col('diversificationLabels')
                ])
                if minute == '42':
                    day_video_df = pl.concat([day_video_df, batch_video_df], how='diagonal_relaxed')
                if hour == '19':
                    hour_video_df = pl.concat([hour_video_df, batch_video_df], how='diagonal_relaxed')

    # load topics data
    topic_desc_df = pl.read_parquet('./data/topic_model_videos/topic_desc.parquet.gzip')
    post_topic_df = pl.read_parquet('./data/topic_model_videos/video_topics.parquet.gzip')
    post_topic_df = post_topic_df.with_columns(pl.col('image_path').str.split('/').list.get(-1).str.split('.').list.get(0).alias('video_id'))
    hour_video_df = hour_video_df.join(post_topic_df, on='video_id', how='left')
    hour_video_df = hour_video_df.unique('video_id')

    child_video_df = pl.read_parquet(os.path.join('.', 'data', 'stats', '24hour', 'video_class_prob_test.parquet.gzip'))
    
    # Process video data
    threshold = 0.43
    child_video_df = child_video_df.with_columns([
        pl.col('id').alias('video_id'),
        (pl.col('child_prob') > threshold).cast(pl.Int32).alias('child_present')
    ])

    root_path = '../tiktok-hour'

    hour_dir = os.path.join(root_path, 'hour')
    day_dir = os.path.join(root_path, 'day')

    write_stats(day_video_df, day_dir)
    write_stats(hour_video_df, hour_dir)

    write_topic_data(topic_desc_df, hour_video_df, hour_dir)

    child_video_df = child_video_df.join(day_video_df, on='video_id', how='left')
    write_child_data(child_video_df, day_dir)

    comments_dir_path = os.path.join('.', 'data', 'comments')
    comment_df = None
    comment_pbar = tqdm(total=len(list(os.listdir(comments_dir_path))), desc='Reading comments')
    for dir_name in os.listdir(comments_dir_path):
        comment_pbar.update(1)
        if dir_name.endswith('.zip'):
            continue
        comment_path = os.path.join(comments_dir_path, dir_name, 'comments.parquet.zstd')
        if not os.path.exists(comment_path):
            continue
        file_df = pl.read_parquet(comment_path)
        if comment_df is None:
            comment_df = file_df
        else:
            comment_df = pl.concat([comment_df, file_df], how='diagonal_relaxed')

    day_video_df = day_video_df.join(comment_df, left_on='video_id', right_on='aweme_id', how='left')
    hour_video_df = hour_video_df.join(comment_df, left_on='video_id', right_on='aweme_id', how='left')

    write_comment_stats(day_video_df, comment_df, day_dir)
    write_comment_stats(hour_video_df, comment_df, hour_dir)

    hour_df = pl.read_csv('./data/stats/1hour/time_counts.csv')
    day_df = pl.read_csv('./data/stats/24hour/time_counts.csv')

    hour_df = hour_df.with_columns(pl.col('createTime').cast(pl.Datetime))
    day_df = day_df.with_columns(pl.col('createTime').cast(pl.Datetime))

    hour_df.sort('createTime').write_csv(os.path.join(hour_dir, 'id_timestamp_counts.csv'))
    day_df.sort('createTime').write_csv(os.path.join(day_dir, 'id_timestamp_counts.csv'))


if __name__ == '__main__':
    main()
