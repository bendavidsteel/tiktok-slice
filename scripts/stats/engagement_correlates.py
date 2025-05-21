import os

import numpy as np
import polars as pl
import statsmodels.api as sm
from tqdm import tqdm

def do_regression(indep_df, dep_df):
    dep_vars = ['playCount']#['commentCount', 'diggCount', 'shareCount', 'playCount']

    indep_cols = indep_df.columns
    df = pl.concat([indep_df, dep_df], how='horizontal')
    # remove nulls and nans
    df = df.drop_nulls()
    df = df.drop_nans()
    indep_df = df.select(indep_cols)
    dep_df = df.select(dep_vars)

    X = indep_df.to_pandas()

    # check for colinearity
    corr_matrix = X.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                print(f"High correlation between {corr_matrix.columns[i]} and {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]}")
    
    X = sm.add_constant(X)

    for dep_var in dep_vars:
        y = dep_df.select(dep_var).to_pandas()

        model = sm.OLS(y, X)
        results = model.fit()
        print(f"Results for {dep_var}, R^2: {results.rsquared}")
        results_df = pl.DataFrame({
            'feature': results.params.index,
            'coefficient': results.params.values,
            'p_value': results.pvalues.values
        })
        predictive_df = results_df.filter(pl.col('p_value') < 0.05)\
            .filter(pl.col('coefficient').abs() > 0.01)\
            .filter(pl.col('feature').is_in(indep_cols))
        if len(predictive_df) > 0:
            if len(predictive_df) > 10:
                print(predictive_df.sort('coefficient').head(5))
                print(predictive_df.sort('coefficient', descending=True).head(5))
            else:
                print(predictive_df)

        # check for problems
        if results.condition_number > 1000:
            print(f"High condition number for {dep_var}: {results.condition_number}")
        if results.eigenvals[-1] < 1e-10:
            print(f"Low eigenvalue for {dep_var}: {results.eigenvals[-1]}")

def drop_low_sum_cols(df):
    for col in df.columns:
        if df[col].sum() < 100:
            df = df.drop(col)
    return df

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", '..', "data", 'topic_model_videos')

    video_df = pl.DataFrame()
    video_dir_path = os.path.join('.', 'data', 'results', '2024_04_10', 'hours', '19')
    video_pbar = tqdm(total=60*60, desc='Reading videos')
    for root, dirs, files in os.walk(video_dir_path):
        for file in files:
            if file == 'videos.parquet.zstd':
                video_pbar.update(1)
                result_path = os.path.join(root, file)
                batch_video_df = pl.read_parquet(result_path)
                batch_video_df = batch_video_df.select([
                    pl.col('video_id'),
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
                video_df = pl.concat([video_df, batch_video_df], how='diagonal_relaxed')

    # TODO remove
    # video_df = video_df.sample(n=1000000, seed=42)

    video_df = video_df.with_columns(pl.col('desc').str.extract_all('#[a-zA-Z0-9_]+').alias('hashtags'))
    hashtag_df = video_df.select(['video_id', 'hashtags']).explode('hashtags').filter(pl.col('hashtags').is_not_null())
    hashtag_count_df = hashtag_df['hashtags'].value_counts().sort('count').filter(pl.col('count') > 50)

    dep_df = video_df.select(['diggCount', 'commentCount', 'shareCount', 'playCount'])

    # normalize dep_df
    dep_df = dep_df.with_columns([
        pl.col('diggCount').log1p(),
        pl.col('commentCount').log1p(),
        pl.col('shareCount').log1p(),
        pl.col('playCount').log1p()
    ]).with_columns([
        ((pl.col('diggCount') - pl.col('diggCount').mean()) / pl.col('diggCount').std()).alias('diggCount'),
        ((pl.col('commentCount') - pl.col('commentCount').mean()) / pl.col('commentCount').std()).alias('commentCount'),
        ((pl.col('shareCount') - pl.col('shareCount').mean()) / pl.col('shareCount').std()).alias('shareCount'),
        ((pl.col('playCount') - pl.col('playCount').mean()) / pl.col('playCount').std()).alias('playCount')
    ])

    location_dummies_df = video_df['locationCreated'].to_dummies().drop('locationCreated_null')
    location_dummies_df = drop_low_sum_cols(location_dummies_df)
    do_regression(location_dummies_df, dep_df)
    location_dummies_df = None
    hashtag_dummies_df = video_df.select('video_id').join(
        hashtag_df.join(hashtag_count_df.drop('count'), on='hashtags', how='right')\
            .to_dummies('hashtags')\
            .group_by('video_id')\
            .sum(),
        on='video_id',
        how='left',
        maintain_order='left'
    ).drop('video_id').fill_null(0)
    hashtag_dummies_df = drop_low_sum_cols(hashtag_dummies_df)
    do_regression(hashtag_dummies_df, dep_df)
    hashtag_dummies_df = None
    other_df = video_df.select(['authorVerified', 'musicOriginal', 'videoDuration'])\
        .with_columns([
            pl.col('authorVerified').cast(pl.Int32),
            pl.col('musicOriginal').cast(pl.Int32)
        ])
    do_regression(other_df, dep_df)
    other_df = None
    diversification_dummies_df = video_df['diversificationLabels'].to_dummies().drop('diversificationLabels_null')
    do_regression(diversification_dummies_df, dep_df)
    diversification_dummies_df = None
    

if __name__ == "__main__":
    main()