import os

import datamapplot
import numpy as np
import pandas as pd
import polars as pl


def main():
    use = '1hour'
    if use == '1hour':
        data_dir_path = './data/topic_model_videos_toponymy'
        title = "Every video from one hour on TikTok"
        sub_title = 'A data map of every TikTok video <a href="https://arxiv.org/pdf/2504.13279">we collected</a> between 5-6pm UTC on 2024-04-10'
    elif use == '24hour':
        data_dir_path = './data/topic_model_videos_toponymy_24hour'
        title = "One minute of videos from each of the 24 hours on TikTok"
        sub_title = 'A data map of every TikTok video <a href="https://arxiv.org/pdf/2504.13279" target="_blank">we collected</a> from the 42<sup>nd</sup> minute of each hour on 2024-04-10 CET'

    topic_df = pl.read_parquet(os.path.join(data_dir_path, 'video_topics.parquet.gzip'))
    topic_cols = sorted([col for col in topic_df.columns if 'topic_layer' in col])

    topic_df = topic_df.with_columns([pl.col(col).replace('Unlabelled', '0') for col in topic_cols])

    max_desc_len = 20
    topic_df = topic_df.with_columns([
        pl.from_epoch(pl.col('createTime')),
        pl.when(pl.col('playCount') > 1e6).then((pl.col('playCount').cast(pl.UInt64) / 1e6).round(decimals=1).cast(pl.String) + pl.lit('m'))\
            .when(pl.col('playCount') > 1e3).then((pl.col('playCount').cast(pl.UInt64) / 1e3).round(decimals=1).cast(pl.String) + pl.lit('k'))\
            .otherwise(pl.col('playCount').cast(pl.UInt64).cast(pl.String)).alias('playCountStr'),
        pl.col('playCount').log1p().alias('playCountLog1p'),
        pl.when(pl.col('desc').str.len_chars() > max_desc_len).then(pl.col('desc').str.slice(0, max_desc_len) + pl.lit('...')).otherwise(pl.col('desc')).alias('desc'),
    ])

    topic_df = topic_df.with_columns(
        pl.concat_str(
            [
                pl.format("Desc: {}", pl.col('desc')),
            ],
            separator='\n'
        ).alias('hover_text')
    )

    kwargs = {}
    if use == '24hour':
        kwargs = {
            'histogram_data': topic_df['createTime'].to_numpy(),
            'histogram_n_bins': 24,
            'histogram_group_datetime_by': 'hour',
            'histogram_range': (pd.to_datetime('2024-04-09 22:00:00'), pd.to_datetime('2024-04-10 21:59:59')),
            'histogram_settings': {
                'histogram_title': 'Hours of the Day'
            }
        }

    sample = None
    if sample:
        topic_df = topic_df.head(sample)

    # Create a datamap plot
    plot = datamapplot.create_interactive_plot(
        topic_df['map'].to_numpy(),
        *[topic_df[col].to_numpy() for col in topic_cols],
        # hover_text=topic_df['hover_text'].to_numpy(),
        noise_label='0',
        title=title,
        sub_title=sub_title,
        enable_search=True,
        darkmode=True,
        marker_size_array=topic_df['playCountLog1p'].to_numpy(),
        font_family="Cinzel",
        minify_deps=True,
        **kwargs
    )

    plot.save(f"{use}_toponymy_datamapplot.html")

if __name__ == '__main__':
    main()
