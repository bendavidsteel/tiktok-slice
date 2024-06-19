import os

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")
    video_topic_df = pd.read_parquet(os.path.join(data_dir_path, "video_topics.parquet.gzip"))
    topic_info_df = pd.read_parquet(os.path.join(data_dir_path, "topic_info.parquet.gzip"))
    num_topics = topic_info_df.shape[0] - 1
    topic_info_df = topic_info_df[topic_info_df['Representation'].map(lambda r: all(w != '' for w in r[:4]))]
    topic_info_df['Name'] = topic_info_df['Representation'].map(lambda r: "_".join(r[:4]))
    topic_name_mapper = {row['Topic']: row['Name'] for i, row in topic_info_df.iterrows()}

    video_topic_df = video_topic_df[video_topic_df['video'].map(lambda v: 'stats' in v and 'playCount' in v['stats'])]
    video_topic_df = video_topic_df.join(pd.get_dummies(video_topic_df['topic']))
    video_topic_df = video_topic_df.rename(columns=topic_name_mapper)

    video_topic_df['duration'] = video_topic_df['video'].map(lambda v: v['video']['duration'])
    video_topic_df['verified'] = video_topic_df['video'].map(lambda v: v['author']['verified']).astype(np.int64)
    video_topic_df['original_music'] = video_topic_df['video'].map(lambda v: v['music']['original']).astype(np.int64)
    video_topic_df['is_duet'] = video_topic_df['video'].map(lambda v: v['duetInfo'] is not None).astype(np.int64)

    topic_names = list(topic_name_mapper.values())
    video_topic_df[topic_names] = video_topic_df[topic_names].astype(np.float64)
    feature_names = topic_names + ['duration', 'verified', 'original_music', 'is_duet']
    X = video_topic_df[feature_names]
    X = sm.add_constant(X)

    video_topic_df['views'] = video_topic_df['video'].map(lambda v: v['stats']['playCount'])
    y = video_topic_df['views']

    model = sm.OLS(y, X)
    results = model.fit()
    results_df = pd.DataFrame({
        'feature': results.params.index,
        'coefficient': results.params.values,
        'p_value': results.pvalues.values
    })
    print(results_df[results_df['p_value'] < 0.05])
    print(results.summary())

if __name__ == "__main__":
    main()