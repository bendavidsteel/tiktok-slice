import json
import os

import matplotlib.pyplot as plt
import pandas as pd

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', 'data', 'results')

    for dir_name in os.listdir(data_dir_path):
        dir_path = os.path.join(data_dir_path, dir_name)
        with open(os.path.join(dir_path, 'parameters.json')) as f:
            params = json.load(f)
        with open(os.path.join(dir_path, 'results.json')) as f:
            results = json.load(f)

        df = pd.DataFrame(results)
        df['pre_time'] = pd.to_datetime(df['pre_time'])
        df['post_time'] = pd.to_datetime(df['post_time'])
        df['num_exceptions'] = df['exceptions'].apply(len)
        timeline_df = df[['pre_time', 'completed', 'num_exceptions']].groupby(pd.Grouper(key='pre_time', freq='1s')).count()

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        axes[0].bar(df['num_exceptions'].value_counts().index, df['num_exceptions'].value_counts())
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Number of exceptions')
        axes[0].set_ylabel('Number of occurrences')
        axes[0].set_title('Number of exceptions')
        axes[1].plot(timeline_df.index, timeline_df)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Number of fetches')
        axes[1].set_title('Fetches over time')
        axes[1].legend(timeline_df.columns)
        fig.savefig(os.path.join(dir_path, 'plot.png'))


if __name__ == '__main__':
    main()