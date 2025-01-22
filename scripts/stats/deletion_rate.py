import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import polars as pl
import tqdm

def main():
    error_df = pl.DataFrame()
    video_dir_path = os.path.join('.', 'data', 'results', '2024_04_10')
    video_pbar = tqdm.tqdm(total=60*60 + 24*60, desc='Reading videos')
    for root, dirs, files in os.walk(video_dir_path):
        for file in files:
            if file == 'errors.parquet.zstd':
                video_pbar.update(1)
                result_path = os.path.join(root, file)
                batch_error_df = pl.read_parquet(result_path)
                batch_error_df = batch_error_df.filter(pl.col('statusMsg') != "item doesn't exist")
                error_df = pl.concat([error_df, batch_error_df], how='diagonal_relaxed')

    all_over_time_df = error_df.sort('post_time').group_by_dynamic('post_time', every='1w').agg(pl.col('statusMsg').count().alias('count_all'))

    fig, ax = plt.subplots()
    statuses = ['success', 'status_deleted', 'status_self_see', 'author_secret', 'status_reviewing']

    # Define the exponential function to fit
    def func(x, a, b, c, d):
        return c * np.log(b * (x - a)) + d

    for status in statuses:
        status_over_time_df = error_df.filter(pl.col('statusMsg') == status).sort('post_time').group_by_dynamic('post_time', every='1w').agg(pl.col('statusMsg').count().alias('count'))
        status_over_time_df = status_over_time_df.join(all_over_time_df, on='post_time', how='left')
        status_over_time_df = status_over_time_df.with_columns((pl.col('count') / pl.col('count_all')).alias('rate'))
        ax.plot(status_over_time_df['post_time'], status_over_time_df['rate'], label=status)
        
        if status == 'status_deleted':
            post_time = datetime.datetime(2024, 4, 10, 0, 0, 0)
            rate = 0
            post_times = status_over_time_df['post_time'].to_list()
            rates = status_over_time_df['rate'].to_list()
            post_times = [post_time] + post_times
            rates = [rate] + rates
            ax.scatter(post_times, rates, label=status)
            
            # Convert timestamps to numerical values for fitting
            time_nums = np.array([(t - post_time).total_seconds() / (60 * 60 * 24) for t in post_times])
            rates = np.array(rates)
            
            # Fit exponential curve
            weights = 1 / (1 + np.arange(time_nums.shape[0]))  # Higher weights for earlier points
            popt, _ = curve_fit(
                func, 
                time_nums, 
                rates,
                p0=[-1, 1, 0.02, 0],
                sigma=weights,
                method='trf',  # Try Trust Region Reflective algorithm
                loss='soft_l1',  # More robust to outliers
                maxfev=10000
            )
            
            # Generate points for smooth curve
            fit_times = np.linspace(min(time_nums), max(time_nums), 100)
            fit_rates = func(fit_times, *popt)
            
            # Convert back to datetime for plotting
            fit_datetimes = [post_time + datetime.timedelta(days=t) for t in fit_times]
            
            # Plot the fitted curve
            ax.plot(fit_datetimes, fit_rates, '--', color='gray', label='Fit')

    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.legend()
    fig.savefig('./figs/errors_over_time.png')

if __name__ == '__main__':
    main()