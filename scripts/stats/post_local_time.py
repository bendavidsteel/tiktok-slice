import configparser
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
import tqdm

import polars as pl
import pytz
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

def get_result_paths(result_dir_path, result_filename='results.parquet.gzip', minute=None, hour=None):
    for dir_path, dir_names, filenames in os.walk(result_dir_path):
        for filename in filenames:
            if filename == result_filename:
                file_hour, file_minute = map(int, dir_path.split('/')[-3:-1])
                if hour is not None and file_hour != hour:
                    continue
                if minute is not None and file_minute != minute:
                    continue
                result_path = os.path.join(dir_path, filename)
                yield result_path

def get_country_timezone(country_code: str) -> str:
    """
    Get the primary timezone for a country code using pytz.
    Falls back to UTC if country code is not found.
    """
    try:
        # Get all timezones
        timezones = pytz.country_timezones(country_code)
        return timezones if timezones else 'UTC'
    except KeyError:
        return 'UTC'

def convert_to_local_hour(ts, tz_list: List[str]) -> int:
    """Convert UTC timestamp to local hour in specified timezone."""
    try:
        local_times = []
        for tz_str in tz_list:
            timezone = pytz.timezone(tz_str)
            local_time = ts.replace(tzinfo=pytz.UTC).astimezone(timezone)
            local_times.append(local_time.hour)
        return sum(local_times) // len(local_times)
    except Exception:
        return None

def convert_to_local_time(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert UTC timestamps to local time based on country codes.
    
    Args:
        df: Polars DataFrame with 'createTime' and 'locationCreated' columns
        
    Returns:
        DataFrame with additional 'local_hour' column
    """
    # Create a mapping of unique country codes to their timezones
    unique_countries = df.get_column('locationCreated').unique()
    tz_mapping = {
        country: get_country_timezone(country) 
        for country in unique_countries 
        if country is not None
    }
    
    # Define the conversion function
    def get_local_hour(row) -> Optional[int]:
        country = row['locationCreated']
        if country is None or country not in tz_mapping:
            return None
        return convert_to_local_hour(row['createTime'], tz_mapping[country])
    
    rows = df.select(['createTime', 'locationCreated']).to_dicts()
    df = df.with_columns(pl.Series('local_hour', [get_local_hour(row) for row in rows]))

    # Add local_hour column using apply
    return df

def plot_local_time_histogram(df: pl.DataFrame, 
                            output_path = None) -> None:
    """
    Create a histogram of post times in local time.
    
    Args:
        df: Polars DataFrame with 'local_hour' column
        output_path: Optional path to save the plot
    """
    plt.figure(figsize=(4, 3))
    
    df = df.filter(pl.col('local_hour').is_not_null())

    # Get hour counts using Polars
    hour_counts = df.group_by('local_hour')\
        .agg((pl.count('local_hour') / df.shape[0]).alias('share'))\
        .sort('local_hour')
    
    # Plot using the Polars data
    plt.bar(hour_counts['local_hour'], hour_counts['share'])
    
    # plt.title('Distribution of TikTok Posts by Local Hour')
    plt.xlabel('Hour of Day (Local Time)')
    plt.ylabel('Share of Posts')
    
    # Set x-axis ticks for each hour
    # plt.xticks(range(0, 24))
    
    # Add grid for better readability
    # plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)

def main():
    config = configparser.ConfigParser()
    config.read('./config/config.ini')

    base_result_path = os.path.join('.', 'data', 'results', '2024_04_10')
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    use = '24hour'
    if use == 'all':
        output_dir_path = os.path.join(this_dir_path, '..', "..", "data", "stats", 'all')
        result_paths = list(get_result_paths(base_result_path, result_filename='videos.parquet.zstd'))
    elif use == '24hour':
        output_dir_path = os.path.join(this_dir_path, '..', "..", "data", "stats", '24hour')
        result_paths = list(get_result_paths(base_result_path, result_filename='videos.parquet.zstd', minute=42))
    elif use == '1hour':
        output_dir_path = os.path.join(this_dir_path, '..', "..", "data", "stats", '1hour')
        result_paths = list(get_result_paths(base_result_path, result_filename='videos.parquet.zstd', hour=19))

    result_paths = sorted(result_paths)

    os.makedirs(output_dir_path, exist_ok=True)

    result_paths = result_paths[:5]
    video_df = None
    val_count_dfs = None
    for result_path in tqdm.tqdm(result_paths):
        batch_df = pl.read_parquet(result_path)
        if video_df is not None:
            video_df = pl.concat([video_df, batch_df], how='diagonal_relaxed')
        else:
            video_df = batch_df

    video_df = video_df.with_columns(pl.from_epoch('createTime'))

    plot_local_time_histogram(convert_to_local_time(video_df), './figs/local_time_histogram.png')

if __name__ == '__main__':
    main()