import datetime
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import polars as pl
import tqdm

def calculate_observed_metrics(comment_df: pl.DataFrame):
    """
    Calculate the actual metrics from observed data that we'll compare against
    """
    # Get comment counts per author to model popularity
    author_counts = comment_df.group_by('authorUniqueId').len()
    
    # Calculate interaction groups as in original code
    comment_interactions = comment_df.group_by('uid').agg(
        pl.col('authorUniqueId').alias('author_list')
    )
    
    comment_interactions = comment_interactions.with_columns([
        pl.col('author_list').list.len().alias('num_interactions'),
        pl.col('author_list').list.unique().alias('unique_authors'),
        pl.col('author_list').list.unique().list.len().alias('num_unique_authors')
    ])
    
    # Get stats about author overlap groups
    author_groups = comment_interactions.group_by('unique_authors').agg([
        pl.col('uid').len().alias('group_size')
    ])
    
    return {
        'author_counts': author_counts,
        'interaction_stats': comment_interactions,
        'author_groups': author_groups
    }

def run_monte_carlo_simulation(comment_df: pl.DataFrame, n_simulations=10):
    """
    Run Monte Carlo simulation to estimate probability of observed comment patterns
    
    Args:
        comment_df: Original comment dataframe
        n_simulations: Number of simulations to run
    """
    # Get observed metrics
    observed = calculate_observed_metrics(comment_df)
    
    # Get basic parameters for simulation
    n_comments = len(comment_df)
    n_commenters = comment_df.select(['uid']).n_unique()
    n_authors = comment_df.select(['authorUniqueId']).n_unique()
    
    # Calculate author popularity distribution
    author_counts = observed['author_counts'].sort('len', descending=True)
    probabilities = author_counts['len'] / author_counts['len'].sum()
    
    # Storage for simulation results
    simulated_metrics = []
    
    # Run simulations
    for _ in tqdm.tqdm(range(n_simulations), desc="Running simulations"):
        # Create simulated comments preserving commenter activity levels
        # but randomizing author selection weighted by popularity
        sim_df = comment_df.with_columns(
            pl.col('authorUniqueId').shuffle().alias('simulated_author')
        )
        
        # Calculate same metrics for simulated data
        sim_metrics = calculate_observed_metrics(sim_df)
        simulated_metrics.append(sim_metrics)
    
    # Calculate probabilities
    results = {}
    
    # Analysis of author group sizes
    observed_groups = observed['author_groups']
    
    # For each observed group size and count
    for group_size in observed_groups['group_size'].unique():
        observed_count = observed_groups.filter(
            pl.col('group_size') == group_size
        )['group_size'].len()
        
        # Count how many simulations had at least this many groups of this size
        sim_counts = [
            sim['author_groups'].filter(
                pl.col('group_size') == group_size
            )['group_size'].len()
            for sim in simulated_metrics
        ]
        
        p_value = sum(count >= observed_count for count in sim_counts) / n_simulations
        
        results[f'group_size_{group_size}'] = {
            'observed_count': observed_count,
            'mean_simulated': np.mean(sim_counts),
            'std_simulated': np.std(sim_counts),
            'p_value': p_value
        }
    
    return results

def analyze_suspicious_patterns(comment_df: pl.DataFrame, min_group_size=5, min_interactors=10):
    """
    Analyze potentially suspicious comment patterns and compare to random expectation
    """
    # First get observed suspicious groups
    suspicious_groups = (
        comment_df.group_by('uid')
        .agg(pl.col('authorUniqueId').alias('authors'))
        .with_columns([
            pl.col('authors').list.len().alias('num_interactions'),
            pl.col('authors').list.unique().alias('unique_authors'),
            pl.col('authors').list.unique().list.len().alias('num_unique_authors')
        ])
        .filter(
            (pl.col('num_unique_authors') > min_group_size) & 
            (pl.col('num_interactions') > min_interactors)
        )
    )
    
    # Run Monte Carlo simulation
    simulation_results = run_monte_carlo_simulation(comment_df)
    
    # Analyze results
    report = {
        'observed_suspicious_groups': len(suspicious_groups),
        'simulation_results': simulation_results,
        'significant_patterns': [
            size for size, stats in simulation_results.items()
            if stats['p_value'] < 0.05
        ]
    }
    
    return report

def main():
    video_df = None
    video_dir_path = os.path.join('.', 'data', 'results', '2024_04_10', 'hours', '19')
    video_pbar = tqdm.tqdm(total=60*60, desc='Reading videos')
    for root, dirs, files in os.walk(video_dir_path):
        for file in files:
            if file == 'videos.parquet.zstd':
                video_pbar.update(1)
                result_path = os.path.join(root, file)
                batch_video_df = pl.read_parquet(result_path)
                if video_df is None:
                    video_df = batch_video_df
                else:
                    video_df = pl.concat([video_df, batch_video_df], how='diagonal_relaxed')

    comments_dir_path = os.path.join('.', 'data', 'comments')
    comment_df = None
    comment_pbar = tqdm.tqdm(total=len(list(os.listdir(comments_dir_path))), desc='Reading comments')
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

    video_df = video_df.select(['video_id', 'createTime', 'authorUniqueId', 'locationCreated', 'playCount']).rename({'createTime': 'video_create_time'})
    comment_df = comment_df.rename({'create_time': 'comment_create_time'})
    comment_df = comment_df.join(video_df, left_on='aweme_id', right_on='video_id', how='left')
    comment_df = comment_df.filter(pl.col('video_create_time').is_not_null() & pl.col('comment_create_time').is_not_null())
    comment_df = comment_df.with_columns(pl.from_epoch(pl.col('comment_create_time')).alias('comment_create_time_epoch'))
    comment_df = comment_df.with_columns(pl.from_epoch(pl.col('video_create_time')).alias('video_create_time_epoch'))
    comment_df = comment_df.with_columns((pl.col('comment_create_time_epoch') - pl.col('video_create_time_epoch')).alias('time_offset'))
    comment_df = comment_df.with_columns(pl.col('time_offset').dt.total_seconds().alias('time_offset_seconds'))

    results = analyze_suspicious_patterns(comment_df, min_group_size=5, min_interactors=10)
    
    print(f"Found {results['observed_suspicious_groups']} suspicious groups")
    print("\nStatistically significant patterns:")
    for pattern in results['significant_patterns']:
        stats = results['simulation_results'][pattern]
        print(f"\n{pattern}:")
        print(f"  Observed: {stats['observed_count']}")
        print(f"  Expected: {stats['mean_simulated']:.1f} ± {stats['std_simulated']:.1f}")
        print(f"  p-value: {stats['p_value']:.4f}")

    comment_interactions_df = comment_df.group_by('uid')\
        .agg(pl.col('authorUniqueId'))

    comment_interactions_df = comment_interactions_df.with_columns(
        pl.col('authorUniqueId').list.len().alias('num_interactions')
    )
    comment_interactions_df = comment_interactions_df.with_columns(
        pl.col('authorUniqueId').list.unique()
    )
    comment_interactions_df = comment_interactions_df.with_columns(
        pl.col('authorUniqueId').list.sort()
    )

    interaction_groups_df = comment_interactions_df.group_by('authorUniqueId')\
        .agg(pl.col('uid').len().alias('num_interactors'), pl.col('num_interactions').sum().alias('num_interactions'))

    suspicious_interactions_df = interaction_groups_df.filter((pl.col('authorUniqueId').list.len() > 5) & (pl.col('num_interactors') > 10))
    
    
    # find the same text between users


if __name__ == '__main__':
    main()