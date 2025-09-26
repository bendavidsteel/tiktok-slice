import configparser
import os

from adjustText import adjust_text
import polars as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib.colors import Norm
from matplotlib.colorbar import ColorbarBase
import matplotlib.ticker as ticker
import geopandas as gpd
import pycountry
from tqdm import tqdm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import shapely.geometry as geometry
from shapely.ops import transform

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

def calculate_significance(success, total, default_p=0.0):
    """Calculate statistical significance using binomial test"""
    return total >= 50

def main():
    # Read config and data
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

    # result_paths = result_paths[:1000]
    video_df = None
    val_count_dfs = None
    for result_path in tqdm(result_paths):
        batch_df = pl.read_parquet(result_path, columns=['locationCreated', 'aigcLabelType'])
        if video_df is not None:
            video_df = pl.concat([video_df, batch_df], how='diagonal_relaxed')
        else:
            video_df = batch_df

    video_df = video_df.with_columns((pl.col('aigcLabelType').is_in(['1', '2']) & pl.col('aigcLabelType').is_not_null()).alias('is_aigc'))
    
    global_p = video_df['is_aigc'].sum() / video_df.shape[0]
    print(f"AI generated content ratio: {global_p:.4f}")
    
    country_df = video_df.group_by('locationCreated').agg([
        pl.col('is_aigc').count().alias('count'),
        pl.col('is_aigc').sum().alias('aigc_count'),
    ])
    
    country_df = country_df.filter(pl.col('locationCreated').is_not_null())
    
    def get_alpha_3(alpha_2):
        try:
            return pycountry.countries.get(alpha_2=alpha_2).alpha_3
        except Exception as e:
            extra_map = {
                'XK': 'UNK',
                'PAKISTAN ': 'PAK',
                'EGYPT ': 'EGY',
                'UK': 'GBR',
                'JAPAN': 'JPN',
                'IRAQ': 'IRQ',
                'USA': 'USA',
                'PAKISTAN': 'PAK',
                'IRAQ ': 'IRQ',
                'UK ': 'GBR',
                'AN': 'AND',
                'EN': 'EST'
            }
            if alpha_2 in extra_map:
                return extra_map[alpha_2]
            else:
                print(f"Couldn't find '{alpha_2}'")
                return None
    
    country_df = country_df.with_columns(pl.col('locationCreated').map_elements(get_alpha_3, pl.String).alias('iso_alpha'))
    country_df = country_df.group_by('iso_alpha').agg([
        pl.sum('aigc_count'), pl.sum('count')
    ])

    country_df = country_df.with_columns([
        (pl.col('aigc_count') / pl.col('count')).alias('aigc_ratio'),
        pl.Series(name='is_significant', values=[calculate_significance(r['aigc_count'], r['count'], default_p=0.0) for r in country_df.to_dicts()], dtype=pl.Boolean),
    ])


    # Convert to pandas for compatibility with geopandas
    country_data = country_df.to_pandas()
    
    # Load world map data
    world = gpd.read_file('./data/countries_lakes/ne_110m_admin_0_countries_lakes.shp')
    world = world[world['CONTINENT'] != 'Antarctica']
    world = world.to_crs('+proj=wintri')
    
    # Merge with country data
    country_data = country_df.to_pandas()
    world = world.merge(country_data, how='left', left_on=['ISO_A3_EH'], right_on=['iso_alpha'])

    world_significant = world[world['is_significant'] == True]

    # Get the actual data range
    vmin = world_significant['aigc_ratio'].min()
    vmax = world_significant['aigc_ratio'].max()

    custom_cmap = plt.get_cmap('turbo')

    # Create figure and plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    # Plot the data with log scale for colors
    world_significant.plot(
        column='aigc_ratio',  
        ax=ax,
        legend=True,
        legend_kwds={
            'label': 'AI generated content ratio',
            'orientation': 'horizontal',
            'shrink': 0.8,
            'fraction': 0.03,
            'pad': 0.04,
        },
        missing_kwds={'color': 'lightgrey'},
        cmap=custom_cmap,
        alpha=1.0,
    )

    world_not_significant = world[~world['is_significant'].fillna(False)]
    world_not_significant.plot(
        ax=ax,
        color='grey',
        hatch='///',
        edgecolor='white',
        linewidth=0,
        alpha=0.5
    )

    # Get the bounds of the geometries and set limits with a small buffer
    bounds = world.total_bounds
    buffer_x = (bounds[2] - bounds[0]) * 0.02  # 2% buffer
    buffer_y = (bounds[3] - bounds[1]) * 0.02
    ax.set_xlim(bounds[0] - buffer_x, bounds[2] + buffer_x)
    ax.set_ylim(bounds[1] - buffer_y, bounds[3] + buffer_y)

    # Customize appearance
    ax.axis('off')
    
    # Adjust layout
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    # Save figure
    plt.savefig('./figs/world_aigc_map.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.)
    plt.close()

if __name__ == '__main__':
    main()