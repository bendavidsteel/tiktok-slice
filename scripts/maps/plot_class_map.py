import configparser
import os
import polars as pl
import matplotlib.pyplot as plt
import geopandas as gpd
import pycountry
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

def calculate_significance(success, total, confidence=0.95):
    """Calculate statistical significance using binomial test"""
    if total == 0:
        return False
    
    # Perform binomial test against null hypothesis of p=0.5
    p_value = stats.binomtest(success, n=total, p=0.5).pvalue
    
    # Return whether p-value is significant at 0.05 level
    return p_value < 0.05

def main():
    # Read config and data
    config = configparser.ConfigParser()
    config.read('./config/config.ini')
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    video_df = pl.read_parquet(os.path.join('.', 'data', 'video_child_prob.parquet.gzip'))
    
    # Process video data
    threshold = 0.6
    video_df = video_df.with_columns((pl.col('child_prob') > threshold).cast(pl.Int32).alias('child_present'))
    country_df = video_df.group_by('locationCreated').agg([
        pl.col('child_present').sum().alias('num_child_videos'),
        pl.count().alias('num_videos')
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
            }
            if alpha_2 in extra_map:
                return extra_map[alpha_2]
            else:
                print(f"Couldn't find {alpha_2}")
                return None
    
    country_df = country_df.with_columns(pl.col('locationCreated').map_elements(get_alpha_3, pl.String).alias('iso_alpha'))
    country_df = country_df.group_by('iso_alpha').agg([
        pl.sum('num_child_videos').alias('num_child_videos'),
        pl.sum('num_videos').alias('num_videos')
    ])
    country_df = country_df.with_columns((pl.col('num_child_videos') / pl.col('num_videos')).alias('share_child_videos'))
    
    country_df = country_df.with_columns(
        pl.struct([pl.col('num_child_videos'), pl.col('num_videos')]).map_elements(lambda r: calculate_significance(r['num_child_videos'], r['num_videos']), return_dtype=pl.Boolean).alias('is_significant')
    )

    # load demographic data
    demographic_df = pl.read_csv('./data/worlddata/population-by-age/population-by-age-group-with-projections.csv')
    demographic_df = demographic_df.filter(pl.col('Year') == 2023)\
        .rename({
            'Population - Sex: all - Age: 0-14 - Variant: estimates': 'population_under_14',
            'Population - Sex: all - Age: all - Variant: estimates': 'population_all'
        })\
        .select(['Code', 'population_all', 'population_under_14'])
    demographic_df = demographic_df.with_columns((pl.col('population_under_14') / pl.col('population_all')).alias('share_under_14'))
    
    country_df = country_df.join(demographic_df, left_on='iso_alpha', right_on='Code', how='left')
    country_df = country_df.with_columns((pl.col('share_child_videos') / pl.col('share_under_14')).alias('child_video_ratio'))

    # Convert to pandas for compatibility with geopandas
    country_data = country_df.to_pandas()

    # Load world map data
    world = gpd.read_file('./data/countries_lakes/ne_110m_admin_0_countries_lakes.shp')
    
    # Remove Antarctica
    world = world[world['CONTINENT'] != 'Antarctica']
    
    # Merge data with world map
    world = world.merge(country_data, how='left', left_on=['ISO_A3_EH'], right_on=['iso_alpha'])
    
    # Get the actual data range
    vmin = world['child_video_ratio'].min()
    vmax = world['child_video_ratio'].max()
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    # Create custom colormap similar to Turbo
    colors = ['#30123b', '#4777ef', '#1ac7c2', '#a6e622', '#fca50a', '#b41325']
    custom_cmap = LinearSegmentedColormap.from_list('custom_turbo', colors)

    # Plot all countries first with full color
    world.plot(
        column='child_video_ratio',
        ax=ax,
        legend=True,
        legend_kwds={
            'label': 'Share of videos containing children, adjusted by share of population under 14.\n(Solid: statistically significant,\nHatched: not significant)',
            'orientation': 'vertical',
            'shrink': 0.8,
            'fraction': 0.046,
            'pad': 0.04
        },
        missing_kwds={'color': 'lightgrey'},
        cmap=custom_cmap,
        vmin=vmin,
        vmax=vmax
    )

    # Add hatching to non-significant countries
    world_not_significant = world[~world['is_significant'].fillna(False)]
    world_not_significant.plot(
        ax=ax,
        color='none',
        hatch='///',
        edgecolor='white',
        linewidth=0,
        alpha=0.5
    )

    # Customize the appearance
    ax.axis('off')
    
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    # Save the figure
    figs_dir_path = os.path.join('.', 'figs')
    plt.savefig(os.path.join(figs_dir_path, 'world_child_map.png'),
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()

if __name__ == '__main__':
    main()