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
    country_df = pl.read_csv(os.path.join('.', 'data', 'stats', 'all', 'location_created_value_counts.csv'))
    
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
        pl.sum('count').alias('count')
    ])
    pop_df = pl.read_csv(os.path.join('.', 'data', 'worlddata', 'pop_data.csv'), skip_rows=4)
    country_df = country_df.join(pop_df, left_on='iso_alpha', right_on='Country Code')
    # country_df = country_df.with_columns(pl.col('count').alias('log_count'))
    country_df = country_df.with_columns((pl.col('count') / (pl.col('2022') / 1000)).log1p().alias('count_per_capita'))

    # Convert to pandas for compatibility with geopandas
    country_data = country_df.to_pandas()
    
    # Load world map data
    world = gpd.read_file('./data/countries_lakes/ne_110m_admin_0_countries_lakes.shp')
    
    # Remove Antarctica
    world = world[world['CONTINENT'] != 'Antarctica']
    
    # Merge data with world map
    world = world.merge(country_data, how='left', left_on=['ISO_A3_EH'], right_on=['iso_alpha'])
    
    # Get the actual data range
    vmin = world['count_per_capita'].min()
    vmax = world['count_per_capita'].max()
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    
    # Create custom colormap similar to Turbo
    colors = ['#30123b', '#4777ef', '#1ac7c2', '#a6e622', '#fca50a', '#b41325']
    custom_cmap = LinearSegmentedColormap.from_list('custom_turbo', colors)
    
    
    # Plot significant countries
    world.plot(
        column='count_per_capita',
        ax=ax,
        legend=True,
        legend_kwds={
            'label': 'Log Count Per Thousand People',
            'orientation': 'vertical',
            'shrink': 0.8,
            'fraction': 0.046,
            'pad': 0.04
        },
        missing_kwds={'color': 'lightgrey'},
        cmap=custom_cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=1.0
    )
    
    # Customize the appearance
    ax.axis('off')
    # plt.title('Share of Child Videos by Country', pad=20, size=16)
    
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    # Save the figure
    figs_dir_path = os.path.join('.', 'figs')
    plt.savefig(os.path.join(figs_dir_path, 'world_count_map.png'),
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()

if __name__ == '__main__':
    main()