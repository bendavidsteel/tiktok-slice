import configparser
import os

from adjustText import adjust_text
import polars as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colorbar import ColorbarBase
import matplotlib.ticker as ticker
import geopandas as gpd
import pycountry
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import shapely.geometry as geometry
from shapely.ops import transform

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
    country_df = pl.read_csv(os.path.join('.', 'data', 'stats', '24hour', 'location_created_value_counts.csv'))
    
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

    coverage = {
        'IND': 0.97,
        'AUS': 0.92,
        'JPN': 0.96,
        'RUS': 0.91,
        'CAN': 1.00,
        'DEU': 1.00,
        'IDN': 1.00,
        'BRA': 1.00,
        'NGA': 1.00
    }

    coverage = [
        ('Oceania', 0.92, ['AUS', 'PNG', 'NZL', 'FJI', 'SLB', 'FSM', 'VUT', 'KIR', 'TON', 'MHL', 'PLW', 'NRU', 'TUV']),
        ('East Asia', 0.96, ['CHN', 'JPN', 'MNG', 'PRK', 'KOR', 'TWN', 'HKG', 'MAC']),
        ('South Asia', 0.97, ['AFG', 'BGD', 'BTN', 'IND', 'MDV', 'NPL', 'PAK', 'LKA']),
        ('Eastern Europe', 0.91, ['BLR', 'RUS', 'UKR', 'MDA', 'ROU'])
    ]

    def get_coverage(alpha_3):
        for region, cov, countries in coverage:
            if alpha_3 in countries:
                return cov
        return 1.0

    country_df = country_df.with_columns(pl.col('iso_alpha').map_elements(lambda x: get_coverage(x), pl.Float32).alias('coverage'))
    country_df = country_df.with_columns((pl.col('count') / pl.col('coverage')).alias('count'))

    # Take logarithm of both variables
    log_pop = np.log(country_df['2022'].to_numpy())
    log_count = np.log(country_df['count'].to_numpy())

    # Perform linear regression on log-transformed data
    result = stats.linregress(log_pop, log_count)

    # Calculate predicted values and residuals in log space
    predicted_log = result.intercept + result.slope * log_pop
    residuals = log_count - predicted_log

    # Get indices of top and bottom 3 residuals
    top_indices = np.argsort(residuals)[-3:]
    bottom_indices = np.argsort(residuals)[:3]
    outlier_indices = np.concatenate([top_indices, bottom_indices])

    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 4))
    scatter = ax.scatter(country_df['2022'], country_df['count'], s=10)

    # Generate points for the fitted line
    x_range = np.linspace(min(log_pop), max(log_pop), 100)
    y_fit = np.exp(result.intercept + result.slope * x_range)
    x_range = np.exp(x_range)

    # Plot the fitted line
    line = ax.plot(x_range, y_fit, color='red')

    # Create texts for adjustText
    fontsize = 12
    texts = []
    for i in outlier_indices:
        if country_df['Country Name'][int(i)] == 'United Arab Emirates':
            texts.append(ax.text(country_df['2022'][int(i)], 
                                country_df['count'][int(i)],
                                'UAE',
                                fontsize=fontsize))
        else:
            texts.append(ax.text(country_df['2022'][int(i)], 
                                country_df['count'][int(i)],
                                country_df['Country Name'][int(i)],
                                fontsize=fontsize))
        ax.scatter(country_df['2022'][int(i)], country_df['count'][int(i)], s=12, c='r')

    # Set scales to log
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Adjust text positions to avoid overlaps
    adjust_text(texts, objects=scatter, force_static=(0.4, 0.4))

    # Add R² and p-value
    ax.set_title(f"$R^2$: {result.rvalue**2:.2f}, p-value: {result.pvalue:.2f}")

    ax.set_xlabel("Population")
    ax.set_ylabel("No. of Videos from Country")

    # Adjust layout to prevent label clipping
    fig.tight_layout()

    fig.savefig('./figs/population_vs_count.png')


    # country_df = country_df.with_columns(pl.col('count').alias('log_count'))
    # count is from 24 minutes from one day. So multiply by 60 to get expected count for a day, then multiply by 365 to get expected count for a year
    country_df = country_df.with_columns((pl.col('count') * 60 / (pl.col('2022'))).log1p().alias('count_per_capita'))

    # Convert to pandas for compatibility with geopandas
    country_data = country_df.to_pandas()
    
    # Load world map data
    world = gpd.read_file('./data/countries_lakes/ne_110m_admin_0_countries_lakes.shp')
    world = world[world['CONTINENT'] != 'Antarctica']
    world = world.to_crs('+proj=wintri')
    
    # Merge with country data
    country_data = country_df.to_pandas()
    world = world.merge(country_data, how='left', left_on=['ISO_A3_EH'], right_on=['iso_alpha'])

    # Get the actual data range
    vmin = world['count_per_capita'].min()
    vmax = world['count_per_capita'].max()

    custom_cmap = plt.get_cmap('turbo')

    # Create figure and plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Plot the data with log scale for colors
    world.plot(
        column='count_per_capita',  
        ax=ax,
        legend=True,
        legend_kwds={
            'label': 'Daily Posts Per Capita',
            'orientation': 'horizontal',
            'shrink': 0.8,
            'fraction': 0.046,
            'pad': 0.04,
            'format': '%.1e'
        },
        missing_kwds={'color': 'lightgrey'},
        cmap=custom_cmap,
        norm=norm,  # Using LogNorm and ensuring vmin > 0
        alpha=1.0,
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
    plt.savefig('./figs/world_count_map.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.)
    plt.close()

if __name__ == '__main__':
    main()