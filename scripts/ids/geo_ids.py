import configparser
import os
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tqdm
import geopandas as gpd
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

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

def process_batch(result_path):
    video_df = pl.read_parquet(result_path, columns=['id', 'locationCreated'])
    video_df = video_df.with_columns(
        pl.col('id').cast(pl.UInt64)
        .map_elements(lambda i: format(i, '064b'), pl.String)
        .str.slice(56, 64)
        .map_elements(lambda s: int(s, 2), pl.UInt64)
        .alias('machine_id')
    ).select(['machine_id', 'locationCreated'])
    return video_df

def create_choropleth(video_df, output_dir_path):
    # Convert polars DataFrame to pandas
    pd_df = video_df.to_pandas()
    
    # Load world shapefile
    world = gpd.read_file('./data/countries_lakes/ne_110m_admin_0_countries_lakes.shp')
    world = world[world['CONTINENT'] != 'Antarctica']
    world = world.to_crs('+proj=wintri')
    
    # Rename columns for easier merging
    world = world.rename(columns={'ISO_A2': 'locationCreated'})
    
    # Group by machine_id and country code, count occurrences
    country_counts = pd_df.groupby(['machine_id', 'locationCreated']).size().reset_index(name='count')
    
    # Create a list to store plots for each machine_id
    for machine_id in pd_df['machine_id'].unique():
        # Filter data for this machine_id
        machine_data = country_counts[country_counts['machine_id'] == machine_id]
        
        # Merge with world data
        merged = world.merge(machine_data, on='locationCreated', how='left')
        merged['count'] = merged['count'].fillna(0)
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # Define a custom colormap (white for 0, blue for values)
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#f7f7f7', '#2171b5'])
        
        # Plot the choropleth
        merged.plot(column='count', 
                    ax=ax, 
                    legend=True,
                    cmap=cmap,
                    legend_kwds={'label': f"Number of Occurrences for Machine ID {machine_id}",
                                'orientation': "horizontal"})
        
        ax.set_title(f'Distribution of Countries Using Machine ID {machine_id}')
        ax.set_axis_off()
        
        # Save the figure
        plt.tight_layout()
        file_path = os.path.join(output_dir_path, f'machine_id_{machine_id}_distribution.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a combined visualization showing all machine IDs
    # We'll create a map where each country is colored by the dominant machine_id
    dominant_machine = pd_df.groupby('locationCreated')['machine_id'].agg(
        lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else None
    ).reset_index()
    
    merged_all = world.merge(dominant_machine, on='locationCreated', how='left')
    
    # Create a qualitative colormap for the 17 machine IDs
    # Using tab20 colormap which has 20 distinct colors
    unique_machines = pd_df['machine_id'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_machines)))
    machine_colors = {mid: colors[i] for i, mid in enumerate(unique_machines)}
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # Plot countries with no data in light gray
    world.plot(ax=ax, color='#f0f0f0')
    
    # Plot each machine_id with its own color
    for machine_id in unique_machines:
        countries = merged_all[merged_all['machine_id'] == machine_id]
        if not countries.empty:
            countries.plot(ax=ax, color=machine_colors[machine_id])
    
    # Create a legend
    patches = [mpatches.Patch(color=machine_colors[mid], label=f'Machine ID {mid}') 
               for mid in unique_machines if not merged_all[merged_all['machine_id'] == mid].empty]
    ax.legend(handles=patches, loc='lower left', title='Dominant Machine ID')
    
    ax.set_title('Dominant Machine ID by Country')
    ax.set_axis_off()
    
    # Save the combined figure
    plt.tight_layout()
    file_path = os.path.join(output_dir_path, 'all_machine_ids_distribution.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

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
    
    # result_paths = result_paths[:5]
    video_df = None
    for result_path in tqdm.tqdm(result_paths):
        batch_df = process_batch(result_path)
        if video_df is None:
            video_df = batch_df
        else:
            video_df = pl.concat([video_df, batch_df])
    
    # Create directory for maps if it doesn't exist
    maps_dir = os.path.join(output_dir_path, 'figs', 'machine_id_maps')
    os.makedirs(maps_dir, exist_ok=True)
    
    # Create choropleth maps
    create_choropleth(video_df, maps_dir)
    
    print(f"Choropleth maps created and saved to {maps_dir}")

if __name__ == "__main__":
    main()