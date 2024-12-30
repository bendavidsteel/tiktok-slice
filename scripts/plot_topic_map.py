import os
import re

import matplotlib as mpl
import numpy as np
import polars as pl
from PIL import Image
import plotly.express as px
import pycountry

def convert_to_image(cols):
    return Image.frombytes(cols['Visual_Aspect_Mode'], tuple(cols['Visual_Aspect_Size']), cols['Visual_Aspect_Bytes'])

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', 'data', f'topic_model_videos_100')

    video_df = pl.read_parquet(os.path.join(data_dir_path, 'video_topics.parquet.gzip'))

    topic_info_df = pl.read_parquet(os.path.join(data_dir_path, 'topic_info.parquet.gzip'))

    pop_df = pl.read_csv(os.path.join('.', 'data', 'worlddata', 'pop_data.csv'), skip_rows=4)

    # find topic associated with children
    topic_id = None

    country_df = video_df.filter(pl.col('topic') == topic_id).group_by('locationCreated').count()
    country_df['count'] = country_df['id']
    country_df = country_df[country_df['locationCreated'].map(lambda c: c is not None)]
    def get_alpha_3(alpha_2):
        try:
            return pycountry.countries.get(alpha_2=alpha_2).alpha_3
        except:
            extra_map = {
                'XK': 'UNK'
            }
            if alpha_2 in extra_map:
                return extra_map[alpha_2]
            else:
                print(f"Couldn't find {alpha_2}")
                return None
    country_df['iso_alpha'] = country_df['locationCreated'].map(get_alpha_3)
    fig = px.choropleth(country_df, locations="iso_alpha",
                        color="count", # lifeExp is a column of gapminder
                        # hover_name="country", # column to add to hover information
                        color_continuous_scale=px.colors.sequential.Plasma)

    country_df['count_per_capita'] = country_df['count'] / pop_df['2022']
    fig = px.choropleth(country_df, locations="iso_alpha",
                        color="count_per_capita", # lifeExp is a column of gapminder
                        # hover_name="country", # column to add to hover information
                        color_continuous_scale=px.colors.sequential.Plasma)
    fig.save(os.path.join(figs_dir_path, 'world_topic_map.html'))

if __name__ == '__main__':
    main()