import os
import re

import datamapplot
import matplotlib as mpl
import numpy as np
import pandas as pd
from PIL import Image

def convert_to_image(cols):
    return Image.frombytes(cols['Visual_Aspect_Mode'], tuple(cols['Visual_Aspect_Size']), cols['Visual_Aspect_Bytes'])

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', 'data')

    video_df = pd.read_parquet(os.path.join(data_dir_path, 'video_topics.parquet.gzip'))

    embeddings_2d = np.load(os.path.join(data_dir_path, 'reduced_embeddings.npy'))

    topic_info_df = pd.read_parquet(os.path.join(data_dir_path, 'topic_info.parquet.gzip'))
    topic_info_df['Visual_Aspect'] = topic_info_df[['Visual_Aspect_Mode', 'Visual_Aspect_Size', 'Visual_Aspect_Bytes']].apply(convert_to_image, axis=1)

    # Prepare text and names
    topic_name_mapping = {row['Topic']: f"Topic-{row['Topic']}" for _, row in topic_info_df[['Topic', 'Name']].iterrows()}
    # if isinstance(custom_labels, str):
    #     names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in unique_topics]
    #     names = [" ".join([label[0] for label in labels[:4]]) for labels in names]
    #     names = [label if len(label) < 30 else label[:27] + "..." for label in names]
    # elif topic_model.custom_labels_ is not None and custom_labels:
    #     names = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in unique_topics]
    # else:
    #     names = [f"Topic-{topic}: " + " ".join([word for word, value in topic_model.get_topic(topic)][:3]) for topic in unique_topics]

    # topic_name_mapping = {topic_num: topic_name for topic_num, topic_name in zip(unique_topics, names)}
    topic_name_mapping[-1] = "Unlabelled"

    # If a set of topics is chosen, set everything else to "Unlabelled"
    chosen_topics = list(range(1, 3))
    if chosen_topics:
        selected_topics = set(chosen_topics)
        for topic_num in topic_name_mapping:
            if topic_num not in selected_topics:
                topic_name_mapping[topic_num] = "Unlabelled"

    # Map in topic names and plot
    named_topic_per_doc = video_df['topic'].map(topic_name_mapping).values

    images = {f"Topic-{topic_num}": topic_info_df[topic_info_df['Topic'] == topic_num]['Visual_Aspect'].values[0] for topic_num in topic_name_mapping.keys()}
    images = {k: np.asarray(v) for (k, v) in images.items()}

    width, height = 1200, 1200

    figure, axes = datamapplot.create_plot(
        embeddings_2d,
        labels=named_topic_per_doc,
        images=images,
        figsize=(width/100, height/100),
        dpi=100,
    )
    # child_artists = axes.get_children()
    # for artist in child_artists:
    #     if isinstance(artist, mpl.text.Text):
    #         # replace the text with representative images
    #         text = artist.get_text()
    #         if not text:
    #             continue
    #         topic_num = int(re.search(r'\d+', text).group())
    #         # artist.set_text("")
    #         # artist.set_visible(False)
    #         # artist.remove()
    #         coords = artist.get_position()
    #         image = topic_info_df[topic_info_df['Topic'] == topic_num]['Visual_Aspect'].values[0]
    #         image_width, image_height = 0.1, 0.1
    #         axes.imshow(
    #             np.asarray(image),
    #             extent=[coords[0], coords[0] + image_width, coords[1], coords[1] + image_height],
    #             transform=axes.transAxes,
    #         )


    figure.savefig(os.path.join(this_dir_path, '..', 'figs', 'datamapplot.png'), bbox_inches='tight')

if __name__ == '__main__':
    main()