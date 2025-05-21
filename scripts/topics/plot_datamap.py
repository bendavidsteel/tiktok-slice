import os

import datamapplot
import numpy as np
import polars as pl


def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data', f'topic_model_videos')

    video_df = pl.read_parquet(os.path.join(data_dir_path, 'video_topics.parquet.gzip'))

    embedding_path = os.path.join(data_dir_path, '2d_embeddings.npy')
    if os.path.exists(embedding_path):
        embeddings_2d = np.load(embedding_path)
    elif os.path.exists(os.path.join(data_dir_path, 'reduced_embeddings.npy')):
        embeddings_2d = np.load(os.path.join(data_dir_path, 'reduced_embeddings.npy'))

    desc_path = os.path.join(data_dir_path, 'topic_desc.parquet.gzip')
    topic_info_df = pl.read_parquet(desc_path)

    top_n_topics = 50
    if top_n_topics:
        topic_info_df = topic_info_df.sort('Count', descending=True).head(top_n_topics)

    # Prepare text and names
    topic_name_mapping = {row['Topic']: row['Desc'] for row in topic_info_df[['Topic', 'Desc']].to_dicts()}
    topic_name_mapping[-1] = "Unlabelled"

    if top_n_topics:
        for topic_num in topic_info_df['Topic'].to_list():
            if topic_num not in topic_name_mapping:
                topic_name_mapping[topic_num] = "Unlabelled"

    # If a set of topics is chosen, set everything else to "Unlabelled"
    chosen_topics = None
    if chosen_topics:
        selected_topics = set(chosen_topics)
        for topic_num in topic_name_mapping:
            if topic_num not in selected_topics:
                topic_name_mapping[topic_num] = "Unlabelled"

    # Map in topic names and plot
    named_topic_per_doc = video_df['topic'].replace_strict(topic_name_mapping, default='Unlabelled').to_list()

    # TODO dot size determined by view count

    fig, axes = datamapplot.create_plot(
        embeddings_2d.astype(np.float32),
        labels=named_topic_per_doc,
        label_over_points=True,
        dynamic_label_size=True,
        use_medoids=True,
        # dynamic_label_size_scaling_factor=0.75,
        min_font_size=8.0,
        max_font_size=16.0,
        dpi=400,
        # marker_size_array=video_df['playCount'].to_numpy(),
        # force_matplotlib=True
        verbose=True
    )

    axes.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    figs_dir_path = os.path.join(this_dir_path, '..', '..', 'figs')
    os.makedirs(figs_dir_path, exist_ok=True)
    fig.savefig(os.path.join(figs_dir_path, 'datamapplot.png'), bbox_inches='tight')

if __name__ == '__main__':
    main()