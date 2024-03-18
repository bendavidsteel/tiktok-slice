import collections
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def do_analysis(video_ids, fig_dir_path):
    video_bits = [format(int(id), '064b') for id in video_ids]

    if False:
        timestamps = [int(b[:32], 2) for b in video_bits]
        time_in_day = [timestamp % (60 * 60 * 24) for timestamp in timestamps]
        hour_in_day = [time // (60 * 60) for time in time_in_day]
        time_in_hour = [timestamp % (60 * 60) for timestamp in timestamps]
        minute_in_hour = [time // 60 for time in time_in_hour]
        second_in_minute = [timestamp % 60 for timestamp in timestamps]

        fig, ax = plt.subplots(ncols=3)
        ax[0].hist(hour_in_day, bins=24)
        ax[0].set_title('Hour in day')
        ax[0].set_xlabel('Hour')
        ax[0].set_ylabel('Count')
        ax[1].hist(minute_in_hour, bins=60)
        ax[1].set_title('Minute in hour')
        ax[1].set_xlabel('Minute')
        ax[1].set_ylabel('Count')
        ax[2].hist(second_in_minute, bins=60)
        ax[2].set_title('Second in minute')
        ax[2].set_xlabel('Second')
        ax[2].set_ylabel('Count')
        fig.savefig(os.path.join(fig_dir_path, 'time_distribution.png'))

    video_other_bits = [b[32:] for b in video_bits]
    if True:
        for section_size in [1, 2, 4, 8]:
            bit_sections = [[] for _ in range(32 // section_size)]
            for other_bits in video_other_bits:
                sections = [int(other_bits[i * section_size: (i + 1) * section_size], 2) for i in range(32 // section_size)]
                for i in range(len(bit_sections)):
                    bit_sections[i].append(sections[i])

            # plot histogram of each section
            fig, ax = plt.subplots(ncols=len(bit_sections), figsize=(len(bit_sections) * 4, 5))
            for i, section in enumerate(bit_sections):
                num_unique = len(set(section))
                counter = collections.Counter(section)
                bins = np.array([i for i in range(2 ** section_size)])
                counts = np.array([counter[i] for i in range(2 ** section_size)])
                if section_size <= 4:
                    ax[i].bar(bins, counts)
                else:
                    ax[i].plot(bins, counts)
                    ax[i].set_ylim([0, max(counts) * 1.1])
                ax[i].set_title(f'Section {i}, {num_unique} unique values')
                ax[i].set_xlabel('Value')
                ax[i].set_ylabel('Count')
            fig.savefig(os.path.join(fig_dir_path, f'section_{section_size}_distribution.png'))

    if False:
        low_std_segments = []
        min_segment_length = 3
        max_segment_length = 12
        for i in range(32 - min_segment_length):
            for j in range(i + min_segment_length, min(i + max_segment_length, 32) + 1):
                segment_bits = [int(b[i:j], 2) for b in video_other_bits]
                counter = collections.Counter(segment_bits)
                counts = [counter[k] for k in range(2 ** (j - i))]
                std = np.std(counts)
                low_std_segments.append(((i, j), std))

        # plot low std segments
        low_std_segments = sorted(low_std_segments, key=lambda x: x[1])
        for (i, j), std in low_std_segments[:5]:
            segment_bits = [int(b[i:j], 2) for b in video_other_bits]
            counter = collections.Counter(segment_bits)
            bins = np.array([v for v in range(2 ** (j - i))])
            counts = np.array([counter[v] for v in range(2 ** (j - i))])
            fig, ax = plt.subplots()
            ax.plot(bins, counts)
            ax.set_title(f'Segment {i}:{j}, std {std}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            fig.savefig(os.path.join(fig_dir_path, f'low_std_segment_{i}_{j}_distribution.png'))

    if False:
        # use logistic regression to see which bits are important to each other
        bit_coefs = []
        for i in range(32):
            X = np.array([[int(bits[j], 2) for j in range(len(bits)) if j != i] for bits in video_other_bits])
            y = np.array([int(b[i], 2) for b in video_other_bits])
            if len(np.unique(y)) == 1:
                bit_coefs.append(np.zeros(32))
                continue
            model = LogisticRegression()
            model.fit(X, y)
            bit_coefs.append(np.concatenate([model.coef_[0,:i], np.array([0]), model.coef_[0,i:]]))

        # plot attention map
        bit_coefs = np.array(bit_coefs)
        fig, ax = plt.subplots()
        im = ax.imshow(bit_coefs, cmap='PiYG', vmin=-10, vmax=10)
        fig.colorbar(im)
        ax.set_title('Bit attention map')
        ax.set_xlabel('Bit')
        ax.set_ylabel('Bit')
        fig.savefig(os.path.join(fig_dir_path, 'bit_attention_map.png'))

    if True:

        found_intervals = [(0,9), (10,13), (14,17), (18,29), (30,31)]
        bit_sections = []
        for interval in found_intervals:
            bit_sections.append([int(b[interval[0]:interval[1]+1], 2) for b in video_other_bits])

        # plot histogram of each section
        fig, ax = plt.subplots(ncols=len(bit_sections), figsize=(len(bit_sections) * 4, 5))
        for i, (interval, section) in enumerate(zip(found_intervals, bit_sections)):
            num_unique = len(set(section))
            counter = collections.Counter(section)
            bins = np.array([j for j in range(2 ** (interval[1] + 1 - interval[0]))])
            counts = np.array([counter[j] for j in range(2 ** (interval[1] + 1 - interval[0]))])
            if (interval[1] + 1 - interval[0]) <= 4:
                ax[i].bar(bins, counts)
            else:
                ax[i].plot(bins, counts)
                ax[i].set_ylim([0, max(counts) * 1.1])
            ax[i].set_title(f'Section {interval[0]}-{interval[1]}, {num_unique} unique values')
            ax[i].set_xlabel('Value')
            ax[i].set_ylabel('Count')
        fig.savefig(os.path.join(fig_dir_path, 'found_segments_distribution.png'))

        # plot percentage of potential IDs hit vs number of combinations
        probs = [1 - (10 ** i) for i in range(-1, -7, -1)]
        all_num_combos = []
        for prob in probs:
            num_combos = 1
            for i, (interval, section) in enumerate(zip(found_intervals, bit_sections)):
                num_unique = len(set(section))
                counter = collections.Counter(section)
                bins = np.array([j for j in range(2 ** (interval[1] + 1 - interval[0]))])
                counts = np.array([counter[j] for j in range(2 ** (interval[1] + 1 - interval[0]))])
                bin_probs = counts / np.sum(counts)
                bin_probs = np.sort(bin_probs)[::-1]
                cum_probs = np.cumsum(bin_probs)
                # get all bins that cover prob of the data
                num_bins = np.argmax(cum_probs > prob) + 1
                num_combos *= num_bins
            all_num_combos.append(num_combos)

        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(probs, all_num_combos)
        ax.set_title('Number of combinations vs percentage of potential IDs hit')
        ax.set_xlabel('Percentage of potential IDs hit')
        ax.set_ylabel('Number of combinations')
        fig.savefig(os.path.join(fig_dir_path, 'potential_ids_vs_combinations.png'))

        prob = 0.999
        combos = {}
        for i, (interval, section) in enumerate(zip(found_intervals, bit_sections)):
            num_unique = len(set(section))
            counter = collections.Counter(section)
            bins = np.array([j for j in range(2 ** (interval[1] + 1 - interval[0]))])
            counts = np.array([counter[j] for j in range(2 ** (interval[1] + 1 - interval[0]))])
            bin_probs = counts / np.sum(counts)
            prob_order = np.argsort(bin_probs)[::-1]
            bins_ordered = bins[prob_order]
            bin_probs = bin_probs[prob_order]
            cum_probs = np.cumsum(bin_probs)
            # get all bins that cover prob of the data
            num_bins = np.argmax(cum_probs > prob) + 1
            combos[str(interval)] = bins_ordered[:num_bins].tolist()

        with open(os.path.join(fig_dir_path, f'{str(prob).replace('.', '_')}_found_segments_combinations.json'), 'w') as file:
            json.dump(combos, file, indent=4)


def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")

    countries = ['brazil', 'canada', 'germany', 'indonesia', 'nigeria']

    all_country_video_ids = []
    for country in countries:
        country_video_ids = []
        for filename in os.listdir(os.path.join(data_dir_path, country, 'videos')):
            if filename.endswith(".json"):
                with open(os.path.join(data_dir_path, country, 'videos', filename), 'r') as file:
                    videos = json.load(file)
                country_video_ids.extend([v['id'] for v in videos])
            elif 'parquet' in filename:
                video_df = pd.read_parquet(os.path.join(data_dir_path, country, 'videos', filename))
                if 'id' in video_df.columns:
                    file_ids = video_df['id'].to_list()
                elif 'video_id' in video_df.columns:
                    file_ids = video_df['video_id'].to_list()
                country_video_ids.extend(file_ids)
        country_video_ids = list(set(country_video_ids))
        country_fig_dir_path = os.path.join(this_dir_path, "..", "figs", country)
        if not os.path.exists(country_fig_dir_path):
            os.makedirs(country_fig_dir_path)
        do_analysis(country_video_ids, country_fig_dir_path)
        all_country_video_ids.extend(country_video_ids)

    canadian_comment_ids = []
    for filename in os.listdir(os.path.join(data_dir_path, 'canada', 'comments')):
        if filename.endswith(".parquet.gzip"):
            comment_df = pd.read_parquet(os.path.join(data_dir_path, 'canada', 'comments', filename))
            canadian_comment_ids.extend(comment_df['comment_id'].to_list())
    canadian_comment_ids = list(set(canadian_comment_ids))
    canadian_fig_comment_dir_path = os.path.join(this_dir_path, "..", "figs", "canada_comments")
    if not os.path.exists(canadian_fig_comment_dir_path):
        os.makedirs(canadian_fig_comment_dir_path)
    do_analysis(canadian_comment_ids, canadian_fig_comment_dir_path)

    canadian_user_ids = []
    for filename in os.listdir(os.path.join(data_dir_path, 'canada', 'comments')):
        if filename.endswith(".parquet.gzip"):
            comment_df = pd.read_parquet(os.path.join(data_dir_path, 'canada', 'comments', filename))
            canadian_user_ids.extend(comment_df['author_id'].to_list())
    canadian_user_ids = list(set(canadian_user_ids))
    canadian_fig_user_dir_path = os.path.join(this_dir_path, "..", "figs", "canada_users")
    if not os.path.exists(canadian_fig_user_dir_path):
        os.makedirs(canadian_fig_user_dir_path)
    do_analysis(canadian_user_ids, canadian_fig_user_dir_path)

    all_fig_dir_path = os.path.join(this_dir_path, "..", "figs", "all_videos")
    do_analysis(all_country_video_ids, all_fig_dir_path)

    


if __name__ == "__main__":
    main()
