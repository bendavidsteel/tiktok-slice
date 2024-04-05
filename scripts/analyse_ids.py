import collections
import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
import tqdm

from copia.plot import accumulation_curve
from copia.stats import species_accumulation

def plot_found_segments(video_other_bits, fig_dir_path):
    plot_found_segments_for_intervals(video_other_bits, fig_dir_path, [(0,9), (10,13), (14,17), (18,25), (26,31)], 'found_segments')
    plot_found_segments_for_intervals(video_other_bits, fig_dir_path, [(10,31)], 'two_segments')

def plot_found_segments_for_intervals(video_other_bits, fig_dir_path, found_intervals, segment_name):
    bit_sections = []
    for interval in found_intervals:
        bit_sections.append([int(b[interval[0]:interval[1]+1], 2) for b in video_other_bits])

    # plot histogram of each section
    fig, axes = plt.subplots(ncols=len(bit_sections), figsize=(len(bit_sections) * 4, 5))
    for i, (interval, section) in enumerate(zip(found_intervals, bit_sections)):
        if len(bit_sections) == 1:
            ax = axes
        else:
            ax = axes[i]
        
        section = np.array(section)
        num_unique = np.unique(section).shape[0]
        bins = np.arange(2 ** (interval[1] + 1 - interval[0]))
        counts = np.bincount(section, minlength=2 ** (interval[1] + 1 - interval[0]))

        entropy = sps.entropy(counts / np.sum(counts))
        # calculate confidence interval on number of unique values
        def bootstrap_fn(data):
            return np.unique(data).shape[0]
        bootstrap = sps.bootstrap((section,), bootstrap_fn, method='percentile', alternative='less')

        if (interval[1] + 1 - interval[0]) <= 4:
            ax.bar(bins, counts)
        else:
            ax.plot(bins, counts)
            ax.set_ylim([0, max(counts) * 1.1])
        ax.set_title(f'{interval[0]}-{interval[1]}, {num_unique} unique, entropy: {entropy:.2f}, CI: {bootstrap.confidence_interval.high}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
    fig.savefig(os.path.join(fig_dir_path, f'{segment_name}_distribution.png'))
    plt.close(fig)

def get_most_probable_bits(video_other_bits, fig_dir_path):
    get_most_probable_bits_for_intervals(video_other_bits, fig_dir_path, [(10,31)], 'two_segments')
    get_most_probable_bits_for_intervals(video_other_bits, fig_dir_path, [(0,9), (10,13), (14,17), (18,29), (30,31)], 'found_segments')

def get_most_probable_bits_for_intervals(video_other_bits, fig_dir_path, found_intervals, segment_name):
    bit_sections = []
    for interval in found_intervals:
        bit_sections.append([int(b[interval[0]:interval[1]+1], 2) for b in video_other_bits])

    # plot percentage of potential IDs hit vs number of combinations
    probs = [1 - (10 ** i) for i in range(-1, -7, -1)]
    all_num_combos = []
    for prob in probs:
        num_combos = 1
        combos = {}
        for interval, section in zip(found_intervals, bit_sections):
            bins = np.arange(2 ** (interval[1] + 1 - interval[0]))
            counts = np.bincount(section, minlength=2 ** (interval[1] + 1 - interval[0]))
            bin_probs = counts / np.sum(counts)

            # get all bins that cover prob of the data
            bin_probs = np.sort(bin_probs)[::-1]
            cum_probs = np.cumsum(bin_probs)
            # get all bins that cover prob of the data
            num_bins = np.argmax(cum_probs > prob) + 1
            num_combos *= num_bins

            # get all values that fit into the prob
            prob_order = np.argsort(bin_probs)[::-1]
            bins_ordered = bins[prob_order]
            bin_probs = bin_probs[prob_order]
            cum_probs = np.cumsum(bin_probs)
            # get all bins that cover prob of the data
            num_bins = np.argmax(cum_probs > prob) + 1
            combos[str(interval)] = bins_ordered[:num_bins].tolist()

        all_num_combos.append(num_combos)
        with open(os.path.join(fig_dir_path, f"{str(prob).replace('.', '_')}_{segment_name}_combinations.json"), 'w') as file:
            json.dump(combos, file, indent=4)

    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(probs, all_num_combos)
    ax.set_title('Number of combinations vs percentage of potential IDs hit')
    ax.set_xlabel('Percentage of potential IDs hit')
    ax.set_ylabel('Number of combinations')
    fig.savefig(os.path.join(fig_dir_path, 'potential_ids_vs_combinations.png'))
    plt.close(fig)

    combos = {}
    for interval, section in zip(found_intervals, bit_sections):
        bins = np.arange(2 ** (interval[1] + 1 - interval[0]))
        counts = np.bincount(section, minlength=2 ** (interval[1] + 1 - interval[0]))
        bins_at_least_1 = bins[counts > 0]
        combos[str(interval)] = bins_at_least_1.tolist()

    with open(os.path.join(fig_dir_path, f'all_{segment_name}_combinations.json'), 'w') as file:
        json.dump(combos, file, indent=4)

def get_rarefaction_extrapolation(video_other_bits, fig_dir_path):
    interval = (10,31)
    section = [int(b[interval[0]:interval[1]+1], 2) for b in video_other_bits]
    counts = np.bincount(section)
    accumulation = species_accumulation(counts, max_steps=len(section) * 2)
    fig, ax = plt.subplots()
    accumulation_curve(counts, accumulation, ax=ax, xlabel='Number of IDs', ylabel='Number of unique Bit Patterns')
    fig.savefig(os.path.join(fig_dir_path, 'rarefaction_extrapolation.png'))

def do_analysis(video_ids, fig_dir_path):
    video_bits = [format(int(id), '064b') for id in video_ids]

    if True:
        timestamps = [int(b[:32], 2) for b in video_bits]
        time_in_day = [timestamp % (60 * 60 * 24) for timestamp in timestamps]
        hour_in_day = [time // (60 * 60) for time in time_in_day]
        time_in_hour = [timestamp % (60 * 60) for timestamp in timestamps]
        minute_in_hour = [time // 60 for time in time_in_hour]
        second_in_minute = [timestamp % 60 for timestamp in timestamps]
        milliseconds = [int(b[32:41], 2) for b in video_bits]

        fig, ax = plt.subplots(ncols=4)
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
        ax[3].hist(milliseconds, bins=1000)
        ax[3].set_title('Milliseconds')
        ax[3].set_xlabel('Millisecond')
        ax[3].set_ylabel('Count')
        fig.savefig(os.path.join(fig_dir_path, 'time_distribution.png'))

    video_other_bits = [b[32:] for b in video_bits]
    if False:
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
        # TODO change to low entropy segments
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

    if True:
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
        plot_found_segments(video_other_bits, fig_dir_path)
        

    if True:
        get_most_probable_bits(video_other_bits, fig_dir_path)

    if True:
        get_rarefaction_extrapolation(video_other_bits, fig_dir_path)


def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")

    countries = ['brazil', 'canada', 'germany', 'indonesia', 'nigeria']

    print("All countries")
    all_videos = []
    for country in tqdm.tqdm(countries):
        country_videos = []
        for filename in os.listdir(os.path.join(data_dir_path, country, 'videos')):
            if filename.endswith(".json"):
                with open(os.path.join(data_dir_path, country, 'videos', filename), 'r') as file:
                    videos = json.load(file)
                if 'author' in videos:
                    videos = pd.DataFrame(videos).to_dict('records')
                country_videos.extend([{'id': v['id'], 'createtime': datetime.datetime.fromtimestamp(v['createTime'])} for v in videos])
            elif 'parquet' in filename:
                video_df = pd.read_parquet(os.path.join(data_dir_path, country, 'videos', filename))
                if 'id' in video_df.columns:
                    video_df = video_df.rename(columns={'createTime': 'createtime'})
                    file_videos = video_df[['id', 'createtime']].to_dict('records')
                    file_videos = [{'id': v['id'], 'createtime': datetime.datetime.fromtimestamp(v['createtime'])} for v in file_videos]
                elif 'video_id' in video_df.columns:
                    video_df = video_df.rename(columns={'video_id': 'id'})
                    file_videos = video_df[['id', 'createtime']].to_dict('records')
                    file_videos = [{'id': v['id'], 'createtime': v['createtime'].to_pydatetime()} for v in file_videos]
                
                country_videos.extend(file_videos)
        country_video_df = pd.DataFrame(country_videos)
        country_video_df = country_video_df.drop_duplicates(subset=['id'])
        country_video_ids = country_video_df['id'].to_list()
        country_fig_dir_path = os.path.join(this_dir_path, "..", "figs", country)
        if not os.path.exists(country_fig_dir_path):
            os.makedirs(country_fig_dir_path)
        do_analysis(country_video_ids, country_fig_dir_path)
        country_videos = country_video_df.to_dict('records')
        all_videos.extend(country_videos)

    all_videos_df = pd.DataFrame(all_videos)
    all_videos_df['createtime'] = pd.to_datetime(all_videos_df['createtime'], utc=True)
    all_videos_df = all_videos_df.drop_duplicates(subset=['id'])
    years = all_videos_df['createtime'].dt.year.unique()
    print("All years")
    for year in tqdm.tqdm(years):
        year_ids = all_videos_df[all_videos_df['createtime'].dt.year == year]['id'].to_list()
        if len(year_ids) < 2:
            continue
        year_fig_dir_path = os.path.join(this_dir_path, '..', 'figs', str(year))
        if not os.path.exists(year_fig_dir_path):
            os.makedirs(year_fig_dir_path)
        do_analysis(year_ids, year_fig_dir_path)

    # canadian_comment_ids = []
    # for filename in os.listdir(os.path.join(data_dir_path, 'canada', 'comments')):
    #     if filename.endswith(".parquet.gzip"):
    #         comment_df = pd.read_parquet(os.path.join(data_dir_path, 'canada', 'comments', filename))
    #         canadian_comment_ids.extend(comment_df['comment_id'].to_list())
    # canadian_comment_ids = list(set(canadian_comment_ids))
    # canadian_fig_comment_dir_path = os.path.join(this_dir_path, "..", "figs", "canada_comments")
    # if not os.path.exists(canadian_fig_comment_dir_path):
    #     os.makedirs(canadian_fig_comment_dir_path)
    # do_analysis(canadian_comment_ids, canadian_fig_comment_dir_path)

    # canadian_user_ids = []
    # for filename in os.listdir(os.path.join(data_dir_path, 'canada', 'comments')):
    #     if filename.endswith(".parquet.gzip"):
    #         comment_df = pd.read_parquet(os.path.join(data_dir_path, 'canada', 'comments', filename))
    #         canadian_user_ids.extend(comment_df['author_id'].to_list())
    # canadian_user_ids = list(set(canadian_user_ids))
    # canadian_fig_user_dir_path = os.path.join(this_dir_path, "..", "figs", "canada_users")
    # if not os.path.exists(canadian_fig_user_dir_path):
    #     os.makedirs(canadian_fig_user_dir_path)
    # do_analysis(canadian_user_ids, canadian_fig_user_dir_path)

    print("All videos")
    all_fig_dir_path = os.path.join(this_dir_path, "..", "figs", "all_videos")
    all_video_ids = all_videos_df['id'].to_list()
    do_analysis(all_video_ids, all_fig_dir_path)



    


if __name__ == "__main__":
    main()
