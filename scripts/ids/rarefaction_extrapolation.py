import collections
import datetime
import json
import math
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
import tqdm


from copia.data import to_copia_dataset
from copia.estimators import ace, iChao1, chao1, egghe_proot
from copia.plot import accumulation_curve
from copia.rarefaction_extrapolation import species_accumulation
from simple_good_turing import SimpleGoodTuring
import goodturingestimator

def get_rarefaction_extrapolation(video_other_bits, fig_dir_path):
    interval = (10,31)
    section = [int(b[interval[0]:interval[1]+1], 2) for b in video_other_bits]

    df = pd.DataFrame({'end_id': section})
    df = df.reset_index()
    ds = to_copia_dataset(section, data_type="abundance", input_type="observations")

    counts = dict(collections.Counter(section))
    unbiased_estimate = ace(ds)
    # estimator = SimpleGoodTuring(counts, max(counts.values()))
    # species = estimator.run_sgt()
    probs, p0 = goodturingestimator.simpleGoodTuringProbs(counts)
    print(f"Percentage of IDs captured: {(100 * (1-p0)):.2f}%")

    n_jobs = multiprocessing.cpu_count() - 1
    
    # accumulation = species_accumulation(ds, max_steps=len(section) * 2, n_jobs=n_jobs)
    # fig, ax = plt.subplots()
    # accumulation_curve(ds, accumulation, ax=ax, xlabel='Number of IDs', ylabel='Number of unique Bit Patterns')
    # fig.savefig(os.path.join(fig_dir_path, 'rarefaction_extrapolation.png'))

    empirical_richness = len(set(section))
    
    # chao1_estimate = chao1(ds)
    # iChao1_estimate = iChao1(ds)
    # egghe_proot_estimate = egghe_proot(ds)
    return section
    
    

def do_analysis(video_ids, fig_dir_path):
    video_bits = [format(int(id), '064b') for id in video_ids]

    video_other_bits = [b[32:] for b in video_bits]
    return get_rarefaction_extrapolation(video_other_bits, fig_dir_path)

def do_country_analysis(country):
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', '..', 'data')
    country_videos = []
    video_dir_path = os.path.join(data_dir_path, 'countries', country, 'videos')
    for filename in os.listdir(video_dir_path):
        if filename.endswith(".json"):
            with open(os.path.join(video_dir_path, filename), 'r') as file:
                videos = json.load(file)
            if 'author' in videos:
                videos = pd.DataFrame(videos).to_dict('records')
            country_videos.extend([{'id': v['id'], 'createtime': datetime.datetime.fromtimestamp(v['createTime'])} for v in videos])
        elif 'parquet' in filename:
            video_df = pd.read_parquet(os.path.join(video_dir_path, filename))
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
    last_section = do_analysis(country_video_ids, country_fig_dir_path)
    country_videos = country_video_df.to_dict('records')
    return country_videos, last_section

def get_overlaps(country_sections):
    """
    Create two heatmaps showing both set overlaps and distribution overlaps
    between countries' sections.
    """
    countries = list(country_sections.keys())
    n_countries = len(countries)
    
    # Calculate set overlaps
    set_overlaps = np.zeros((n_countries, n_countries))
    
    # Calculate distribution overlaps
    dist_overlaps = np.zeros((n_countries, n_countries))
    
    # Get all unique sections for distribution analysis
    all_sections = sorted(set().union(*country_sections.values()))
    
    # Create distribution vectors for each country
    country_distributions = {}
    for country, sections in country_sections.items():
        # Count frequency of each section
        counts = collections.Counter(sections)
        # Create normalized distribution vector
        dist = np.array([counts.get(section, 0) for section in all_sections])
        dist = dist / np.sum(dist) if np.sum(dist) > 0 else dist
        country_distributions[country] = dist
    
    # Calculate both overlap matrices
    for i, country1 in enumerate(countries):
        for j, country2 in enumerate(countries):
            # Set overlap (Jaccard index)
            set1 = set(country_sections[country1])
            set2 = set(country_sections[country2])
            overlap = set1.intersection(set2)
            union = set1.union(set2)
            set_overlaps[i, j] = len(overlap) / len(union)
            
            # Distribution overlap (1 - Jensen-Shannon divergence)
            dist1 = country_distributions[country1]
            dist2 = country_distributions[country2]
            if i == j:
                dist_overlaps[i, j] = 1.0
            else:
                # JSD returns a value between 0 and 1, where 0 means identical distributions
                # We subtract from 1 to make it an overlap measure rather than a distance
                dist_overlaps[i, j] = 1 - jensenshannon(dist1, dist2)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot set overlaps
    im1 = ax1.imshow(set_overlaps)
    ax1.set_title('Set Overlap (Jaccard Index)')
    plt.colorbar(im1, ax=ax1)
    
    # Plot distribution overlaps
    im2 = ax2.imshow(dist_overlaps)
    ax2.set_title('Distribution Overlap (1 - Jensen-Shannon)')
    plt.colorbar(im2, ax=ax2)
    
    # Configure both axes
    for ax in (ax1, ax2):
        ax.set_xticks(np.arange(len(countries)))
        ax.set_yticks(np.arange(len(countries)))
        ax.set_xticklabels(countries)
        ax.set_yticklabels(countries)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(countries)):
            for j in range(len(countries)):
                value = set_overlaps[i, j] if ax == ax1 else dist_overlaps[i, j]
                text = ax.text(j, i, f"{value:.2f}",
                             ha="center", va="center",
                             color="white" if value > 0.5 else "black")
    
    plt.tight_layout()
    fig.savefig(os.path.join('.', 'figs', 'overlaps.png'))


def read_result_path(result_path):
    with open(result_path, 'r') as f:
        try:
            results = json.load(f)
        except:
            return []
    return [r['args'] for r in results if possible_created_video(r)]

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")

    all_videos = []
    countries = ['brazil', 'canada', 'germany', 'indonesia', 'nigeria', 'india', 'australia', 'japan', 'russia']

    print("All countries")
    # with multiprocessing.Pool(processes=len(countries)) as pool:
    #     all_country_videos = pool.map(do_country_analysis, countries)
    # for country_videos in all_country_videos:
    #     all_videos.extend(country_videos)
    country_sections = {}
    for country in countries:
        print(f"Country: {country}")
        all_country_videos, last_section = do_country_analysis(country)
        print(f"Number of videos: {len(all_country_videos)}")
        all_videos.extend(all_country_videos)
        country_sections[country] = last_section

    get_overlaps(country_sections)

    
    all_videos_df = pd.DataFrame(all_videos)
    all_videos_df['createtime'] = pd.to_datetime(all_videos_df['createtime'], utc=True)
    all_videos_df = all_videos_df.drop_duplicates(subset=['id'])
    years = all_videos_df['createtime'].dt.year.unique()
    print("All years")
    def do_year_analysis(year):
        year_ids = all_videos_df[all_videos_df['createtime'].dt.year == year]['id'].to_list()
        if len(year_ids) < 2:
            return
        year_fig_dir_path = os.path.join(this_dir_path, '..', 'figs', str(year))
        if not os.path.exists(year_fig_dir_path):
            os.makedirs(year_fig_dir_path)
        do_analysis(year_ids, year_fig_dir_path)

    # process_amap(do_year_analysis, years, pbar_desc="Do year analysis")

    print("All videos")
    all_fig_dir_path = os.path.join(this_dir_path, "..", "figs", "all_videos")
    all_video_ids = all_videos_df['id'].to_list()
    print(f"Number of videos: {len(all_video_ids)}")
    do_analysis(all_video_ids, all_fig_dir_path)



    


if __name__ == "__main__":
    main()
