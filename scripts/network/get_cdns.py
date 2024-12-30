import json
import os
import re

import geopandas as gpd
from geopy import distance
import ipinfo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

def get_ip_info_df():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")

    # Access built-in Natural Earth data via GeoPandas
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Get a list (dataframe) of country centroids
    centroid_list = pd.concat([world.name, world.centroid], axis=1).set_index('name')

    region_map = {
        'UK': 'United Kingdom',
    }
    request_regions = ['UK', 'India', 'Germany', 'Canada', 'Japan']
    origin_regions = ['Germany', 'Canada']
    
    video_cache_data = []
    for origin_region in origin_regions:
        for request_region in request_regions:
            request_region_name = request_region if request_region not in region_map else region_map[request_region]
            request_centroid = centroid_list.loc[request_region_name][0]
            request_centroid = (request_centroid.y, request_centroid.x)


            origin_centroid = centroid_list.loc[origin_region][0]
            origin_centroid = (origin_centroid.y, origin_centroid.x)

            request_region_dir = request_region.lower()
            origin_region_dir = origin_region.lower()
            region_dir_path = os.path.join(data_dir_path, origin_region_dir, "video_ips", request_region_dir)

            if not os.path.exists(region_dir_path):
                continue

            all_videos = []
            for filename in os.listdir(region_dir_path):
                file_path = os.path.join(region_dir_path, filename)
                with open(file_path, "r") as file:
                    video = json.load(file)
                all_videos.append(video)

            for v in all_videos:
                headers = v['network']['play_addr']['headers']
                if 'x-parent-response-time' in headers:
                    ip_response_times = headers['x-parent-response-time']
                    ip_response_times = ip_response_times.split(', ')
                    parent_ips = [ip.split(',')[1] for ip in ip_response_times]
                else:
                    parent_ips = []
                
                cache_ips = []

                # first cache
                first_cache_data = headers['x-cache']
                if 'TCP_HIT' in first_cache_data or 'TCP_MISS' in first_cache_data:
                    first_cache_hit = True if 'TCP_HIT' in first_cache_data else False
                    first_cache_ip = re.search(r"from a([0-9\-]+).deploy", first_cache_data).group(1).replace('-','.')
                    cache_ips.append(first_cache_ip)

                # second cache
                if 'x-cache-remote' in headers:
                    second_cache_data = headers['x-cache-remote']
                    second_cache_hit = True if 'TCP_HIT' in second_cache_data else False
                    second_cache_ip = re.search(r"from a([0-9\-]+).deploy", second_cache_data).group(1).replace('-','.')
                    cache_ips.append(second_cache_ip)

                # more caches
                if len(parent_ips) > 0:
                    for i in range(-1, -len(parent_ips), -1):
                        cache_ips.append(parent_ips[i])
                    final_ip = parent_ips[0]

                if 'akamai-amd-bc-debug' in headers:
                    debug_info = headers['akamai-amd-bc-debug']
                    # contains multiple ips, first one seems to be CDN, others seem to refer to bytedance servers
                    ips = re.findall(r"\[a=([0-9\.]+),", debug_info)
                    cache_ips.append(ips[0])
                    final_ip = ips[-1]

                if 'x-origin-response-time' in headers:
                    ip_response_times = headers['x-origin-response-time']
                    origin_ip = ip_response_times.split(',')[1]
                    if origin_ip not in cache_ips:
                        cache_ips.append(ip)
                    final_ip = ip
                else:
                    origin_ip = None

                d = {
                    'cache_hit': first_cache_hit,
                    'remote_cache_hit': second_cache_hit,
                    'cache_ip': first_cache_ip,
                    'remote_cache_ip': second_cache_ip,
                    'origin_ip': origin_ip,
                    'final_ip': final_ip,
                    'cache_ips': cache_ips,
                    'play_count': v['stats']['playCount'],
                    'request_centroid': request_centroid,
                    'origin_centroid': origin_centroid,
                    'origin_region': origin_region,
                }
                video_cache_data.append(d)

    ips = [v['final_ip'] for v in video_cache_data] + \
        [v['origin_ip'] for v in video_cache_data] + \
        [v['cache_ip'] for v in video_cache_data] + \
        [v['remote_cache_ip'] for v in video_cache_data]

    ipinfo_access_key = "878b0eb8a156d1"
    handler = ipinfo.getHandler(ipinfo_access_key)

    locations = {}
    unique_ips = set(ips)

    ip_detail_cache_path = os.path.join(this_dir_path, '..', 'data', 'ip_detail_cache.json')
    if os.path.exists(ip_detail_cache_path):
        with open(ip_detail_cache_path, 'r') as file:
            ip_detail_cache = json.load(file)
    else:
        ip_detail_cache = {}

    for ip in unique_ips:
        if ip in ip_detail_cache:
            loc = ip_detail_cache[ip]
        else:
            details = handler.getDetails(ip)
            loc = {}
            loc['ip_country'] = details.country
            loc['ip_region'] = details.region
            loc['ip_city'] = details.city
            loc['ip_loc'] = details.loc
        locations[ip] = loc

    with open(os.path.join(this_dir_path, '..', 'data', 'ip_detail_cache.json'), 'w') as file:
        json.dump(locations, file, indent=4)

    for v in video_cache_data:
        if v['cache_ip'] in locations:
            v['cache_ip_loc'] = locations[ip]
        if v['remote_cache_ip'] in locations:
            v['remote_cache_ip_loc'] = locations[ip]
        if v['origin_ip'] in locations:
            v['origin_ip_loc'] = locations[ip]
        if v['final_ip'] in locations:
            v['final_ip_loc'] = locations[ip]
            loc = v['ip_loc']
            request_centroid = v['request_centroid']
            origin_centroid = v['origin_centroid']
            coords = tuple(float(c) for c in loc.split(','))
            v['request_ip_dist'] = distance.distance(request_centroid, coords).km
            v['origin_ip_dist'] = distance.distance(origin_centroid, coords).km
            request_origin_dist = distance.distance(request_centroid, origin_centroid).km
            if request_origin_dist > 0:
                v['request_ip_dist_share'] = v['request_ip_dist'] / request_origin_dist
                v['origin_ip_dist_share'] = v['origin_ip_dist'] / request_origin_dist

    ip_info_df = pd.DataFrame(video_cache_data)

    return ip_info_df

def main():
    ip_info_df = get_ip_info_df()
    # ip_info_df[['ip_country', 'ip_region', 'ip_city']].value_counts().to_csv(os.path.join(data_dir_path, f"{region}_ip_locations.csv"))

    # get regression
    X = np.log(ip_info_df['play_count'])
    X = sm.add_constant(X)
    y = ip_info_df['request_ip_dist_share']
    request_results = OLS(y, X).fit()

    X = np.log(ip_info_df['play_count'])
    X = sm.add_constant(X)
    y = ip_info_df['origin_ip_dist_share']
    origin_results = OLS(y, X).fit()

    # plot view count vs country dist and view count vs canada dist
    fig, axes = plt.subplots(ncols=2)
    axes[0].scatter(ip_info_df['play_count'], ip_info_df['request_ip_dist_share'])
    xlim = [ip_info_df['play_count'].min(), ip_info_df['play_count'].max()]
    axes[0].plot([xlim[0], xlim[1]], [request_results.params.iloc[0] + request_results.params.iloc[1] * np.log(xlim[0]), request_results.params.iloc[0] + request_results.params.iloc[1] * np.log(xlim[1])], color='red', linestyle='--')
    axes[0].set_ylabel('Final IP distance share to requesting country centroid')
    axes[0].set_xlabel('View count')
    axes[0].set_xscale('log')
    axes[1].scatter(ip_info_df['play_count'], ip_info_df['origin_ip_dist_share'])
    xlim = [ip_info_df['play_count'].min(), ip_info_df['play_count'].max()]
    axes[1].plot([xlim[0], xlim[1]], [origin_results.params.iloc[0] + origin_results.params.iloc[1] * np.log(xlim[0]), origin_results.params.iloc[0] + origin_results.params.iloc[1] * np.log(xlim[1])], color='red', linestyle='--')
    axes[1].set_ylabel('Final IP distance share to origin country centroid')
    axes[1].set_xlabel('View count')
    axes[1].set_xscale('log')
    fig.savefig(os.path.join(this_dir_path, '..', 'figs', 'view_dist_scatter.png'))


if __name__ == "__main__":
    main()