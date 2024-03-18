import json
import os
import re

import geopandas as gpd
from geopy import distance
import ipinfo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import get_cdns

def main():
    ip_info_df = get_cdns.get_ip_info_df()

    data_df = ip_info_df[['request_ip_dist_share', 'origin_ip_dist_share', 'origin_region']].dropna()
    X = data_df[['request_ip_dist_share', 'origin_ip_dist_share']].values
    y = data_df['origin_region'].values
    # map y to unique integers
    y_map = {region: i for i, region in enumerate(np.unique(y))}
    y = np.array([y_map[region] for region in y])

    clf = LogisticRegression(random_state=0).fit(X, y)
    print(f"Accuracy: {clf.score(X, y)}")
    print(f"Coefficients: {clf.coef_}")

if __name__ == '__main__':
    main()