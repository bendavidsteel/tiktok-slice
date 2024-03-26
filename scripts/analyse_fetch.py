import json
import os

import matplotlib.pyplot as plt
import pandas as pd

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', 'data')

    time_span = '100ms'
    num_workers = 64
    rpip = 100
    with open(os.path.join(data_dir_path, f"{time_span}_nw{num_workers}_rpip{rpip}_potential_video_ids.json")) as f:
        potential_video_ids = json.load(f)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))


if __name__ == '__main__':
    main()