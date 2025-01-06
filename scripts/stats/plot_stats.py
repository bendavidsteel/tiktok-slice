import configparser
import os
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

def plot_loglog_hist(df, col_name, title, xlabel, ylabel, fig_path, bins=20):
    fig, ax = plt.subplots()
    
    # Get data range
    bin_start = df[col_name].min()
    bin_end = df[col_name].max()
    
    # Create log-spaced bins
    if bin_start == 0:
        # Create log-spaced bins starting from 1, but make first bin start at 0
        bins = np.unique(np.round(np.logspace(0, np.log10(bin_end), bins)))
        # Make the first bin start at 0 instead of 1
        logbins = np.concatenate(([0], bins))
    else:
        # Round bin edges to integers and remove duplicates
        logbins = np.unique(np.round(np.logspace(np.log10(bin_start), np.log10(bin_end), bins+1)))
    
    # Convert value counts to histogram
    hist = np.zeros(len(logbins) - 1)
    for i in range(len(logbins) - 1):
        hist[i] = df.filter(pl.col(col_name).ge(logbins[i]) & pl.col(col_name).lt(logbins[i+1]))['count'].sum()
    
    # Calculate bin edges and widths for plotting
    # x_pos = np.sqrt(logbins[:-1] * logbins[1:])
    widths = np.diff(logbins)
    
    # Plot bars using x_pos for center positions
    ax.bar(logbins[:-1], hist, width=widths, align='edge')
    
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    # ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    fig.savefig(fig_path)

def load_and_plot(df, col_name, title, xlabel, ylabel, fig_path):
    df = df.drop_nulls()
    plot_loglog_hist(df, col_name, title, xlabel, ylabel, fig_path)
    print(f"Average number of {col_name}: {np.average(df[col_name], weights=df['count'])}")

def finalize_results(output_dir_path, fig_dir_path):
    comment_counts_df = pl.read_csv(os.path.join(output_dir_path, "comment_count_value_counts.csv"))
    like_counts_df = pl.read_csv(os.path.join(output_dir_path, "like_count_value_counts.csv"))
    share_counts_df = pl.read_csv(os.path.join(output_dir_path, "share_count_value_counts.csv"))
    play_counts_df = pl.read_csv(os.path.join(output_dir_path, "play_count_value_counts.csv"))
    duration_counts_df = pl.read_csv(os.path.join(output_dir_path, "video_duration_value_counts.csv"))
    
    load_and_plot(comment_counts_df, "commentCount", "Comment Count Histogram", "Comment Count", "Frequency", os.path.join(fig_dir_path, "comment_count_hist.png"))
    load_and_plot(like_counts_df, "diggCount", "Like Count Histogram", "Like Count", "Frequency", os.path.join(fig_dir_path, "like_count_hist.png"))
    load_and_plot(share_counts_df, "shareCount", "Share Count Histogram", "Share Count", "Frequency", os.path.join(fig_dir_path, "share_count_hist.png"))
    load_and_plot(play_counts_df, "playCount", "Play Count Histogram", "Play Count", "Frequency", os.path.join(fig_dir_path, "play_count_hist.png"))
    load_and_plot(duration_counts_df, "videoDuration", "Video Duration Histogram", "Video Duration", "Frequency", os.path.join(fig_dir_path, "video_duration_hist.png"))

def main():
    config = configparser.ConfigParser()
    config.read('./config/config.ini')
    output_dir_path = os.path.join(".", "data", "stats", "1hour") # TODO to change to 1 hour
    os.makedirs(output_dir_path, exist_ok=True)
    fig_dir_path = os.path.join(".", "figs", "stats")
    os.makedirs(fig_dir_path, exist_ok=True)
    finalize_results(output_dir_path, fig_dir_path)

if __name__ == "__main__":
    main()