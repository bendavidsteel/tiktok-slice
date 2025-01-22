import configparser
import os
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

def plot_loglog_hist(df, col_name, title, xlabel, ylabel, fig_path, bins=20):
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    
    # Get data range
    bin_start = df[col_name].min()
    bin_end = df[col_name].max()
    
    # Create linear-spaced bins in log space to ensure equal widths
    if bin_start == 0:
        # Handle zero values by starting from 1
        log_start = 0  # Will be transformed to 1 in logspace
        log_end = np.log10(bin_end)
        # Create linearly spaced points in log space
        bin_edges = np.linspace(log_start, log_end, bins + 1)
        # Transform back to original scale
        logbins = np.concatenate(([0], np.power(10, bin_edges)))
    else:
        log_start = np.log10(bin_start)
        log_end = np.log10(bin_end)
        # Create linearly spaced points in log space
        bin_edges = np.linspace(log_start, log_end, bins + 1)
        # Transform back to original scale
        logbins = np.power(10, bin_edges)
    
    # Round bin edges to integers and ensure uniqueness
    logbins = np.unique(np.round(logbins))
    
    # Convert value counts to histogram
    hist = np.zeros(len(logbins) - 1)
    for i in range(len(logbins) - 1):
        hist[i] = df.filter(
            pl.col(col_name).ge(logbins[i]) & 
            pl.col(col_name).lt(logbins[i+1])
        )['count'].sum()
    
    # Calculate widths in log space to ensure visual consistency
    widths = np.diff(logbins)
    
    # Plot bars
    ax.bar(logbins[:-1], hist, width=widths, align='edge')
    
    # Set scales with appropriate parameters and custom tick spacing
    ax.set_xscale('symlog', linthresh=1.0)  # Linear below 1.0
    ax.set_yscale('symlog', linthresh=1.0)
    
    # Import and use SymmetricalLogLocator for custom tick spacing
    from matplotlib.ticker import SymmetricalLogLocator
    ax.xaxis.set_major_locator(SymmetricalLogLocator(linthresh=1.0, base=10))
    
    # Adjust the linear region scaling
    ax.set_xscale('symlog', linthresh=1.0, linscale=0.25)  # Reduce linscale to compress linear region
    
    # Set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.xaxis.label.set_fontsize(8)
    # ax.yaxis.label.set_fontsize(8)

    ax.set_xlim(left=0)
    if len(ax.xaxis.get_ticklabels()) > 5:
        for label in ax.xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save figure
    fig.savefig(fig_path)
    plt.close(fig)  # Close figure to free memory


def load_and_plot(df, col_name, title, xlabel, ylabel, fig_path):
    df = df.drop_nulls()
    plot_loglog_hist(df, col_name, title, xlabel, ylabel, fig_path)
    print(f"Average number of {col_name}: {np.average(df[col_name], weights=df['count'])}")

def finalize_results(output_dir_path, fig_dir_path):
    author_counts_df = pl.read_csv(os.path.join(output_dir_path, "author_unique_id_value_counts.csv"))
    print(f"Number of unique users: {author_counts_df['authorUniqueId'].n_unique()}")
    print(f"Number of videos: {author_counts_df['count'].sum()}")

    comment_counts_df = pl.read_csv(os.path.join(output_dir_path, "comment_count_value_counts.csv"))
    like_counts_df = pl.read_csv(os.path.join(output_dir_path, "like_count_value_counts.csv"))
    share_counts_df = pl.read_csv(os.path.join(output_dir_path, "share_count_value_counts.csv"))
    play_counts_df = pl.read_csv(os.path.join(output_dir_path, "play_count_value_counts.csv"))
    duration_counts_df = pl.read_csv(os.path.join(output_dir_path, "video_duration_value_counts.csv"))
    
    load_and_plot(comment_counts_df, "commentCount", "Comment Count Histogram", "Comment Count", "Frequency", os.path.join(fig_dir_path, "comment_count_hist.png"))
    load_and_plot(like_counts_df, "diggCount", "Like Count Histogram", "Like Count", "Frequency", os.path.join(fig_dir_path, "like_count_hist.png"))
    load_and_plot(share_counts_df, "shareCount", "Share Count Histogram", "Share Count", "Frequency", os.path.join(fig_dir_path, "share_count_hist.png"))
    load_and_plot(play_counts_df, "playCount", "Play Count Histogram", "View Count", "Frequency", os.path.join(fig_dir_path, "play_count_hist.png"))
    load_and_plot(duration_counts_df, "videoDuration", "Video Duration Histogram", "Video Duration", "Frequency", os.path.join(fig_dir_path, "video_duration_hist.png"))

def main():
    config = configparser.ConfigParser()
    config.read('./config/config.ini')
    output_dir_path = os.path.join(".", "data", "stats", "1hour")
    os.makedirs(output_dir_path, exist_ok=True)
    fig_dir_path = os.path.join(".", "figs", "stats")
    os.makedirs(fig_dir_path, exist_ok=True)
    finalize_results(output_dir_path, fig_dir_path)

if __name__ == "__main__":
    main()