import polars as pl

def main():
    video_id_df = pl.read_csv('./data/stats/all/video_ids.csv')
    error_df = pl.read_csv('./data/stats/all/error_value_counts.csv')

    messages = [
        'status_deleted',
        'status_self_see',
        'status_reviewing',
        'status_audit_not_pass',
        'content_classification',
        'cross_border_violation',
        'copyright_geo_filter'
    ]

    num_original_videos = len(video_id_df) + error_df.filter(pl.col('statusMsg') != "item doesn't exist")['count'].sum()

    for message in messages:
        message_df = error_df.filter(pl.col('statusMsg').str.contains(message, literal=True))
        num_errors = message_df['count'].sum()
        print(f"{message}: {num_errors} ({num_errors / num_original_videos * 100:.2f}%)")


if __name__ == "__main__":
    main()