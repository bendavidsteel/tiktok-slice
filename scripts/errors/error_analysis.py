import polars as pl

def main():
    day_df = pl.read_csv('./data/stats/1hour/time_counts.csv')
    hour_df = pl.read_csv('./data/stats/24hour/time_counts.csv')
    exist_df = pl.concat([day_df, hour_df]).unique('createTime')
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

    num_original_videos = exist_df['count'].sum()

    for message in messages:
        message_df = error_df.filter(pl.col('statusMsg').str.contains(message, literal=True))
        num_errors = message_df['count'].sum()
        print(f"{message}: {num_errors} ({num_errors / num_original_videos * 100:.2f}%)")

    # how many contain both deleted and self see
    num_errors = error_df.filter(pl.col('statusMsg').str.contains('status_deleted') & pl.col('statusMsg').str.contains('status_self_see'))['count'].sum()
    print(f"Num both: {num_errors}")


if __name__ == "__main__":
    main()