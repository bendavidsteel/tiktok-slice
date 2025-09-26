import os

import polars as pl
from tqdm import tqdm

from detect_aigc import get_middle_frame

def get_result_paths(result_dir_path, result_filename='results.parquet.gzip', minute=None, hour=None):
    for dir_path, dir_names, filenames in os.walk(result_dir_path):
        for filename in filenames:
            if filename == result_filename:
                file_hour, file_minute = map(int, dir_path.split('/')[-3:-1])
                if hour is not None and file_hour != hour:
                    continue
                if minute is not None and file_minute != minute:
                    continue
                result_path = os.path.join(dir_path, filename)
                yield result_path


def main():
    # Read config and data
    base_result_path = os.path.join('.', 'data', 'results', '2024_04_10')
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    use = '24hour'
    if use == 'all':
        output_dir_path = os.path.join(this_dir_path, '..', "..", "data", "stats", 'all')
        result_paths = list(get_result_paths(base_result_path, result_filename='videos.parquet.zstd'))
    elif use == '24hour':
        output_dir_path = os.path.join(this_dir_path, '..', "..", "data", "stats", '24hour')
        result_paths = list(get_result_paths(base_result_path, result_filename='videos.parquet.zstd', minute=42))
    elif use == '1hour':
        output_dir_path = os.path.join(this_dir_path, '..', "..", "data", "stats", '1hour')
        result_paths = list(get_result_paths(base_result_path, result_filename='videos.parquet.zstd', hour=19))

    result_paths = sorted(result_paths)

    os.makedirs(output_dir_path, exist_ok=True)

    # result_paths = result_paths[:1000]
    video_df = None
    val_count_dfs = None
    for result_path in tqdm(result_paths):
        batch_df = pl.read_parquet(result_path, columns=['id', 'locationCreated', 'aigcLabelType'])
        if video_df is not None:
            video_df = pl.concat([video_df, batch_df], how='diagonal_relaxed')
        else:
            video_df = batch_df

    video_df = video_df.with_columns((pl.col('aigcLabelType').is_in(['1', '2']) & pl.col('aigcLabelType').is_not_null()).alias('tiktok_aigc_pred'))
    
    aigc_predictions_path = './data/aigc/predictions.parquet.zstd'
    aigc_prediction_df = pl.read_parquet(aigc_predictions_path)
    aigc_prediction_df = aigc_prediction_df.with_columns([
        pl.col('path').str.split('/').list.get(-1).str.split('.').list.get(0).alias('id'),
        (pl.col('prob_fake') > 0.5).alias('classifier_aigc_pred')
    ])

    video_df = video_df.join(aigc_prediction_df, on='id', how='inner')

    aif_aigc_pred_df = pl.read_parquet('./data/aigc/24hour_0.5_finetune_predictions.parquet.zstd')
    aif_aigc_pred_df = aif_aigc_pred_df.with_columns([
        pl.col('path').str.split('/').list.get(-1).str.split('.').list.get(0).alias('id'),
        pl.col('pred').cast(pl.Boolean).alias('aif_aigc_pred')
    ])
    video_df = video_df.join(aif_aigc_pred_df, on='id', how='inner')

    # write out confusion matrix
    tiktok_no_classifier_no_aif_no_df = video_df.filter(~pl.col('tiktok_aigc_pred') & ~pl.col('classifier_aigc_pred') & ~pl.col('aif_aigc_pred'))
    tiktok_no_classifier_no_aif_yes_df = video_df.filter(~pl.col('tiktok_aigc_pred') & ~pl.col('classifier_aigc_pred') & pl.col('aif_aigc_pred'))
    tiktok_no_classifier_yes_aif_no_df = video_df.filter(~pl.col('tiktok_aigc_pred') & pl.col('classifier_aigc_pred') & ~pl.col('aif_aigc_pred'))
    tiktok_no_classifier_yes_aif_yes_df = video_df.filter(~pl.col('tiktok_aigc_pred') & pl.col('classifier_aigc_pred') & pl.col('aif_aigc_pred'))
    tiktok_yes_classifier_no_aif_no_df = video_df.filter(pl.col('tiktok_aigc_pred') & ~pl.col('classifier_aigc_pred') & ~pl.col('aif_aigc_pred'))
    tiktok_yes_classifier_no_aif_yes_df = video_df.filter(pl.col('tiktok_aigc_pred') & ~pl.col('classifier_aigc_pred') & pl.col('aif_aigc_pred'))
    tiktok_yes_classifier_yes_aif_no_df = video_df.filter(pl.col('tiktok_aigc_pred') & pl.col('classifier_aigc_pred') & ~pl.col('aif_aigc_pred'))
    tiktok_yes_classifier_yes_aif_yes_df = video_df.filter(pl.col('tiktok_aigc_pred') & pl.col('classifier_aigc_pred') & pl.col('aif_aigc_pred'))
    print(f"Confusion Matrix:\n"
          f"  TikTok No, Classifier No, AIF No: {len(tiktok_no_classifier_no_aif_no_df)}\n"
          f"  TikTok No, Classifier No, AIF Yes: {len(tiktok_no_classifier_no_aif_yes_df)}\n"
          f"  TikTok No, Classifier Yes, AIF No: {len(tiktok_no_classifier_yes_aif_no_df)}\n"
          f"  TikTok No, Classifier Yes, AIF Yes: {len(tiktok_no_classifier_yes_aif_yes_df)}\n"
          f"  TikTok Yes, Classifier No, AIF No: {len(tiktok_yes_classifier_no_aif_no_df)}\n"
          f"  TikTok Yes, Classifier No, AIF Yes: {len(tiktok_yes_classifier_no_aif_yes_df)}\n"
          f"  TikTok Yes, Classifier Yes, AIF No: {len(tiktok_yes_classifier_yes_aif_no_df)}\n"
          f"  TikTok Yes, Classifier Yes, AIF Yes: {len(tiktok_yes_classifier_yes_aif_yes_df)}\n")
    
    # link examples for 5 from each category
    example_dir_path = './data/aigc/examples'
    num_examples = 10
    print("\nExamples:")
    for df, label in zip([tiktok_no_classifier_no_aif_no_df,
                          tiktok_no_classifier_no_aif_yes_df,
                          tiktok_no_classifier_yes_aif_no_df,
                          tiktok_no_classifier_yes_aif_yes_df,
                          tiktok_yes_classifier_no_aif_no_df,
                          tiktok_yes_classifier_no_aif_yes_df,
                          tiktok_yes_classifier_yes_aif_no_df,
                          tiktok_yes_classifier_yes_aif_yes_df],
                         ['TikTok No, Classifier No, AIF No',
                          'TikTok No, Classifier No, AIF Yes',
                          'TikTok No, Classifier Yes, AIF No',
                          'TikTok No, Classifier Yes, AIF Yes',
                          'TikTok Yes, Classifier No, AIF No',
                          'TikTok Yes, Classifier No, AIF Yes',
                          'TikTok Yes, Classifier Yes, AIF No',
                          'TikTok Yes, Classifier Yes, AIF Yes']):
        print(f"  {label}:")
        label_dir_path = os.path.join(example_dir_path, label.replace(' ', '_').replace(',', '').lower())
        os.makedirs(label_dir_path, exist_ok=True)
        for example in df.head(num_examples).to_dicts():
            print(f"    - {example['path']}")
            middle_frame = get_middle_frame(example['path'])
            if middle_frame is not None:
                middle_frame.save(os.path.join(label_dir_path, f"{example['id']}.jpg"))

if __name__ == '__main__':
    main()
