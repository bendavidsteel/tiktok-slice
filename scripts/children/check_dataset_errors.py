import joblib
import os

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from train_classifier import get_videos_embeddings, StackingEnsembleClassifier

def main():
    child_dir = os.path.join('.', 'data', 'children_val_set', 'child_videos')
    non_child_dir = os.path.join('.', 'data', 'children_val_set', 'non_child_videos')

    classifier = joblib.load('./models/stacking_ensemble.joblib')

    child_embeddings, child_img_features, child_video_df = get_videos_embeddings(child_dir)
    non_child_embeddings, non_child_img_features, non_child_video_df = get_videos_embeddings(non_child_dir)

    df = pl.concat([child_video_df.with_row_index(), non_child_video_df.with_row_index()])

    X = np.vstack([child_embeddings, non_child_embeddings])
    y = np.concatenate([
        np.ones(len(child_embeddings)),
        np.zeros(len(non_child_embeddings))
    ])
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, df['aweme_id'].to_numpy(), test_size=0.5, random_state=42, stratify=y
    )

    test_probs = classifier.predict_proba(X_test)
    threshold = 0.43
    test_pred = test_probs > threshold
    # get false positives
    print("False Positives")
    false_positive_idx = np.where((test_pred == True) & (y_test == False))[0]
    for aweme_id in id_test[false_positive_idx][:10]:
        file_path = f'./data/children_val_set/non_child_videos/{aweme_id}.mp4'
        print(f"File path: {file_path}")

    print("True Negatives")
    false_positive_idx = np.where((test_pred == False) & (y_test == True))[0]
    for aweme_id in id_test[false_positive_idx][:10]:
        file_path = f'./data/children_val_set/child_videos/{aweme_id}.mp4'
        print(f"File path: {file_path}")



if __name__ == '__main__':
    main()