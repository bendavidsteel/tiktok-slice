import datetime
import os

import polars as pl

from get_random_sample import TaskDataset

def test_task_dataset():
    potential_video_ids = [1, 2, 3, 4, 5]

    dataset = TaskDataset()
    dataset.add_potential_ids(potential_video_ids)

    tasks = dataset.get_batch(5)
    results = []
    for i, task in enumerate(tasks):
        if i % 2 == 0:
            task.result = {'return': {'video_id': 1}, 'post_time': datetime.datetime.now()}
            results.append(task.result)
        else:
            task.result = {'return': None, 'post_time': datetime.datetime.now()}
            task.exceptions.append({'exception': Exception()})
            results.append(task.result)

    dataset.update_tasks(tasks)

    for r, row in zip(results, dataset.tasks.to_dicts()):
        assert r['return'] == row['result']['return']

def test_existing_task_dataset():

    results_dir_path = './data/results/2024_04_10/hours/19/0/0'
    existing_df = pl.read_parquet(os.path.join(results_dir_path, 'results.parquet.gzip'))

    dataset = TaskDataset()
    dataset.load_existing_df(existing_df)

    # add ids that haven't been collected
    existing_ids = set(existing_df['args'])
    potential_video_ids = [1, 2, 3]
    potential_video_ids = [i for i in potential_video_ids if i not in existing_ids]
    dataset.add_potential_ids(potential_video_ids)

    assert dataset.num_left() == 3

def main():
    test_existing_task_dataset()
    test_task_dataset()

if __name__ == '__main__':
    main()