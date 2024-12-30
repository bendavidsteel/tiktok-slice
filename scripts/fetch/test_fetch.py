from get_random_sample import TaskDataset

def test_task_dataset():
    potential_video_ids = [1, 2, 3, 4, 5]

    dataset = TaskDataset()
    dataset.add_potential_ids(potential_video_ids)

    tasks = dataset.get_batch(5)
    for task in tasks:
        task.result = {'result': 1}

    dataset.update_tasks(tasks)

    assert dataset.tasks['result'].to_list() == [{'result': 1} for _ in range(5)]

def main():
    test_task_dataset()

if __name__ == '__main__':
    main()