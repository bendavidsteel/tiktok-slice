import datetime

import get_random_sample

def main():
    generation_strategy = 'all'
    start_time = datetime.datetime(2024, 3, 1, 16, 0, 0)
    num_time = 1
    time_unit = 's'
    num_workers = 32
    reqs_per_ip = 2000
    batch_size = 80000
    task_batch_size = 120
    task_nthreads = 12
    task_timeout = 20
    worker_cpu = 256
    worker_mem = 512
    cluster_type = 'raspi'
    method = 'async'
    get_random_sample.get_random_sample(
        generation_strategy,
        start_time,
        num_time,
        time_unit,
        num_workers,
        reqs_per_ip,
        batch_size,
        task_batch_size,
        task_nthreads,
        task_timeout,
        worker_cpu,
        worker_mem,
        cluster_type,
        method
    )

if __name__ == '__main__':
    main()