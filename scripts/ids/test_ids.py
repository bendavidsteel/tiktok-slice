import datetime
import itertools
import json
import os

def main(
        generation_strategy,
        start_time,
        num_time,
        time_unit
    ):
    print(f"Getting random sample at {start_time} for {num_time} {time_unit}")
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    
    with open(os.path.join(this_dir_path, '..', 'figs', 'all_videos', f'{generation_strategy}_two_segments_combinations.json'), 'r') as file:
        data = json.load(file)

    # get bits of non timestamp sections of ID
    # order dict according to interval
    data = [(tuple(map(int, interval.strip('()').split(', '))), vals) for interval, vals in data.items()]
    data = sorted(data, key=lambda x: x[0][0])
    # get rid of millisecond bits
    data = [t for t in data if t[0] != (0,9)]
    interval_bits = []
    intervals = [d[0] for d in data]
    for interval, vals in data:
        # format ints to binary
        num_bits = interval[1] - interval[0] + 1
        bits = [format(i, f'0{num_bits}b') for i in vals]
        interval_bits.append(bits)
    other_bit_sequences = itertools.product(*interval_bits)
    other_bit_sequences = [''.join(bits) for bits in other_bit_sequences]

    # get all videos in 1 millisecond
    
    unit_map = {
        'ms': 'milliseconds',
        's': 'seconds',
        'm': 'minutes',
    }
    time_delta = datetime.timedelta(**{unit_map[time_unit]: num_time})
    
    end_time = start_time + time_delta
    c_time = start_time
    all_timestamp_bits = []
    while c_time < end_time:
        unix_timestamp_bits = format(int(c_time.timestamp()), '032b')
        milliseconds = int(format(c_time.timestamp(), '.3f').split('.')[1])
        milliseconds_bits = format(milliseconds, '010b')
        timestamp_bits = unix_timestamp_bits + milliseconds_bits
        all_timestamp_bits.append(timestamp_bits)
        c_time += datetime.timedelta(milliseconds=1)

    potential_video_bits = itertools.product(all_timestamp_bits, other_bit_sequences)
    potential_video_bits = [''.join(bits) for bits in potential_video_bits]
    potential_video_ids = [int(bits, 2) for bits in potential_video_bits]


if __name__ == '__main__':
    main('all', datetime.datetime(2024, 4, 10), 1, 'm')