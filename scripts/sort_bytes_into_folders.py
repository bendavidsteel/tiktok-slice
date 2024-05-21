
import os
import re
import shutil

import tqdm

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")
    bytes_dir_path = os.path.join('/', 'media', 'bsteel', 'Elements', 'repos', 'what-for-where', 'data', 'bytes')

    for filename in tqdm.tqdm(os.listdir(bytes_dir_path)):
        video_id = re.search(r"([0-9]+)\.mp4", filename).group(1)
        id_bits = format(int(video_id), '064b')
        timestamp_bits = id_bits[:32]
        timestamp = int(timestamp_bits, 2)
        timestamp_dir = os.path.join(bytes_dir_path, str(timestamp))
        if not os.path.exists(timestamp_dir):
            os.makedirs(timestamp_dir)
        shutil.move(os.path.join(bytes_dir_path, filename), os.path.join(timestamp_dir, filename))

if __name__ == '__main__':
    main()