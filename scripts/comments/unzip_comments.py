import os
import subprocess

import tqdm

def main():
    comments_dir_path = os.path.join('.', 'data', 'comments')
    for zip_file_name in tqdm.tqdm(os.listdir(comments_dir_path)):
        if zip_file_name.endswith('.zip'):
            zip_file_path = os.path.join(comments_dir_path, zip_file_name)
            subprocess.run(['unzip', zip_file_path, '-d', f"./data/comments/{zip_file_name.split('.')[0]}"])

if __name__ == '__main__':
    main()
