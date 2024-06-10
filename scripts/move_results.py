import os
import shutil
import subprocess

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    results_dir_path = os.path.join(this_dir_path, "..", "data", "results")

    for dir_path, dirname, filenames in os.walk(results_dir_path):
        if 'results.parquet.gzip' in filenames:
            dirs = dir_path.split(os.path.sep)
            results_dir_idx = dirs.index('results')
            datetime_dirs = dirs[results_dir_idx + 1:results_dir_idx + 6]
            if len(dirs[results_dir_idx + 1:]) < 6:
                continue
            new_dir_path = os.path.join('/', *dirs[:results_dir_idx], 'results', *datetime_dirs)
            if not os.path.exists(new_dir_path):
                os.makedirs(new_dir_path)
            for filename in filenames:
                old_path = os.path.join(dir_path, filename)
                new_path = os.path.join(new_dir_path, filename)
                shutil.move(old_path, new_path)
            os.rmdir(dir_path)

                

if __name__ == '__main__':
    main()