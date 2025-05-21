import datetime
import json
import os

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data", "germany")
    with open(os.path.join(data_dir_path, 'videos', 'all_010324.json'), 'r') as file:
        videos = json.load(file)

    diffs = []
    for v in videos:
        video_id = v['id']
        bits = format(int(video_id), '064b')
        timestamp = datetime.datetime.fromtimestamp(int(bits[:32], 2))
        video_createtime = datetime.datetime.fromtimestamp(v['createTime'])
        diff = abs((timestamp - video_createtime).total_seconds())
        diffs.append(diff)

    print(f"Mean difference: {sum(diffs) / len(diffs)}")
    print(f"Max difference: {max(diffs)}")

if __name__ == "__main__":
    main()