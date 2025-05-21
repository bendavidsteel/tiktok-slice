import os
import shutil

from tqdm import tqdm

def main():
    base_dir = '/media/bsteel/NAS/TikTok_Kids/Studie_2025/Kinderaccounts/Training_Data'
    all_dir = base_dir + '/All'
    child_dir = base_dir + '/Children_only'

    all_dirs = [all_dir + sub_d for sub_d in ['/50/50', '/100/100', '/150/150']]

    all_accounts = [os.path.join(p_d, d) for p_d in all_dirs for d in os.listdir(p_d)]
    child_accounts = os.listdir(child_dir)
    child_videos = []
    for child_account in child_accounts:
        vids_path = os.path.join(child_dir, child_account, 'vids')
        if os.path.exists(vids_path):
            for file in os.listdir(vids_path):
                child_videos.append(os.path.join(vids_path, file))
        else:
            pass

    all_videos = []
    for account in all_accounts:
        for file in os.listdir(os.path.join(all_dir, account, 'vids')):
            all_videos.append(os.path.join(all_dir, account, 'vids', file))

    child_file_names = [os.path.basename(f) for f in child_videos]
    non_child_videos = [f for f in all_videos if os.path.basename(f) not in child_file_names]
    
    for child_video in tqdm(child_videos):
        shutil.move(child_video, os.path.join('./data', 'children_val_set', 'child_videos', os.path.basename(child_video)))

    for non_child_video in tqdm(non_child_videos):
        shutil.move(non_child_video, os.path.join('./data', 'children_val_set', 'non_child_videos', os.path.basename(non_child_video)))

if __name__ == "__main__":
    main()