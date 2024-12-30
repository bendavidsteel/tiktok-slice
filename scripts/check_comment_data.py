import json
import os

import pandas as pd

def main():
    comments_path = './data/10'
    fps = [os.path.join(comments_path, fp) for fp in os.listdir(comments_path)]

    comments = []
    for fp in fps:
        with open(fp, 'r') as f:
            ls = f.readlines()
            ds = [json.loads(l) for l in ls]
            for d in ds:
                if d and 'data' in d and d['data'] and 'comments' in d['data']:
                    comments += d['data']['comments']

    pass

if __name__ == '__main__':
    main()