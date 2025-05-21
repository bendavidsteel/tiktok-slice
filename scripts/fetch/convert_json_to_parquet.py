import asyncio
import json
import os

import asyncssh
import pandas as pd
import tqdm




def cleanup(result):
    if result['result']['return'] and 'item_control' in result['result']['return'] and len(result['result']['return']['item_control']) == 0:
        result['result']['return']['item_control'] = None
    return result

def write_parquet(results, result_path):
    pq_results = [cleanup(r) for r in results]
    df = pd.DataFrame(pq_results)
    parquet_path = result_path.replace('results.json', 'results.parquet.gzip')
    df.to_parquet(parquet_path, compression='gzip')
    return parquet_path

def convert_json_to_parquet(result_path):
    
    
    try:
        with open(result_path, 'r') as f:
            results = json.load(f)
    except json.JSONDecodeError:
        with open(result_path, 'r') as f:
            json_string = f.read()
            while True:
                if not json_string:
                    raise ValueError("Couldn't fix JSON")
                try:
                    results = json.loads(json_string + "]")
                except json.decoder.JSONDecodeError:
                    last_right_curly = json_string.rfind("}")
                    if last_right_curly == -1:
                        raise ValueError("Couldn't fix JSON")
                    elif last_right_curly == len(json_string) - 1:
                        json_string = json_string[:last_right_curly]
                    json_string = json_string[:last_right_curly+1]
                    continue
                break
    parquet_path = write_parquet(results, result_path)
    
    # df = pd.read_parquet(parquet_path)
    # # check contents same as results
    # df_results = df.to_dict('records')
    # assert len(df_results) == len(results)
    # for df_result, result in zip(df_results, results):
    #     assert df_result['args'] == result['args']
    #     if 'id' in result:
    #         assert df_result['id'] == result['id']
    #         assert df_result['video'] == result['video']
    #     elif 'statusCode' in result:
    #         assert int(df_result['statusCode']) == int(result['statusCode'])
    #     assert len(df_result['exceptions']) == len(result['exceptions'])
    #     assert df_result['completed'] == result['completed']
    os.remove(result_path)
    return parquet_path

async def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data")
    remote = True
    if remote:
        local_result_dir = os.path.join(data_dir_path, 'results')
        conn = await asyncssh.connect(os.environ['PRODESK_HOST'], username=os.environ['USERNAME'], password=os.environ['PASSWORD'])
        result_paths = await get_remote_result_paths(conn)
    else:
        result_paths = list(get_result_paths(data_dir_path))

    for found_result_path in tqdm.tqdm(result_paths):
        if remote:
            local_result_path = os.path.join(local_result_dir, found_result_path.split('results/')[1])
            if not os.path.exists(os.path.dirname(local_result_path)):
                os.makedirs(os.path.dirname(local_result_path))
            await asyncssh.scp((conn, found_result_path), local_result_path)
            result_path = local_result_path
        else:
            result_path = found_result_path
        parquet_path = convert_json_to_parquet(result_path)
        if remote:
            remote_parquet_path = found_result_path.replace('results.json', 'results.parquet.gzip')
            await asyncssh.scp(parquet_path, (conn, remote_parquet_path))
            # delete remote json file
            await conn.run(f'rm {found_result_path}')



if __name__ == "__main__":
    asyncio.run(main())