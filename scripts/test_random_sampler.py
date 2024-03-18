import asyncio
import datetime
import json
import os
import random
import urllib.parse as url_parsers

import requests

from pytok.tiktok import PyTok
from TikTokApi import TikTokApi
from TikTokApi.exceptions import NotFoundException


async def main():
    common_user = "therock"
    common_user_video_id = 7248300636498890011
    random_comment_id = '1058392585424913160' # random comment id, doesn't have to match real comment

    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", "data", "germany")
    with open(os.path.join(data_dir_path, 'videos', 'all_010324.json'), 'r') as file:
        videos = json.load(file)
    test_video_ids = []
    test_truths = []
    score = 0
    for i in range(100):
        if random.random() > 0.5:
            test_video_ids.append(videos[i]['id'])
            test_truths.append(True)
        else:
            timestamp_1year_time = int((datetime.datetime.now() + datetime.timedelta(days=365)).timestamp())
            # convert to binary
            timestamp_binary = format(timestamp_1year_time, '032b')
            # create random 32 bit number
            random_32bit = format(random.getrandbits(32), '032b')
            # concatenate into 64 bit number
            random_64bit = int(timestamp_binary + random_32bit, 2)
            test_video_ids.append(random_64bit)
            test_truths.append(False)

    headless = False
    request_delay = 1
    method = 'tiktokapi'
    api_method = 'video'
    
    if method == 'pytok' or (method == 'tiktokapi' and api_method == 'comment'):
        async with PyTok(headless=headless, request_delay=request_delay) as api:
            common_video = api.video(id=common_user_video_id, username=common_user)
            video_info = await common_video.info()
            # get a working comment api request
            async for comment in common_video.comments(count=1):
                pass
            if method == 'pytok':
                for i in range(len(test_video_ids)):
                    valid_video_id = test_video_ids[i]
                    if api_method == 'comment':
                        async for comment in api.video(id=valid_video_id).comments(count=1):
                            if comment:
                                print("Valid video id")
                    elif api_method == 'comment_reply':
                        data_request = common_video.parent.request_cache['comments']
                        url_parsed = url_parsers.urlparse(data_request.url)
                        params = url_parsers.parse_qs(url_parsed.query)
                        params['cursor'] = 0
                        del params['aweme_id']
                        params['count'] = 1
                        params['item_id'] = valid_video_id
                        params['comment_id'] = random_comment_id
                        params['focus_state'] = 'true'
                        url_path = url_parsed.path.replace("api/comment/list", "api/comment/list/reply")
                        next_url = f"{url_parsed.scheme}://{url_parsed.netloc}{url_path}?{url_parsers.urlencode(params, doseq=True)}"
                        cookies = await common_video.parent._context.cookies()
                        cookies = {cookie['name']: cookie['value'] for cookie in cookies}
                        r = requests.get(next_url, headers=data_request.headers, cookies=cookies)
                        res = r.json()
                        predicted_truth = 'comments' in res
                        ground_truth = test_truths[i]
                        if predicted_truth == ground_truth:
                            score += 1
                    elif api_method == 'related':
                        data_request = common_video.parent.request_cache['comments']
                        url_parsed = url_parsers.urlparse(data_request.url)
                        params = url_parsers.parse_qs(url_parsed.query)
                        params['itemID'] = valid_video_id
                        del params['aweme_id']
                        params['focus_state'] = 'true'
                        url_path = url_parsed.path.replace("api/comment/list", "api/related/item_list")
                        next_url = f"{url_parsed.scheme}://{url_parsed.netloc}{url_path}?{url_parsers.urlencode(params, doseq=True)}"
                        cookies = await common_video.parent._context.cookies()
                        cookies = {cookie['name']: cookie['value'] for cookie in cookies}
                        r = requests.get(next_url, headers=data_request.headers, cookies=cookies)
                        res = r.json()
                        predicted_truth = 'itemList' in res
                        ground_truth = test_truths[i]
                        if predicted_truth == ground_truth:
                            score += 1
                        else:
                            print(f"Failed for {valid_video_id}")
            else:
                working_request = common_video.parent.request_cache['comments']
                url_parsed = url_parsers.urlparse(working_request.url)
                working_params = url_parsers.parse_qs(url_parsed.query)
                working_headers = working_request.headers
                ms_token = os.environ.get("ms_token", None)

    if method == 'tiktokapi':
        if api_method != 'comment':
            ms_token = os.environ.get("ms_token", None)

        async with TikTokApi() as api:
            await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3)
            for i in range(len(test_video_ids)):
                valid_video_id = test_video_ids[i]
                if api_method == 'video':
                    video = api.video(url=f"https://www.tiktok.com/@therock/video/{valid_video_id}")
                    try:
                        video_data = await video.info()
                        predicted_truth = True
                    except NotFoundException:
                        predicted_truth = False
                    ground_truth = test_truths[i]
                    if predicted_truth == ground_truth:
                        score += 1
                    else:
                        print(f"Failed for {valid_video_id}")
                elif api_method == 'comment':
                    video = api.video(id=valid_video_id)
                    async for comment in video.comments(count=30, params={'verifyFp': working_params['verifyFp'], '_signature': working_params['_signature']}, headers=working_headers):
                        if comment:
                            print("Valid video id")
                elif api_method == 'comment_reply':
                    comment = api.comment(data={'cid': random_comment_id, 'text': None, 'digg_count': None, 'user': {'uid': valid_video_id, 'unique_id': None, 'sec_uid': None}})
                    params = {
                        "count": 20,
                        "cursor": 0,
                        "item_id": comment.author.user_id,
                        "comment_id": comment.id,
                    }

                    resp = await comment.parent.make_request(
                        url="https://www.tiktok.com/api/comment/list/reply/",
                        params=params,
                        headers=None,
                        session_index=None,
                    )

                    if resp:
                        print("Valid video id")
                elif api_method == 'related':
                    params = {
                        "itemID": valid_video_id,
                        "count": 16
                    }

                    resp = await api.make_request(
                        url="https://www.tiktok.com/api/related/item_list/",
                        params=params,
                        headers=None,
                        session_index=None,
                    )

                    predicted_truth = 'itemList' in resp
                    ground_truth = test_truths[i]
                    if predicted_truth == ground_truth:
                        score += 1
                    else:
                        print(f"Failed for {valid_video_id}")

    elif method == 'requests':
        for i in range(len(test_video_ids)):
            if api_method == 'video':
                valid_video_id = test_video_ids[i]
                r = requests.get(f"https://www.tiktok.com/@therock/video/{valid_video_id}")
                predicted_truth = r.status_code == 200
                ground_truth = test_truths[i]
                if predicted_truth == ground_truth:
                    score += 1
                else:
                    print(f"Failed for {valid_video_id}")

    print(f"Score: {score}/{len(test_video_ids)}")

if __name__ == "__main__":
    asyncio.run(main())