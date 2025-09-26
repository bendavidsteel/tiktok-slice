import asyncio
import json
import os

import httpx
import polars as pl
from tqdm import tqdm

class ProcessVideo:
    def __init__(self, headers=None):
        self.text = ""
        self.headers = headers
        self.start = -1
        self.json_start = '"webapp.video-detail":'
        self.json_start_len = len(self.json_start)
        self.end = -1
        self.json_end = ',"webapp.a-b":'
    
    def process_chunk(self, text_chunk):
        self.text += text_chunk
        if len(self.text) < self.json_start_len:
            return 'continue'
        if self.start == -1:
            self.start = self.text.find(self.json_start)
            if self.start != -1:
                self.text = self.text[self.start + self.json_start_len:]
                self.start = 0
        if self.start != -1:
            self.end = self.text.find(self.json_end)
            if self.end != -1:
                self.text = self.text[:self.end]
                return 'break'
        return 'continue'
            
    def process_response(self):
        if self.start == -1 or self.end == -1:
            err_data = {'text': self.text}
            if self.headers:
                err_data['headers'] = self.headers
            raise Exception(
                "Could not find normal JSON section in returned HTML."
            )
        video_detail = json.loads(self.text)
        if video_detail.get("statusCode", 0) != 0: # assume 0 if not present
            # TODO retry when status indicates server error
            return video_detail
        video_info = video_detail.get("itemInfo", {}).get("itemStruct")
        if video_info is None:
            raise Exception(
                video_detail, "TikTok JSON did not contain expected JSON."
            )
        return video_info

def get_headers():
    # TODO different user agent results in different html encoding, need to update process video class for each user agent
    # software_names = [rug_params.SoftwareName.CHROME.value, rug_params.SoftwareName.FIREFOX.value]
    # operating_systems = [rug_params.OperatingSystem.WINDOWS.value, rug_params.OperatingSystem.ANDROID.value, rug_params.OperatingSystem.IOS.value, rug_params.OperatingSystem.MAC_OS_X.value]   
    
    # user_agent_rotator = rug_user_agent.UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)

    # # Get Random User Agent String.
    # user_agent = user_agent_rotator.get_random_user_agent()
    
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-CA',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
    }
    return headers

async def main():
    df = pl.read_excel('./data/annotations-tk-20250626-from-results.xlsx')
    
    df = df.filter(pl.col('choice').is_in(['Partial GenAI', 'Not GenAI', 'GenAI']))

    os.makedirs('./data/aif_aigc', exist_ok=True)

    headers = get_headers()

    for post in tqdm(df.to_dicts()):
        file_path = f"./data/aif_aigc/{post['id']}.mp4"
        if os.path.exists(file_path):
            continue

        url = f"https://www.tiktok.com/@{post['author']}/video/{post['id']}"

        async with httpx.AsyncClient() as client:
            info_res = await client.get(url, headers=headers)
            if info_res.status_code != 200:
                continue
            text_chunk = info_res.text
            video_processor = ProcessVideo()
            do = video_processor.process_chunk(text_chunk)

            bytes_headers = {
                'sec-ch-ua': '"HeadlessChrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"', 
                'referer': 'https://www.tiktok.com/', 
                'accept-encoding': 'identity;q=1, *;q=0', 
                'sec-ch-ua-mobile': '?0', 
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.4 Safari/537.36', 
                'range': 'bytes=0-', 
                'sec-ch-ua-platform': '"Windows"'
            }

            video_d = video_processor.process_response()

            if 'video' not in video_d:
                continue

            download_url = video_d['video']['downloadAddr'] if video_d['video'].get('downloadAddr') else video_d['video'].get('playAddr')

            if not download_url:
                continue

            cookies = {c: info_res.cookies[c] for c in info_res.cookies}
            try:
                bytes_res = await client.get(download_url, headers=bytes_headers, cookies=cookies)
            except Exception as e:
                print(f"Error fetching video bytes for {post['id']}: {e}")
                continue
            if 200 <= bytes_res.status_code >= 300:
                continue
            content = bytes_res.content

            with open(file_path, 'wb') as f:
                f.write(content)

if __name__ == "__main__":
    asyncio.run(main())