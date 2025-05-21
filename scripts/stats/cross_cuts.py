import os

import numpy as np
import polars as pl
from scipy import stats
from tqdm import tqdm

country_codes = [
    "AF",  # Afghanistan
    "DZ",  # Algeria
    "AO",  # Angola
    "AG",  # Antigua and Barbuda
    "AR",  # Argentina
    "BS",  # Bahamas
    "BH",  # Bahrain
    "BD",  # Bangladesh
    "BB",  # Barbados
    "BZ",  # Belize
    "BJ",  # Benin
    "BT",  # Bhutan
    "BO",  # Bolivia
    "BA",  # Bosnia and Herzegovina
    "BW",  # Botswana
    "BR",  # Brazil
    "BN",  # Brunei Darussalam
    "BF",  # Burkina Faso
    "BI",  # Burundi
    "CV",  # Cabo Verde/Cape Verde
    "KH",  # Cambodia
    "CF",  # Central African Republic
    "TD",  # Chad
    "CL",  # Chile
    "CN",  # China
    "CO",  # Colombia
    "KM",  # Comoros
    "CD",  # Congo (Dem. Rep.)
    "CG",  # Congo (Republic of)
    "CR",  # Costa Rica
    "CI",  # Côte d'Ivoire/Ivory Coast
    "CU",  # Cuba
    "DJ",  # Djibouti
    "DM",  # Dominica
    "DO",  # Dominican Republic
    "EC",  # Ecuador
    "EG",  # Egypt
    "SV",  # El Salvador
    "GQ",  # Equatorial Guinea
    "ER",  # Eritrea
    "SZ",  # Eswatini
    "ET",  # Ethiopia
    "FJ",  # Fiji
    "GA",  # Gabon
    "GM",  # Gambia
    "GH",  # Ghana
    "GD",  # Grenada
    "GT",  # Guatemala
    "GN",  # Guinea
    "GW",  # Guinea-Bissau
    "GY",  # Guyana
    "HT",  # Haiti
    "HN",  # Honduras
    "IN",  # India
    "ID",  # Indonesia
    "IR",  # Iran
    "IQ",  # Iraq
    "JM",  # Jamaica
    "JO",  # Jordan
    "KE",  # Kenya
    "KI",  # Kiribati
    "LA",  # Laos
    "LB",  # Lebanon
    "LS",  # Lesotho
    "LR",  # Liberia
    "LY",  # Libya
    "MG",  # Madagascar
    "MW",  # Malawi
    "MY",  # Malaysia
    "MV",  # Maldives
    "ML",  # Mali
    "MH",  # Marshall Islands
    "MR",  # Mauritania
    "MU",  # Mauritius
    "MX",  # Mexico
    "FM",  # Micronesia
    "MN",  # Mongolia
    "MA",  # Morocco
    "MZ",  # Mozambique
    "MM",  # Myanmar
    "NA",  # Namibia
    "NR",  # Nauru
    "NP",  # Nepal
    "NI",  # Nicaragua
    "NE",  # Niger
    "NG",  # Nigeria
    "KP",  # North Korea
    "OM",  # Oman
    "PK",  # Pakistan
    "PS",  # Palestine
    "PA",  # Panama
    "PG",  # Papua New Guinea
    "PY",  # Paraguay
    "PE",  # Peru
    "PH",  # Philippines
    "RW",  # Rwanda
    "KN",  # Saint Kitts and Nevis
    "LC",  # Saint Lucia
    "VC",  # Saint Vincent and the Grenadines
    "WS",  # Samoa
    "ST",  # São Tomé and Príncipe
    "SA",  # Saudi Arabia
    "SN",  # Senegal
    "SC",  # Seychelles
    "SL",  # Sierra Leone
    "SB",  # Solomon Islands
    "SO",  # Somalia
    "ZA",  # South Africa
    "SS",  # South Sudan
    "LK",  # Sri Lanka
    "SD",  # Sudan
    "SR",  # Suriname
    "SY",  # Syria
    "TJ",  # Tajikistan
    "TZ",  # Tanzania
    "TH",  # Thailand
    "TL",  # Timor-Leste
    "TG",  # Togo
    "TO",  # Tonga
    "TT",  # Trinidad and Tobago
    "TN",  # Tunisia
    "TR",  # Turkey
    "TM",  # Turkmenistan
    "UG",  # Uganda
    "UY",  # Uruguay
    "VU",  # Vanuatu
    "VE",  # Venezuela
    "VN",  # Vietnam
    "YE",  # Yemen
    "ZM",  # Zambia
    "ZW"   # Zimbabwe
]

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(this_dir_path, "..", '..', "data", 'topic_model_videos')

    hour_video_df = pl.DataFrame()
    day_video_df = pl.DataFrame()
    video_dir_path = os.path.join('.', 'data', 'results', '2024_04_10', 'hours')
    video_pbar = tqdm(total=60*60 + 23 * 60, desc='Reading videos')
    for root, dirs, files in os.walk(video_dir_path):
        for file in files:
            if file == 'videos.parquet.zstd':
                video_pbar.update(1)
                root_sections = root.split('/')
                hour, minute, second = root_sections[-3], root_sections[-2], root_sections[-1]
                result_path = os.path.join(root, file)
                batch_video_df = pl.read_parquet(result_path)
                batch_video_df = batch_video_df.select([
                    pl.col('video_id'),
                    pl.col('authorVerified'),
                    pl.col('musicOriginal'),
                    pl.col('videoDuration'),
                    pl.col('videoQuality'),
                    pl.col('locationCreated'),
                    pl.col('desc'),
                    pl.col('shareCount'),
                    pl.col('diggCount'),
                    pl.col('commentCount'),
                    pl.col('playCount'),
                    pl.col('diversificationLabels')
                ])
                if minute == '42':
                    day_video_df = pl.concat([day_video_df, batch_video_df], how='diagonal_relaxed')
                if hour == '19':
                    hour_video_df = pl.concat([hour_video_df, batch_video_df], how='diagonal_relaxed')

    # load topics data
    topic_desc_df = pl.read_parquet('./data/topic_model_videos/topic_desc.parquet.gzip')
    post_topic_df = pl.read_parquet('./data/topic_model_videos/video_topics.parquet.gzip')
    post_topic_df = post_topic_df.with_columns(pl.col('image_path').str.split('/').list.get(-1).str.split('.').list.get(0).alias('video_id'))
    hour_video_df = hour_video_df.join(post_topic_df, on='video_id', how='left')
    hour_video_df = hour_video_df.unique('video_id')

    vid_view_mean = hour_video_df['playCount'].mean()
    daily_view_mean = day_video_df['playCount'].mean()
    print(f"Daily view mean: {daily_view_mean}")

    # Country statistics with t-test
    countries = ['UA']
    for country in countries:
        print(country)
        country_df = day_video_df.filter(pl.col('locationCreated') == country)
        country_mean = country_df['playCount'].mean()
        print(f"Country play mean: {country_mean}")
        
        # t-test comparing country mean to global mean
        t_stat, p_value = stats.ttest_ind(
            country_df.drop_nans('playCount')['playCount'].to_numpy(),
            day_video_df.drop_nans('playCount')['playCount'].to_numpy(),
            equal_var=False  # Using Welch's t-test, not assuming equal variance
        )
        print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.5f}")
        print(f"Statistically {'different' if p_value < 0.05 else 'not different'} from global mean (α=0.05)")

    print(f"Hour view mean: {vid_view_mean}")

    # Topic analysis with statistical significance
    topic_keywords = ['palestin', 'militar', 'medica', 'police operations']
    for topic_keyword in topic_keywords:
        topic_d = topic_desc_df.row(by_predicate=pl.col('Desc').str.to_lowercase().str.contains(topic_keyword))
        topic_idx = topic_d[0]
        topic_post_df = hour_video_df.filter(pl.col('topic') == topic_idx)
        pct_topic = 100 * topic_post_df.shape[0] / hour_video_df.shape[0]
        print(topic_d[-1])
        print(f"Pct: {pct_topic}, count: {topic_post_df.shape[0]}")
        
        # Calculate topic mean
        topic_view_mean = topic_post_df['playCount'].mean()
        print(f"Topic view mean: {topic_view_mean}")
        
        # t-test comparing topic mean to global mean
        if topic_post_df.shape[0] > 1:  # Ensure we have at least 2 samples for the t-test
            t_stat, p_value = stats.ttest_ind(
                topic_post_df.drop_nans('playCount')['playCount'].to_numpy(),
                hour_video_df.drop_nans('playCount')['playCount'].to_numpy(),
                equal_var=False  # Using Welch's t-test
            )
            print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.5f}")
            print(f"Statistically {'different' if p_value < 0.05 else 'not different'} from global mean (α=0.05)")

        
    # global south
    global_south_df = day_video_df.filter(pl.col('locationCreated').is_in(country_codes))
    g_s_view_mean = global_south_df['playCount'].mean()
    print(f"Global south percentage: {global_south_df.shape[0] * 100 / day_video_df.shape[0]}%")
    print(f"Global south view mean: {g_s_view_mean}")

if __name__ == '__main__':
    main()