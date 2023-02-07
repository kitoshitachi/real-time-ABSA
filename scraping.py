import json
import re
import sys
from langdetect import detect
import pandas as pd
from bs4 import BeautifulSoup
from requests_html import HTMLSession

session = HTMLSession()

def get_reviews(location_id: int):
    schema_json = [{
        "query": "ea9aad8c98a6b21ee6d510fb765a6522",
        "variables": {
            "locationId": int(location_id),
            "offset": 0,
            "limit": 9999,
            "filters": [{"axis": "LANGUAGE", "selections": ["vi"]}],
            "prefs": None,
            "initialPrefs":{},
            "filterCacheKey": f"locationReviewFilters_d{location_id}",
            "prefsCacheKey": f"locationReviewPrefs_d{location_id}",
            "needKeywords": False,
            "keywordVariant": "location_keywords_v2_llr_order_30_vi"
        }
    }]

    headers = {
        "accept": "*/*",
        "accept-language": "vi;q=0.9,en;q=0.8",
        "cache-control": "no-cache",
        "content-type": "application/json",
        "x-requested-by": "kito",
        "Access-Control-Allow-Origin": "true",
        "Origin": "https://www.tripadvisor.com.vn"
    }

    r = session.post("https://www.tripadvisor.com.vn/data/graphql/ids", headers=headers, json=schema_json)

    content_json = json.loads(r.text)[0]['data']['locations'][0]['reviewListPage']['reviews']
    try:
        # , 'userProfile',['text','rating','title',['userId','username']]) \
        df = pd.json_normalize(content_json)[[
            'id', 'text',
            'createdDate',
            'userProfile.userId', 'userProfile.username'
        ]]
        df.columns = ['id', 'text', 'createdDate', 'userId', 'username']
        df['text'] = df['text'].apply(lambda x: x.splitlines())
        df = df.explode('text')
        return df.dropna().drop_duplicates()
    except KeyError:
        print(f'id {location_id} error')


def get_restaurants(city_id: int, offset: int):
    r = session.get(
        f"https://www.tripadvisor.com.vn/RestaurantSearch?Action=PAGE&ajax=1&availSearchEnabled=false&sortOrder=popularity&geo={city_id}&itags=10591&o=a{offset}")
    soup = BeautifulSoup(r.text, 'lxml')
    items = [item for item in soup.select('div[data-test*=list_item]') if item.attrs["data-test"] != "SL_list_item"]
    names = [re.sub('\d+\.\s+', '', item.select("a")[1].text, 1) for item in items]
    ids = [re.search('-d(\d+)', item.select("a")[1].attrs["href"])[1] for item in items]
    df = pd.DataFrame({"id": ids, "Name": names})
    return df


def run(city_id: int, offset: int, header: bool = True, mode: str = 'w'):
    restaurants = get_restaurants(city_id, offset)
    restaurants.to_csv(f'data/{city_id}_restaurants.csv', encoding='utf-8', index=False, mode=mode, header=header)
    for id in restaurants['id']:
        df = get_reviews(id)
        
        if df is not None:
            df.to_csv(f'data/{city_id}_restaurant_reviews.csv', encoding='utf-8', index=False, mode=mode, header=header)

    if len(restaurants) < 30:
        return False

    return True


if __name__ == "__main__":
    city_id = sys.argv[1]
    run(city_id, 0)

    offset = 30
    try:
        while run(city_id, offset, header=False, mode='a'):
            offset += 30
    except KeyboardInterrupt:
        print("end!")

