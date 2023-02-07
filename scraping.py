import json
import re

import pandas as pd
from bs4 import BeautifulSoup
from requests_html import HTMLSession

session = HTMLSession()


def get_hotels(offset: int):
    url = "https://www.tripadvisor.com.vn/Restaurants-g293921-Vietnam.html"

    schema_json = f"offset={offset}"
    #
    headers = {
        "accept": "text/html, */*",
        "accept-language": "vi",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "x-requested-with": "XMLHttpRequest"
    }

    r = session.post(url, headers=headers, data=schema_json)

    soup = BeautifulSoup(r.text, 'lxml')
    result = soup.select_one('div[data-hotels-data]')['data-hotels-data']

    contents_json = json.loads(result)
    df = pd.json_normalize(contents_json['hotels'])[['id', 'name', 'numReviews']]
    return df.drop_duplicates().dropna()[df['numReviews'] != 0]


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
    result = soup.select_one('#component_2')['data-component-props']

    items = [item for item in soup.select('div[data-test*=list_item]') if item.attrs["data-test"] == "SL_list_item"]
    names = [re.sub('\d+\.\s+', '', item.select("a")[1].text, 1) for item in items]
    ids = [re.search('-d(\d+)', item.select("a")[1].attrs["href"])[1] for item in items]
    df = pd.DataFrame({"Id": ids, "Name": names})
    return df


def run(offset: int, header: bool = True, mode: str = 'w'):
    hotels = get_hotels(offset)
    hotels.to_csv('hotels.csv', encoding='utf-8', index=False, mode=mode, header=header)
    for id in hotels['id']:
        df = get_reviews(id)
        if df is not None:
            df.to_csv('hotel_reviews.csv', encoding='utf-8', index=False, mode=mode, header=header)

    if len(hotels) < 30:
        return False

    return True


if __name__ == "__main__":
    get_restaurants(293925, 0)
    # run(0)

    # offset = 30
    # try:
    #     while run(offset, header=False, mode='a'):
    #         offset += 30
    # except KeyboardInterrupt:
    #     print("end!")
