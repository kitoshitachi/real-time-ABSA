import json,re
from requests_html import HTMLSession
import pandas as pd
session = HTMLSession()

def get_content(url):
    location_id = re.findall(r'-d\d*-',url)[0].replace('-','').replace('d','')
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
    #, 'userProfile',['text','rating','title',['userId','username']]) \
    df = pd.json_normalize(content_json)[[
        'id','text',
        'createdDate',
        'userProfile.userId','userProfile.username'
    ]]
    df.columns = ['id','text','createdDate','userId','username']
    df['text'] = df['text'].apply(lambda x : x.splitlines())
    df = df.explode('text')
    return df.dropna().drop_duplicates()

if __name__ == "__main__":
    url = "https://www.tripadvisor.com.vn/Hotel_Review-g298085-d16891462-"
    print(get_content(url).info())