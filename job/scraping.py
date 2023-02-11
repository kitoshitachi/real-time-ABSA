import gc
import re, sys, json
from time import sleep
import pandas as pd
from bs4 import BeautifulSoup
from requests_html import HTMLSession
import findspark
findspark.init()
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from utils import clean_doc
from settings import CATERGORIES, INPUT_TOKERNIZER, KAFKA_SERVERS, LABELS, MAX_LEN, TOPIC_NAME
from keras.utils import pad_sequences
import tensorflow as tf
from pyspark.sql.types import StructType,StructField, StringType, ArrayType, IntegerType

scala_version = '2.13'
spark_version = '3.3.1'
packages = [
    f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
     'org.apache.kafka:kafka-clients:3.3.1'
]

spark = SparkSession.builder.master("local")\
        .appName("ABSA")\
        .config("spark.jars.packages", ",".join(packages))\
        .getOrCreate()

# Reduce logging
spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled","true")

sc = spark.sparkContext 
tokenizer_bc = sc.broadcast(INPUT_TOKERNIZER)
settings_bc = sc.broadcast({
        'MAX_LEN':MAX_LEN,
        'CATERGORIES':CATERGORIES,
        'LABELS':LABELS
})

# ['review_id', 'createdDate', 'userId', 'username', 'text']
mySchema = StructType([ 
    StructField("review_id", StringType(), True),
    StructField("createdDate", StringType(), True),
    StructField("userId", StringType(), True),
    StructField("username", StringType(), True),
    StructField("text", StringType(), True),
    StructField("vector", ArrayType(IntegerType()), True),
])

sc.setLogLevel("OFF")

session = HTMLSession()

def pipeline(text:str):
    vector = clean_doc(text)
    vector = tokenizer_bc.value.texts_to_sequences([vector])
    vector = pad_sequences(vector, settings_bc.value['MAX_LEN'], padding="post").tolist()
    return vector[0]

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
        del content_json, r, schema_json, headers
        df.columns = ['review_id', 'text', 'createdDate', 'userId', 'username']
        df['text'] = df['text'].apply(lambda x: x.splitlines())
        df = df.explode('text').dropna().drop_duplicates()
        df = df[df['text'].str.match(r'[a-zA-Z1-9]') == True].reset_index()
        df['text'] = df["text"].apply(lambda x: x.split('.'))
        df = df.explode('text').dropna().drop_duplicates()
        df = df[df['text'].str.match(r'[a-zA-Z1-9]') == True].reset_index()
        df['text'] = df['text'].str.strip()
        df['count'] = df.groupby('review_id').cumcount() + 1
        df['review_id'] = df['count'].astype(str) + df["review_id"].astype(str)
        return df
    except KeyError:
        print(f'id {location_id} error')


def get_restaurants(city_id: int, offset: int):
    r = session.get(
        f"https://www.tripadvisor.com.vn/RestaurantSearch?Action=PAGE&ajax=1&availSearchEnabled=false&sortOrder=popularity&geo={city_id}&itags=10591&o=a{offset}")
    soup = BeautifulSoup(r.text, 'lxml')
    del r
    items = [item for item in soup.select('div[data-test*=list_item]') if item.attrs["data-test"] != "SL_list_item"]
    del soup
    names = [re.sub('\d+\.\s+', '', item.select("a")[1].text, 1) for item in items]
    ids = [re.search('-d(\d+)', item.select("a")[1].attrs["href"])[1] for item in items]
    del items
    df = pd.DataFrame({"restaurant_id": ids, "Name": names})
    del names, ids
    df.insert(0,'city_id',city_id)
    return df


def run(city_id: int, offset: int, header: bool = True, mode: str = 'w'):
    restaurants = get_restaurants(city_id, offset)
    restaurants.to_csv(f'data/{city_id}_restaurants.csv', encoding='utf-8', index=False, mode=mode, header=header)
    for id in restaurants['restaurant_id']:
        df = get_reviews(id)
        df.insert(0,'restaurant_id',id)
        
        if df is not None:
            df = df[['review_id', 'createdDate', 'userId', 'username', 'text']]
            df.to_csv(
                f'data/{city_id}_restaurant_reviews.csv', encoding='utf-8', 
                index=False, mode=mode, header=header
            )
            df['vector'] = df['text'].apply(pipeline)
            sdf:DataFrame = spark.createDataFrame(df, schema=mySchema)
            
            query = sdf.select(
                    F.col('review_id').cast("string").alias("key"),
                    # F.col('text').cast("string"),
                    F.array_join(F.col('vector'),",").cast("string").alias("value")
                )
            query.write.format("kafka") \
                .option("kafka.bootstrap.servers",KAFKA_SERVERS)\
                .option("topic", TOPIC_NAME).save()
            # print(query.show(truncate=True))
            print(f'successfully save data of {id}')
            sleep(10)

    if len(restaurants) < 30:
        return False

    return True


if __name__ == "__main__":
    city_id = sys.argv[1]
    running = run(city_id, 0)
    sleep(10)
    offset = 30
    try:
        while running:
            running = run(city_id, offset, header=False, mode='a')
            offset += 30
            gc.collect()
            sleep(10)
    except KeyboardInterrupt:
        print("end!")

