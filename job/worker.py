from typing import List
import findspark
from my_model import BiLSTM_CNN
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from settings import CATERGORIES, CHECKPOINT_PATH, KAFKA_SERVERS, LABELS, MAX_LEN, TOPIC_NAME
from pyspark.sql.types import StructType,StructField, StringType, ArrayType, IntegerType
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

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
sc = spark.sparkContext 

sc.setLogLevel("OFF")


model = BiLSTM_CNN()
model.build((None, MAX_LEN))
optimizer = tfa.optimizers.RectifiedAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-5)
list_loss = ['categorical_crossentropy' for _ in range(len(CATERGORIES))]
model.compile(loss = list_loss, optimizer=optimizer, metrics=['accuracy'])
model.load_weights(CHECKPOINT_PATH).expect_partial()

model_bc = sc.broadcast(model)
# tokenizer_bc = sc.broadcast(INPUT_TOKERNIZER)
settings_bc = sc.broadcast({
        'MAX_LEN':MAX_LEN,
        'CATERGORIES':CATERGORIES,
        'LABELS':LABELS
})

def get_result(vector:List[int]):
    # text = clean_doc(text)
    # origin = text
    # text = tokenizer_bc.value.texts_to_sequences([text])
    # text = pad_sequences(text, settings_bc.value['MAX_LEN'], padding="post").tolist()
    vector = tf.convert_to_tensor([vector])
    #predict
    predict = model_bc.value.predict(vector, verbose=0)[0]
    predict = ['{'+ f'{settings_bc.value["CATERGORIES"][id]}, {settings_bc.value["LABELS"][np.argmax(label)]}' +'}'
                     for id, label in enumerate(predict) if np.argmax(label)]
    
    predict = "{" + ", ".join(predict) +"}"
    # print(f'\n{predict}\n =====================\n')
    return predict

udf_get_result = F.udf(get_result,StringType())

kafkaDf = spark.read.format("kafka")\
    .option("kafka.bootstrap.servers", KAFKA_SERVERS)\
    .option("subscribe", TOPIC_NAME)\
    .option("startingOffsets", "earliest")\
    .load()

query = kafkaDf\
    .select(
        F.col('key').cast('string').alias('review_id'),
        F.split(F.col('value').cast('string'),",").cast(ArrayType(IntegerType())).alias('vector'),
        # udf_get_result(F.col('value').cast('string')).alias('predict')
    )#.printSchema()

query1 = query.select(
    F.col('review_id'),
    F.col('vector'),
    F.size(F.col('vector')).alias('max_len'),
    udf_get_result(F.col('vector')).alias('predict')
)#.printSchema()
# query2 = query1.filter(F.col('predict') != "{\}")

from IPython.display import display
display(query1.toPandas())
#streaming sink to console 
# streaming = query2.writeStream.outputMode('append').format('console')\
    # .start()

# streaming.awaitTermination(60000)


#streaming sink to kafka
# query.writeStream.format("kafka")\
#     .option("kafka.bootstrap.servers", KAFKA_SERVERS)\
#     .option("topic", "result")\
#     .option("checkpointLocation", "D:/19521204/python/realtime_ABSA/job/checkpoint_streaming")\
#     .start()\
#     .awaitTermination(60000)
    