import findspark
from my_model import BiLSTM_CNN
from utils import clean_doc
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from settings import CATERGORIES, CHECKPOINT_PATH, INPUT_TOKERNIZER, KAFKA_SERVERS, LABELS, MAX_LEN, TOPIC_NAME
from keras.utils import pad_sequences
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
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
        .appName("kafka-example")\
        .config("spark.jars.packages", ",".join(packages))\
        .getOrCreate()
# Reduce logging
sc = spark.sparkContext 

sc.setLogLevel("WARN")


model = BiLSTM_CNN()
model.build((None, MAX_LEN))
optimizer = tfa.optimizers.RectifiedAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-5)
list_loss = ['categorical_crossentropy' for _ in range(len(CATERGORIES))]
model.compile(loss = list_loss, optimizer=optimizer, metrics=['accuracy'])
model.load_weights(CHECKPOINT_PATH)

model_bc = sc.broadcast(model)
tokenizer_bc = sc.broadcast(INPUT_TOKERNIZER)
settings_bc = sc.broadcast({
        'MAX_LEN':MAX_LEN,
        'CATERGORIES':CATERGORIES,
        'LABELS':LABELS
})

def get_result(text:str):
    text = clean_doc(text)
    text = tokenizer_bc.value.texts_to_sequences([text])
    text = pad_sequences(text, settings_bc.value['MAX_LEN'], padding="post").tolist()
    text = tf.convert_to_tensor(text)
    #predict
    predict = model_bc.value.predict(text, verbose=0)[0]
    predict = ['{'+ f'{settings_bc.value["CATERGORIES"][id]}, {settings_bc.value["LABELS"][np.argmax(label)]}' +'}'
                     for id, label in enumerate(predict) if np.argmax(label)]
    
    predict = "{" + ", ".join(predict) +"}"
    print(predict)
    return predict

udf_get_result = F.udf(get_result,StringType())

kafkaDf = spark.readStream.format("kafka")\
    .option("kafka.bootstrap.servers", KAFKA_SERVERS)\
    .option("subscribe", TOPIC_NAME)\
    .option("startingOffsets", "earliest")\
    .load()

query = kafkaDf\
    .select(
        F.col('key').cast('string').alias('id'),
        F.col('value').cast('string').alias('text'), 
        udf_get_result(F.col('value').cast('string').alias('predict'))
    )

query.writeStream.outputMode("append").format("kafka")\
    .option("kafka.bootstrap.servers", KAFKA_SERVERS)\
    .option("subscribe", "result")\
    .start()\
    .awaitTermination(600)
    

