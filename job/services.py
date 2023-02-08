import time
from settings import KAFKA_SERVERS, NUM_PARTITIONS, REPLICATION_FACTOR, TOPIC_NAME
from kafka import KafkaAdminClient
from kafka.errors import TopicAlreadyExistsError
from kafka.admin.new_topic import NewTopic

def create_topic(topic_name:str):
    client = KafkaAdminClient(bootstrap_servers=KAFKA_SERVERS)

    try:
        client.create_topics([
            NewTopic(
                name=topic_name,
                num_partitions=NUM_PARTITIONS,
                replication_factor=REPLICATION_FACTOR,
            )
        ])
        
        print("Topic created!")
    except TopicAlreadyExistsError:
        print("Topic exists!")
        
create_topic(TOPIC_NAME)
create_topic("result")


import os
cmd = 'start cmd /k python'
os.system(f"{cmd} scraping.py 293925")
time.sleep(5)
os.system(f"{cmd} consumer.py")



