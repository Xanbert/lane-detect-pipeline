from kafka import KafkaProducer
import os
import cv2
import time

broker = os.getenv("BROKER")

producer = KafkaProducer(bootstrap_servers=[broker])

for picture in os.listdir("data"):
    img = cv2.imread(os.path.join("data", picture))
    producer.send("pics", img)
    time.sleep(0.5)
