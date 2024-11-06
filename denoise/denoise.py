from kafka import KafkaConsumer

import os

import grpc
import image_pb2
import image_pb2_grpc
import numpy as np

broker = os.getenv("BROKER")


def denoise(img):
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    return denoised_img


def main():
    consumer = KafkaConsumer('pics', bootstrap_servers=[broker])
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = image_pb2_grpc.ImageServiceStub(channel)
        for msg in consumer:
            image = msg.value
            image = denoise(image)
            height, width, _ = image.shape
            pixels = image.flatten().astype(np.uint8).tobytes()
            response = stub.SendImage(
                image_pb2.Image(pixels=pixels, width=width, height=height))
            print("Image sent successfully.")


if __name__ == '__main__':
    main()
