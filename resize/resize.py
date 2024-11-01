from concurrent import futures
import image_pb2_grpc
import image_pb2
import grpc

import numpy as np
import cv2
import os

resize_width = 512
resize_height = 256
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

server = os.getenv("MAIN_SERVER")


def data_trans(image):
    image = cv2.resize(image, (resize_width, resize_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = image.astype(np.float32) / 255.0
    for i in range(3):  # 3
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
    return image


class ImageRecv(image_pb2_grpc.ImageServiceServicer):

    def SendImage(self, request, context):
        pixels = np.frombuffer(request.pixels, dtype=np.uint8)
        height, width = request.height, request.width
        pixels = pixels.reshape((height, width, 3))
        pixels = data_trans(pixels)
        height = resize_height
        width = resize_width
        pixels = pixels.flatten().tobytes()
        with grpc.insecure_channel(server) as channel:
            stub = image_pb2_grpc.ImageServiceStub(channel)
            response = stub.SendImage(
                image_pb2.Image(pixels=pixels, width=width, height=height))
        return image_pb2.Empty()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    image_pb2_grpc.add_ImageServiceServicer_to_server(ImageRecv(), server)
    server.add_insecure_port('0.0.0.0:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
