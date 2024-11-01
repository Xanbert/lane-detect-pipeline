import grpc
import image_pb2
import image_pb2_grpc
import numpy as np


def denoise(img):
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    return denoised_img


def main():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = image_pb2_grpc.ImageServiceStub(channel)
        image = cv2.imread(\"test_input.jpg\")
        image = denoise(image)
        height, width, _ = image.shape
        pixels = image.flatten().astype(np.double).tobytes()
        response = stub.SendImage(
            image_pb2.Image(pixels=pixels, width=width, height=height))
        print(\"Image sent successfully.\")


if __name__ == '__main__':
    main()
