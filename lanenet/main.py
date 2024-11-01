import image_pb2
import image_pb2_grpc
import grpc
from concurrent import futures

import torch
import numpy as np
import cv2
from torchvision import transforms as T

from model.lanenet.LaneNet import LaneNet
from model.utils.cli_helper_test import parse_args

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = parse_args()
model_path = args.model
model = LaneNet(arch=args.model_type)
state_dict = torch.load(model_path,
                        map_location=torch.device('cpu'),
                        weights_only=False)
model.load_state_dict(state_dict)
model.eval()
model.to(DEVICE)


def process(img):
    return model(img)


class ImageRecv(image_pb2_grpc.ImageServiceServicer):

    def SendImage(self, request, context):
        pixels = np.frombuffer(request.pixels, dtype=np.float32)
        height, width = request.height, request.width
        pixels = pixels.reshape((height, width, 3))
        pixels = T.ToTensor()(pixels)
        pixels = torch.unsqueeze(pixels, dim=0)        
        outputs = process(pixels)
        print("Output")
        instance_pred = torch.squeeze(
            outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
        binary_pred = torch.squeeze(
            outputs['binary_seg_pred']).to('cpu').numpy() * 255
        cv2.imwrite('instance_output.jpg', instance_pred.transpose((1, 2, 0)))
        cv2.imwrite('binary_output.jpg', binary_pred)
        return image_pb2.Empty()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    image_pb2_grpc.add_ImageServiceServicer_to_server(ImageRecv(), server)
    server.add_insecure_port('0.0.0.0:50052')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
