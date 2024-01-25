import os
import numpy as np
from PIL import Image
from torchvision import transforms
from ctypes import *
import urllib
import torch

def break_signal():
    so_file = "trace_signal.so"
    functions = CDLL(so_file)
    functions.break_signal()


def trace_resnet(model_size: int = 18):
    print(torch.__version__)
    print(f"[INFO] Started tracing Resnet-{model_size}")
    model_name = f"resnet{model_size}"
    model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    with torch.no_grad():
        output = model(input_batch)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print("[INFO] Inference completed")
    break_signal()

if __name__ == "__main__":
    # print("[INFO] Numpy information:")
    # np.show_config()
    # print("[INFO] Torch information:")
    print(*torch.__config__.show().split("\n"), sep="\n")
    trace_resnet()
