from json import load
import warnings
import torch
import time
import numpy as np

import argparse


def timer(thunk, repeat=1, number=10, dryrun=3, min_repeat_ms=1000):
    """Helper function to time a function"""
    for i in range(dryrun):
        thunk()
    ret = []
    for _ in range(repeat):
        while True:
            beg = time.time()
            for _ in range(number):
                thunk()
            end = time.time()
            lat = (end - beg) * 1e3
            if lat >= min_repeat_ms:
                break
            number = int(max(min_repeat_ms / (lat / number) + 1, number * 1.618))
        ret.append(lat / number)
    return ret

def load_model(model_name,batch_size):
    
    PATH = f"./{model_name}_{batch_size}/"
    model = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장


    return model 

def benchmark(model_name,batchsize,imgsize):
    input_shape = (batchsize, 3, imgsize, imgsize)
    data_array = np.random.uniform(0, 255, size=input_shape).astype("float32")
    torch_data = torch.tensor(data_array)

    model = load_model(model_name,batchsize)
    model.eval()

    res = timer(lambda: model(torch_data),
                repeat=3,
                dryrun=5,
                min_repeat_ms=1000)
    print(f"Pytorch {model_name} latency for batch {batchsize} : {np.mean(res):.2f} ms")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str)
    parser.add_argument('--batchsize',default=1 , type=int)

    args = parser.parse_args()

    model_name = args.model
    batchsize = args.batchsize
  
    img_size = 224
    
    if model_name == 'inception_v3':
        img_size = 299
    benchmark(model_name,batchsize,img_size)
