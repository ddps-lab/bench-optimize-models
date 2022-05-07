from json import load
import warnings
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import nd, gluon
import time
import numpy as np

import argparse


ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()


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

def load_model(model_name):
    model_json = f"./{model_name}/model-symbol.json"
    model_params = f"./{model_name}/model-0000.params"


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gluon.nn.SymbolBlock.imports(model_json, ['data'], model_params, ctx=ctx)
    return model 

def benchmark(model_name,imgsize,batchsize):
    input_shape = (batchsize, 3, imgsize, imgsize)
    data = np.random.uniform(size=input_shape)

    input_data = mx.nd.array(data, ctx=ctx)

    model = load_model(model_name)

    res = timer(lambda: model(input_data).wait_to_read(),
                repeat=3,
                dryrun=5,
                min_repeat_ms=1000)
    print(f"MXNet {model_name} latency for batch {batchsize} : {np.mean(res):.2f} ms")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str)
    parser.add_argument('--batchsize',default=8,type=int)
    args = parser.parse_args()

    model_name = args.model
    batchsize = args.batchsize

    img_size = 224
    if args.model == "all":
        models = ["mobilenet", "mobilenet_v2", "inception_v3","resnet50","alexnet","vgg16","vgg19"]
    else:
        models = [args.model]

    for model in models:
        if model == 'inception_v3':
            img_size = 299
        benchmark(model,img_size,batchsize)
