import json
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
import numpy as np


ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

def download_model(model_name,batchsize,imgsize=224):
    models_detail = {
            'densenet' : vision.densenet161(pretrained=True, ctx=ctx),
        'resnet18' : vision.resnet18_v1(pretrained=True, ctx=ctx),
        'squeezenet' : vision.squeezenet1_0(pretrained=True, ctx=ctx),
        'mobilenet':vision.mobilenet0_5(pretrained=True, ctx=ctx),
        'mobilenet_v2':vision.get_mobilenet_v2(1, pretrained=True),
        'inception_v3':vision.inception_v3(pretrained=True, ctx=ctx),
        'resnet50': vision.get_resnet(1, 50, pretrained=True),
        'alexnet':vision.alexnet(pretrained=True,ctx=ctx),
        'vgg16':vision.vgg16(pretrained=True, ctx=ctx),
        'vgg19':vision.vgg19(pretrained=True, ctx=ctx)
    }

    model = models_detail[model_name]
    model.hybridize()

    input_shape = (batchsize, 3, imgsize, imgsize)
    data = np.random.uniform(size=input_shape)

    data_array = np.random.uniform(0, 255, size=input_shape).astype("float32")
    mx_data = mx.nd.array(data_array)
    model(mx_data)
    
    target_path = f"./{model_name}"
    from pathlib import Path
    Path(target_path).mkdir(parents=True, exist_ok=True)

    model.export(f'{model_name}/model')
    print("-"*10,f"Download and export {model_name} complete","-"*10)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str)
    parser.add_argument('--batchsize',default=8 , type=int)


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
        download_model(model,batchsize,img_size)
