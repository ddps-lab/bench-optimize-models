import torchvision.models as models
import torch.onnx
import onnx
import onnxoptimizer 
import numpy as np
import argparse


def convert(model_name,batchsize,size):
    PATH = f"../base/{model_name}/"
    model = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장

    # ------------------------ onnx export -----------------------------
    output_onnx = f'{model_name}_{batch_size}.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(batch_size, 3, size, size)

    torch_out = torch.onnx._export(model, inputs, output_onnx, export_params=True, verbose=False,
                                input_names=input_names, output_names=output_names)

def optimize_onnx(model_name,batch_size,skip_fuse_bn=False):
    opt_onnx = f'{model_name}_{batch_size}.opt.onnx'
    model_path = f"./{model_name}_{batch_size}.onnx"

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    onnx.helper.strip_doc_string(model)
    optimizers_list = onnxoptimizer.get_fuse_and_elimination_passes()
    if skip_fuse_bn:
        optimizers_list.remove('fuse_bn_into_conv')
    print(optimizers_list)
    model = onnxoptimizer.optimize(model, optimizers_list,
                                   fixed_point=True)
    onnx.checker.check_model(model)
    with open(opt_onnx, "wb") as f:
        f.write(model.SerializeToString())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str)
    parser.add_argument('--batchsize',default=1 , type=int)

    args = parser.parse_args()

    model_name = args.model
    batch_size = args.batchsize


    img_size=224
    if model_name == "inception_v3":
        img_size = 299 
    convert(model_name,batch_size,img_size)
    optimize_onnx(model_name,batch_size)
