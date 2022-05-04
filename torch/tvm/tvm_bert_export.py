from json import load
import warnings
import time
import numpy as np
import tvm
from tvm import relay
import tvm.contrib.graph_executor as runtime
import torch

import argparse



def load_model(model_name):

    PATH = f"../../base/{model_name}/"
    model = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장
    
    return model 


def convert_to_nhwc(mod):
    """Convert to NHWC layout"""
    desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
    seq = tvm.transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod


def compile_export(mod,params,target,batch_size):
    if target == "arm":
        target = tvm.target.arm_cpu()
    with tvm.transform.PassContext(opt_level=3):
        mod = relay.transform.InferType()(mod)
        lib = relay.build(mod, target=target, params=params)
    lib.export_library(f"./{model_name}_{batch_size}.tar")
    return lib 


def benchmark(model_name,seq_length,batch_size,target,dtype="float32",layout="NCHW"):
    input_name = "input0"
   
    inputs = np.random.randint(0, 2000, size=(seq_length))
    token_types = np.random.randint(0,2,size=(seq_length))

    tokens_tensor = torch.tensor(np.array([inputs]))
    segments_tensors = torch.tensor(np.array([token_types]))

    model = load_model(model_name)
    model.eval()

    traced_model = torch.jit.trace(model, tokens_tensor,segments_tensors)
   

    mod, params = relay.frontend.from_pytorch(traced_model, input_infos=[('input0', [batch_size,seq_length])],default_dtype=dtype)

    if layout == "NHWC":
        mod = convert_to_nhwc(mod)
    else:
        assert layout == "NCHW"

    lib=compile_export(mod,params,target,batch_size)
    print("export done :",f"{model_name}_{batch_size}.tar")

    dev = tvm.cpu()
    module = runtime.GraphModule(lib["default"](dev))

    module.set_input(data0=tokens_tensor,data1=segments_tensors)
    
    # Evaluate
    ftimer = module.module.time_evaluator("run", dev, min_repeat_ms=500, repeat=10)
    prof_res = np.array(ftimer().results) * 1000
    print(f"TVM {model_name} latency for batch {batch_size} : {np.mean(prof_res[1:]):.2f} ms")
 


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='bert' , type=str)
    parser.add_argument('--batchsize',default=1 , type=int)
    parser.add_argument('--seq_length',default=128 , type=int)
    parser.add_argument('--target',default='llvm -mcpu=core-avx2' , type=str)

    args = parser.parse_args()

    model_name = args.model
    batch_size = args.batchsize
    seq_length = args.seq_length
    target = args.target

    benchmark(model_name,seq_length,batch_size,target)
