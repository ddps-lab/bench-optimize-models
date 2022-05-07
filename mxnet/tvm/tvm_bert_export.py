import time
import argparse
import numpy as np
import mxnet as mx
import gluonnlp as nlp
import tvm
from tvm import relay
# import tvm.contrib.graph_runtime as runtime
import tvm.contrib.graph_executor as runtime



import tvm.testing
import warnings
from mxnet import gluon

warnings.filterwarnings(action='ignore') 

def load_model(model_name,batch_size):
    ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

    model_json = f"../base/{model_name}_{batch_size}/model-symbol.json"
    model_params = f"../base/{model_name}_{batch_size}/model-0000.params"

    if model_name == "bert_base":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = gluon.nn.SymbolBlock.imports(model_json, ['data0','data1','data2'], model_params, ctx=ctx)
    elif model_name == "distilbert":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = gluon.nn.SymbolBlock.imports(model_json, ['data0','data1'], model_params, ctx=ctx)
    return model 

def compile_tvm(model_name,batch_size,seq_length,target):

    # load origianl mxnet model 
    model = load_model(model_name,batch_size)

    # Prepare input data
    dtype = "float32"
    inputs = np.random.randint(0, 2000, size=(batch_size, seq_length)).astype(dtype)
    token_types = np.random.uniform(size=(batch_size, seq_length)).astype(dtype)
    valid_length = np.asarray([seq_length] * batch_size).astype(dtype)

    ######################################
    # Optimize the BERT model using TVM
    ######################################

    # First, Convert the MXNet model into TVM Relay format
    if model_name == "bert_base":
        shape_dict = {
            'data0': (batch_size, seq_length),
            'data1': (batch_size, seq_length),
            'data2': (batch_size,)
        }
    elif model_name=="distilbert":
        shape_dict = {
            'data0': (batch_size, seq_length),
            'data1': (batch_size,)
        }

    mod, params = relay.frontend.from_mxnet(model, shape_dict)

    # Compile the imported model
    if target == "arm":
        target = tvm.target.arm_cpu()

    with tvm.transform.PassContext(opt_level=3):
        mod = relay.transform.InferType()(mod)
        lib = relay.build(mod, target=target, params=params)
    lib.export_library(f"./{model_name}_{batch_size}.tar")

    dev = tvm.cpu()
    module = runtime.GraphModule(lib["default"](dev))

    if model_name == "bert_base":
        module.set_input(data0=inputs, data1=token_types, data2=valid_length)
    elif model_name == "distilbert":
        module.set_input(data0=inputs, data1=valid_length)

    # Evaluate
    ftimer = module.module.time_evaluator("run", dev, min_repeat_ms=500, repeat=5)
    prof_res = np.array(ftimer().results) * 1000
    print(f"TVM {model_name} latency for batch {batch_size} : {np.mean(prof_res):.2f} ms")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='bert_base' , type=str)
    parser.add_argument('--target',default="llvm -mcpu=core-avx2" , type=str)
    parser.add_argument('--batchsize',default=1 , type=int)
    parser.add_argument('--seq_length',default=128 , type=int)

    args = parser.parse_args()

    model_name = args.model
    target = args.target
    batch_size = args.batchsize
    seq_length = args.seq_length

    compile_tvm(model_name,batch_size,seq_length,target)
