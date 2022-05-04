import numpy as np
import time 
import argparse
import tvm.contrib.graph_executor as runtime

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te

def benchmark(model_name, img_size,batch_size, repeat=3):
    ctx = tvm.cpu()

    input_name = "input0"
    input_shape = (batch_size, 3, img_size, img_size)
    output_shape = (batch_size, 1000)
    
    loaded_lib = tvm.runtime.load_module(f'./{model_name}_{batch_size}.tar')

    module = runtime.GraphModule(loaded_lib["default"](ctx))

    # Feed input data
    data = np.random.uniform(size=input_shape)
    module.set_input(input_name, data)

    # Evaluate
    ftimer = module.module.time_evaluator("run", ctx, min_repeat_ms=500, repeat=repeat)
    prof_res = np.array(ftimer().results) * 1000
    print(f"TVM {model_name} latency for batch {batch_size} : {np.mean(prof_res):.2f} ms")
    
    return np.array(ftimer().results)

    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str)
    parser.add_argument('--batchsize',default=1 , type=int)

    args = parser.parse_args()

    model_name = args.model
    batch_size = args.batchsize
    img_size = 224

   
    if model_name == 'inception_v3':
        img_size = 299
    benchmark(model_name,img_size,batch_size)
