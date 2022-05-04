import os
import torch

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_bert import BertModel, BertForMaskedLM


import numpy as np 
from pathlib import Path
import time

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

def inference_model(model_name,batchsize,seq_length,dtype="float32"):

    inputs = np.random.randint(0, 2000, size=(seq_length))
    token_types = np.random.randint(0,2,size=(seq_length))


    tokens_tensor = torch.tensor(np.array([inputs]))
    segments_tensors = torch.tensor(np.array([token_types]))


    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    model(tokens_tensor, segments_tensors)    

    target_path = f"./{model_name}/"
    from pathlib import Path
    Path(target_path).mkdir(parents=True, exist_ok=True)

    torch.save(model, target_path + 'model.pt')  # 전체 모델 저장
    torch.save(model.state_dict(), target_path + 'model_state_dict.pt') 

    print("-"*10,f"Download and export {model_name} complete","-"*10)
    
    
    res = timer(lambda: model(tokens_tensor,segments_tensors),
                    repeat=3,
                    dryrun=5,
                    min_repeat_ms=1000)
    print(f"Pytorch {model_name} latency for batch {batchsize} : {np.mean(res):.2f} ms")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='bert' , type=str)
    parser.add_argument('--batchsize',default=1 , type=int)
    parser.add_argument('--seq_length',default=128 , type=int)


    args = parser.parse_args()

    seq_length= args.seq_length
    model_name = args.model
    batchsize = args.batchsize

    inference_model(model_name,batchsize,seq_length)
