import os
import torch
import torchvision.models as models
import numpy as np 
from pathlib import Path

def download_model(model_name,imgsize=224):
    models_detail = {
        'efficientnet_b0' : models.efficientnet_b0(pretrained=True),
        'mobilenet_v2':models.mobilenet_v2(pretrained=True),
        'resnet18' : models.resnet18(pretrained=True),
        'mobilenet_v3_small' : models.mobilenet_v3_small(pretrained=True),
        'mnasnet':models.mnasnet1_0(pretrained=True),
        'shufflenet' : models.shufflenet_v2_x1_0(pretrained=True),
        'squeezenet' :models.squeezenet1_0(pretrained=True),
        
        'inception_v3':models.inception_v3(pretrained=True),
        'resnet50': models.resnet50(pretrained=True),
        'alexnet':models.alexnet(pretrained=True),
        'vgg16':models.vgg16(pretrained=True),
        'vgg19':models.vgg19(pretrained=True),
    }

    model = models_detail[model_name]
    model.eval()
    # input_shape = (batchsize, 3, imgsize, imgsize)
    # data = np.random.uniform(size=input_shape)

    # model(data)
    
    target_path = f"./{model_name}/"
    from pathlib import Path
    Path(target_path).mkdir(parents=True, exist_ok=True)

    torch.save(model, target_path + 'model.pt')  # 전체 모델 저장
    torch.save(model.state_dict(), target_path + 'model_state_dict.pt') 

    print("-"*10,f"Download and export {model_name} complete","-"*10)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='resnet50' , type=str)


    args = parser.parse_args()

    model_name = args.model
    img_size = 224


    if model_name == 'inception_v3':
        img_size = 299
    download_model(model_name,img_size)
