import time
import argparse
import numpy as np
import mxnet as mx
import gluonnlp as nlp

ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()


def bert_download(model_name,seq_length, batch_size, dtype="float32"):
    inputs = np.random.randint(0, 2000, size=(batch_size, seq_length)).astype(dtype)
    token_types = np.random.uniform(size=(batch_size, seq_length)).astype(dtype)
    valid_length = np.asarray([seq_length] * batch_size).astype(dtype)
        
    inputs_nd = mx.nd.array(inputs, ctx=ctx)
    token_types_nd = mx.nd.array(token_types, ctx=ctx)
    valid_length_nd = mx.nd.array(valid_length, ctx=ctx)

    # Instantiate a BERT classifier using GluonNLP
    if model_name == "bert_base":
        model_name_ = "bert_12_768_12"
        dataset = "book_corpus_wiki_en_uncased"
        model, _ = nlp.model.get_model(
                name=model_name_,
                dataset_name=dataset,
                pretrained=True,
                use_pooler=True,
                use_decoder=False,
                use_classifier=False,
            )
        model = nlp.model.BERTClassifier(model, dropout=0.1, num_classes=2)
        model.initialize(ctx=ctx)
        model.hybridize(static_alloc=True)
                
        mx_out = model(inputs_nd, token_types_nd, valid_length_nd)
        mx_out.wait_to_read()

        # print model info
        #print("-"*10,f"{model_name} Parameter Info","-"*10)
        #print(model.summary(inputs_nd,token_types_nd, valid_length_nd))       

    elif model_name == "distilbert":
        model_name_="distilbert_6_768_12"
        dataset = "distilbert_book_corpus_wiki_en_uncased"
        model, _ = nlp.model.get_model(
                name=model_name_,
                dataset_name=dataset,
                pretrained=True,
            )
        model.hybridize(static_alloc=True)

        mx_out = model(inputs_nd, valid_length_nd)
        mx_out.wait_to_read()

        # print("-"*10,f"{model_name} Parameter Info","-"*10)
        # print(model.summary(inputs_nd, valid_length_nd))
  

    target_path = f"./{model_name}_{batch_size}"
    from pathlib import Path
    Path(target_path).mkdir(parents=True, exist_ok=True)  

    model.export(f'{model_name}_{batch_size}/model')
    print("-"*10,f"Download {model_name} complete","-"*10)  

    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='bert_base' , type=str)
    parser.add_argument('--batchsize',default=1 , type=int)
    parser.add_argument('--seq',default=128 , type=int)

    args = parser.parse_args()
    model_name = args.model
    batchsize = args.batchsize
    seq_length=args.seq


    bert_download(model_name,seq_length,batchsize)
