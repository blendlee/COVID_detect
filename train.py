import random
import pandas as pd
import numpy as np
import os
import librosa
import torch

from tqdm import tqdm
from dataset import *
from transformers import 
from torch.utils.data import (DataLoader, RandomSampler)

CFG = {
    'SR':16000,
    'N_MFCC':32, # MFCC 벡터를 추출할 개수
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def main():
    seed_everything(CFG['SEED']) # Seed 고정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    df = pd.read_csv('data/train_data.csv')

    #df 나누기

    train_df = df[:100]
    val_df =  df[100:]

    train_dataset = CovidDataset(train_df)
    valid_dataset = CovidDataset(val_df)

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    trained_data = train(train_dataset,model,feature_extractor)

    inputs = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])
    

    


def train(dataset, model,feature_extractor):

    #kfold validation 추가 예정
    train_sampler = RandomSampler(datasets)
    train_dataloader = DataLoader(datasets, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

    val_sampler=RandomSampler(val_datasets)
    val_dataloader = DataLoader(val_datasets, sampler=val_sampler, batch_size=args.per_device_eval_batch_size)


    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)


    global_step = 0
    model.zero_grad()
    torch.cuda.empty_cache()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        total_loss=0
        steps=0

        for _, batch in enumerate(epoch_iterator):
    
            steps+=1
            model.train()
            images= batch[3]
            if torch.cuda.is_available():
                batch = tuple(t.cuda()  for t in batch if type(t) != tuple)
            inputs = {'image': batch[0],
                      'age' = batch[1]
                      'condition' = batch[2]
                      'pain' = batch[3]
                        }
            

            target = batch[-1]
            outputs = model(**inputs,labels=target)

            
            loss = outputs.loss
            total_loss+=loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            torch.cuda.empty_cache()

        torch.save(model.state_dict(), f'/opt/ml/Project/VQA/models/vqa_1020_epoch{epoch+11}.pth')
        print(f'train loss : {total_loss/steps}')

        # with torch.no_grad():
        #     print('Calculating Valdiation Result............')
        #     model.eval()
        #     with open('/opt/ml/Project/VQA/data/1020_trainval_label2ans.kvqa.pkl','rb') as f :
        #         label2ans = pickle.load(f)

        #     val_loss=0
        #     val_step=0
        #     exact_match=0
        #     for _, batch in enumerate(tqdm(val_dataloader)):
        #         val_step+=1
        #         images= batch[3]
        #         if torch.cuda.is_available():
        #             batch = tuple(t.cuda()  for t in batch if type(t) != tuple)
        #         inputs = {'input_ids': batch[0],
        #                     'token_type_ids': batch[1],
        #                     'attention_mask': batch[2],
        #                     'image' : images
        #                     }
        #         target = batch[-1]
        #         outputs = model(**inputs,labels=target)
        #         loss = outputs.loss
        #         logits=outputs.logits
        #         pred_idx = logits.argmax(-1)
        #         val_loss+=loss
                
        #         for step, idx in enumerate(pred_idx):
        #             if target[step][idx] != 0:
        #                 exact_match+=1


        #     val_loss /= val_step
        #     exact_match /= len(val_datasets)
        #     print(f'Validation loss : {val_loss}')
        #     print(f'Validation Exact match : {exact_match*100}%')
        

        
    return model