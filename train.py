import random
import pandas as pd
import numpy as np
import os
import torch

from tqdm import tqdm,trange
from dataset import *
from model import *
from transformers import ResNetConfig, TrainingArguments,get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import (DataLoader, RandomSampler)
from sklearn.model_selection import StratifiedKFold

CFG = {
    'SR':16000,
    'N_MFCC':32, # MFCC 벡터를 추출할 개수
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def split_data(dataset, num_splits):
    split = StratifiedKFold(n_splits=num_splits, random_state=42, shuffle=True)
    for train_index, dev_index in split.split(dataset, dataset["covid19"]):
        train_dataset = dataset.loc[train_index]
        dev_dataset = dataset.loc[dev_index]
    
        yield train_dataset, dev_dataset

def main():

    args = TrainingArguments(
        output_dir='models',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        
    )

    seed_everything(CFG['SEED']) # Seed 고정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    MODEL_NAME = "microsoft/resnet-50"
    dataset = pd.read_csv('data/new_train_data.csv')
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    for fold, (train_df,val_df) in enumerate(split_data(dataset, num_splits=5), 1):
    
        train_dataset = CovidDataset(train_df.reset_index(drop=True),extractor)
        valid_dataset = CovidDataset(val_df.reset_index(drop=True),extractor)

        config = ResNetConfig(MODEL_NAME)
        model = COVIDModel(config,MODEL_NAME)
        model.to(device)
        trained_model = train(args,train_dataset,valid_dataset,model)
    

    save_pth=f'/opt/ml/COVID_detect/models/demo.pth'
    torch.save(trained_model.state_dict(),save_pth)

    


def train(args,train_dataset,val_dataset, model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #kfold validation 추가 예정
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

    val_sampler=RandomSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.per_device_eval_batch_size)


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


            #if torch.cuda.is_available():
            #    batch = tuple(t.cuda()  for t in batch if type(t) != tuple)
            inputs = {'image': batch[0].squeeze().to(device),
                      'age' : batch[1],
                      'condition' : batch[2],
                      'pain' : batch[3]
                        }
            

            target = batch[-1].to(device)
            outputs = model(**inputs,labels=target)

            
            loss = outputs.loss
            total_loss+=loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            torch.cuda.empty_cache()

        #torch.save(model.state_dict(), f'/opt/ml/Project/VQA/models/vqa_1020_epoch{epoch+11}.pth')
        print(f'train loss : {total_loss/steps}')

        with torch.no_grad():
            print('Calculating Valdiation Result............')
            model.eval()
            val_loss=0
            val_step=0
            exact_match=0
            for _, batch in enumerate(tqdm(val_dataloader)):
                val_step+=1
                inputs = {'image': batch[0].squeeze().to(device),
                        'age' : batch[1],
                        'condition' : batch[2],
                        'pain' : batch[3]
                            }
                target = batch[-1].to(device)
                outputs = model(**inputs,labels=target)
                loss = outputs.loss
                logits=outputs.logits
                pred_idx = logits.argmax(-1)
                val_loss+=loss
                
                for i in range(len(target)):
                    if target[i] == pred_idx[i]:
                        exact_match+=1

            val_loss /= val_step
            exact_match = exact_match/len(val_dataset)
            print(f'Validation loss : {val_loss}')
            print(f'Validation Exact match : {exact_match*100}%')
     
        
    return model

if __name__ == '__main__':
    main()