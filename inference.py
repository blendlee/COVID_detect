import random
import pandas as pd
import numpy as np
import os
import torch

from tqdm import tqdm,trange
from dataset import *
from model import *
from transformers import ResNetConfig, TrainingArguments,get_linear_schedule_with_warmup
from torchvision.transforms import ToTensor
from PIL import Image


def main():

    MODEL_NAME = "microsoft/resnet-50"
    test_df = pd.read_csv('data/test_data.csv')
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    config = ResNetConfig(MODEL_NAME)
    model = COVIDModel(config,MODEL_NAME)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(f'/opt/ml/COVID_detect/models/demo.pth'))
    totensor=ToTensor()


    with torch.no_grad():
        model.eval()
        ids=[]
        preds=[]
        for id,age,gender,condition,pain in tqdm(zip(test_df['id'],test_df['age'],test_df['gender'],test_df['respiratory_condition'],test_df['fever_or_muscle_pain'])):
            ids.append(id)
            img_path = f'/opt/ml/COVID_detect/data/test_image/{str(id).zfill(5)+".png"}'
            if str(id).zfill(5)+".png" not in os.listdir('/opt/ml/COVID_detect/data/test_image'):
                print('not found')
                preds.append(0)
                continue
            img = Image.open(img_path).convert('RGB')
            img = totensor(img)
            img = extractor(img,return_tensors='pt')['pixel_values'].to(device)

            inputs = {'image':img,
                      'age' : age,
                      #'gender':gender,
                      'condition' : condition,
                      'pain' : pain
                        }

            outputs = model(**inputs)

            logits=outputs.logits
            pred_idx = logits.argmax(-1).item()
            preds.append(pred_idx)


    submission = pd.DataFrame({'id' : ids , 'covid19':preds})
    submission.to_csv('data/submission/submit.csv', index=False)
if __name__ =='__main__':
    main()