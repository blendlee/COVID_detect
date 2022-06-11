import random
import pandas as pd
import numpy as np
import os
import librosa

from tqdm.auto import tqdm

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings(action='ignore') 


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

    df = pd.read_csv('data/train_data.csv')

    #df 나누기

    train_df = df[:100]
    val_df =  df[100:]

    trained_data = train(train_df)


def train():
    






def get_mfcc_feature(df, data_type, save_path):
    # Data Folder path
    root_folder = 'data/'
    if os.path.exists(save_path):
        print(f'{save_path} is exist.')
        return
    features = []
    for uid in tqdm(df['id']):
        root_path = os.path.join(root_folder, data_type)
        path = os.path.join(root_path, str(uid).zfill(5)+'.wav')

        # librosa패키지를 사용하여 wav 파일 load
        y, sr = librosa.load(path, sr=CFG['SR'])
        
        # librosa패키지를 사용하여 mfcc 추출
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG['N_MFCC'])

        y_feature = []
        # 추출된 MFCC들의 평균을 Feature로 사용
        for e in mfcc:
            y_feature.append(np.mean(e))
        features.append(y_feature)
    
    # 기존의 자가진단 정보를 담은 데이터프레임에 추출된 오디오 Feature를 추가
    mfcc_df = pd.DataFrame(features, columns=['mfcc_'+str(x) for x in range(1,CFG['N_MFCC']+1)])
    df = pd.concat([df, mfcc_df], axis=1)
    df.to_csv(save_path, index=False)
    print('Done.')

get_mfcc_feature(train_df, 'train', 'data/train_mfcc_data.csv')
get_mfcc_feature(test_df, 'test', 'data/test_mfcc_data.csv')

train_df = pd.read_csv('data/train_mfcc_data.csv')


# 학습데이터를 모델의 input으로 들어갈 x와 label로 사용할 y로 분할
train_x = train_df.drop(columns=['id', 'covid19'])
train_y = train_df['covid19']

def onehot_encoding(ohe, x):
    # 학습데이터로 부터 fit된 one-hot encoder (ohe)를 받아 transform 시켜주는 함수
    encoded = ohe.transform(x['gender'].values.reshape(-1,1))
    encoded_df = pd.DataFrame(encoded, columns=ohe.categories_[0])
    x = pd.concat([x.drop(columns=['gender']), encoded_df], axis=1)
    return x

ohe = OneHotEncoder(sparse=False)
ohe.fit(train_x['gender'].values.reshape(-1,1))
train_x = onehot_encoding(ohe, train_x)

model = MLPClassifier(random_state=CFG['SEED']) # Sklearn에서 제공하는 Multi-layer Perceptron classifier 사용
print('start training')
model.fit(train_x, train_y) # Model Train

test_x = pd.read_csv('data/test_mfcc_data.csv')
test_x = test_x.drop(columns=['id'])
# Data Leakage에 유의하여 train data로만 학습된 ohe를 사용
test_x = onehot_encoding(ohe, test_x)

# Model 추론
print('start predicting')
preds = model.predict(test_x)

print('making submission csv.....')
submission = pd.read_csv('data/submission/sample_submission.csv')
submission['covid19'] = preds
submission.to_csv('data/submission/submit.csv', index=False)
print('DONE!!!!!!')