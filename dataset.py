from torch.utils.data import Dataset
from torchvision import transforms

import torch
import librosa


class CovidDataset(Dataset):
    def __init__(self,df):
        super(CovidDataset, self).__init__()
        self.ids = df['id']
        self.labels = df['covid19']
        self.ages = df['age']
        self.respiratory_conditions = df['respiratory_condition']
        self.fever_or_muscle_pains = df['fever_or_muscle_pain']
        self.totensor = transforms.ToTensor()


    def __getitem__(self,index):
        id = self.ids[index]
        label = self.labels[index]
        age = self.ages[index]
        respiratory_condition = self.respiratory_conditions[index]
        fever_or_muscle_pain = self.fever_or_muscle_pains[index]
        image = path[id]
        image = self.totensor(image)
        return image, age, respiratroy_condition, fever_or_muscle_pain,label
        
    def __len__(self):
        return len(self.ids)