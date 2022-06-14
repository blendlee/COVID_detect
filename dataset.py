from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import torch


class CovidDataset(Dataset):
    def __init__(self,df,extractor):
        super(CovidDataset, self).__init__()
        self.totensor = ToTensor()
        self.ids = df['id']
        self.labels = df['covid19']
        self.ages = df['age']
        self.respiratory_conditions = df['respiratory_condition']
        self.fever_or_muscle_pains = df['fever_or_muscle_pain']
        self.extractor=extractor

    def __getitem__(self,index):
        id = self.ids[index]
        label = torch.tensor(self.labels[index])
        age = self.ages[index]
        respiratory_condition = self.respiratory_conditions[index]
        fever_or_muscle_pain = self.fever_or_muscle_pains[index]
        img_path = f'/opt/ml/COVID_detect/data/train_image/{str(id).zfill(5)+".png"}'
        img = Image.open(img_path).convert('RGB')
        img = self.totensor(img)
        img = self.extractor(img,return_tensors='pt')['pixel_values']

        return img, age, respiratory_condition, fever_or_muscle_pain,label
        
    def __len__(self):
        return len(self.ids)