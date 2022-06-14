

from transformers import (ResNetModel,ResNetPreTrainedModel,AutoFeatureExtractor,)
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
import torch
import torch.nn as nn




class COVIDModel(ResNetPreTrainedModel):
    def __init__(self,config,MODEL_NAME):
        super(COVIDModel,self).__init__(config)
        self.config=config
        self.model = ResNetModel.from_pretrained(MODEL_NAME)

        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.config.hidden_sizes[-1], 2)
        )


    def forward(self,image,age,condition,pain,labels=None):

        outputs = self.model(image)
        pooled_output = outputs.pooler_output
        logits = self.cls(pooled_output)
        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))


        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
