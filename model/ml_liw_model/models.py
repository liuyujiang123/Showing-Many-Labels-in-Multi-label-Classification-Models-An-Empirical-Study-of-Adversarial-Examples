import torchvision.models as models
from torch.nn import Parameter
from model.ML_GCN_model.util import *
import torch
import torch.nn as nn
import pickle
import torch.optim as optim


class Inceptionv3Rank(nn.Module):
    def __init__(self, model, num_classes):
        super(Inceptionv3Rank, self).__init__()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes, bias=True)
        self.model = model
        self.num_classes = num_classes
        self.std = Parameter(torch.tensor([0.5, 0.5, 0.5]))
        self.mean = Parameter(torch.tensor([0.5, 0.5, 0.5]))
        self.std.requires_grad = False
        self.mean.requires_grad = False
    def forward(self, input):
        model = self.model
        x = input.permute(0, 2, 3, 1) - self.mean
        x = x / self.std
        x = x.permute(0, 3, 1, 2)
        x = model(x)
        x = torch.sigmoid(x)
        return x

def inceptionv3_attack(num_classes, save_model_path=None):
    model = models.inception_v3(pretrained=True)
    model = Inceptionv3Rank(model, num_classes)
    checkpoint = torch.load(save_model_path) #测试使用 ,map_location='cpu'
    model.load_state_dict(checkpoint, strict=False)
    return model