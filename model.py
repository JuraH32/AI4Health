import torch.nn as nn
import timm

class SwinTransformerClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformerClassificationModel, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ConvNeXtClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXtClassificationModel, self).__init__()
        self.model = timm.create_model('convnext_base', pretrained=True)
        in_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)