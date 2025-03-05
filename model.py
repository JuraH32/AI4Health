import torch.nn as nn
import timm

class SwinTransformerClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformerClassificationModel, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        # old_proj = self.model.patch_embed.proj  # original conv layer
        # self.model.patch_embed.proj = nn.Conv2d(
        #     in_channels=1,
        #     out_channels=old_proj.out_channels,
        #     kernel_size=old_proj.kernel_size,
        #     stride=old_proj.stride,
        #     padding=old_proj.padding,
        #     bias=old_proj.bias is not None
        # )

    def forward(self, x):
        out = self.model(x)
        return out
    
class SwinMammoClassifier(nn.Module):
    def __init__(self, embed_dim=768, num_classes=2):
        super(SwinMammoClassifier, self).__init__()
        self.swin = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=0)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8)
        self.classifier_mlo = nn.Linear(2 * embed_dim, num_classes)
        self.classifier_cc = nn.Linear(2 * embed_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x_mlo, x_cc):
        # Feature Extraction
        f_mlo = self.swin(x_mlo)  # [B, D]
        f_cc = self.swin(x_cc)    # [B, D]

        # Cross Attention
        attended_cc, _ = self.cross_attention(f_cc.unsqueeze(1), f_mlo.unsqueeze(1), f_mlo.unsqueeze(1))
        attended_mlo, _ = self.cross_attention(f_mlo.unsqueeze(1), f_cc.unsqueeze(1), f_cc.unsqueeze(1))
        
        # Flatten and Concatenate
        f_mlo = attended_mlo.squeeze(1)
        f_cc = attended_cc.squeeze(1)
        f_joint = torch.cat([f_mlo, f_cc], dim=-1)  # [B, 2*D]
        f_joint = self.dropout(f_joint)

        # Classification
        y_mlo = self.classifier_mlo(f_joint)
        y_cc = self.classifier_cc(f_joint)

        return y_mlo, y_cc


class ConvNeXtClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXtClassificationModel, self).__init__()
        self.model = timm.create_model('convnext_base', pretrained=True)
        in_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)