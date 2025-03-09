import torch.nn as nn
import torch
import timm

class SwinTransformerClassificationModel(nn.Module):
    def __init__(self, num_classes=3, in_channels=1, trainable=True):
        super(SwinTransformerClassificationModel, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes, global_pool='avg', in_chans=in_channels)
        
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)
    
class SwinMammoClassifier(nn.Module):
    def __init__(self, embed_dim=1024, num_classes=3, in_channels=1):
        super(SwinMammoClassifier, self).__init__()
        self.swin1 = timm.create_model("swin_base_patch4_window7_224", pretrained=True, in_chans=in_channels)       
        self.swin2 = timm.create_model("swin_base_patch4_window7_224", pretrained=True, in_chans=in_channels)
        
        # Ensure full training: set requires_grad True for all parameters in the backbone.
        for param in self.swin1.parameters():
            param.requires_grad = True
            
        for param in self.swin2.parameters():
            param.requires_grad = True
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8)
        self.mlo_classifier = nn.Linear(2 * embed_dim, num_classes)
        self.cc_classifier = nn.Linear(2 * embed_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x_mlo, x_cc):
        # Feature Extraction
        f_mlo = self.swin1(x_mlo)  # [B, D]
        f_cc = self.swin2(x_cc)    # [B, D]

        # Cross Attention
        attended_cc, _ = self.cross_attention(f_cc.unsqueeze(0), f_mlo.unsqueeze(0), f_mlo.unsqueeze(0))
        attended_mlo, _ = self.cross_attention(f_mlo.unsqueeze(0), f_cc.unsqueeze(0), f_cc.unsqueeze(0))
        
        # Flatten and Concatenate
        f_mlo = attended_mlo.squeeze(0)
        f_cc = attended_cc.squeeze(0)
        
        if f_mlo.size(0) != f_cc.size(0):
            raise ValueError("Feature sizes do not match.")
            
        f_joint = torch.cat([f_mlo, f_cc], dim=-1)  # [B, 2*D]
        if self.training:
            f_joint = self.dropout(f_joint)

        y_mlo = self.mlo_classifier(f_joint)
        y_cc = self.cc_classifier(f_joint)

        return y_mlo, y_cc


class ConvNeXtClassificationModel(nn.Module):
    def __init__(self, num_classes=3, in_channels=1):
        super(ConvNeXtClassificationModel, self).__init__()
        self.model = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=num_classes, global_pool='avg', in_chans=in_channels)
        
        # Make sure all parameters are trainable.
        for param in self.model.parameters():
            param.requires_grad = True
        

    def forward(self, x):
        return self.model(x)

class LeisureDetectionModel(nn.Module):
    def __init__(self, backbone_cls, backbone_kwargs={}, head_type="simple", detector_type="custom", **detector_kwargs):
        super(LeisureDetectionModel, self).__init__()
        # Initialize backbone using provided classification model.
        self.backbone = backbone_cls(**backbone_kwargs)
        backbone_model = self.backbone.model if hasattr(self.backbone, 'model') else self.backbone
        
        # Remove the classifier head and pooling.
        if hasattr(backbone_model, 'reset_classifier'):
            backbone_model.reset_classifier(0)
        if hasattr(backbone_model, 'global_pool'):
            backbone_model.global_pool = nn.Identity()
            
        # Determine feature dimension.
        in_channels = getattr(backbone_model, 'num_features', 1024)
        self.detector_type = detector_type

        if detector_type == "custom":
            # Detection head architectures.
            if head_type == "simple":
                self.bbox_regressor = nn.Conv2d(in_channels, 4, kernel_size=1)
                self.classifier = nn.Conv2d(in_channels, 1, kernel_size=1)
            elif head_type == "fcn":
                self.bbox_regressor = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels, 4, kernel_size=1)
                )
                self.classifier = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels, 1, kernel_size=1)
                )
            elif head_type == "custom":
                reduced_channels = in_channels // 2
                self.bbox_regressor = nn.Sequential(
                    nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(reduced_channels),
                    nn.ReLU(),
                    nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(reduced_channels),
                    nn.ReLU(),
                    nn.Conv2d(reduced_channels, 4, kernel_size=1)
                )
                self.classifier = nn.Sequential(
                    nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(reduced_channels),
                    nn.ReLU(),
                    nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(reduced_channels),
                    nn.ReLU(),
                    nn.Conv2d(reduced_channels, 1, kernel_size=1)
                )
            else:
                raise ValueError(f"Unknown head_type: {head_type}")
        elif detector_type in ["fasterrcnn", "retinanet", "efficientdet", "yolo"]:
            # Wrap the backbone to return a dict with key "0"
            class CustomBackboneWrapper(nn.Module):
                def __init__(self, backbone, out_channels):
                    super(CustomBackboneWrapper, self).__init__()
                    self.backbone = backbone
                    self.out_channels = out_channels
                def forward(self, x):
                    features = self.backbone(x)
                    return {"0": features}
            self.custom_backbone = CustomBackboneWrapper(backbone_model, in_channels)
            if detector_type == "fasterrcnn":
                from torchvision.models.detection.faster_rcnn import FasterRCNN
                num_classes = detector_kwargs.get("num_classes", 3)
                self.detector = FasterRCNN(self.custom_backbone, num_classes=num_classes)
            elif detector_type == "retinanet":
                from torchvision.models.detection.retinanet import RetinaNet
                num_classes = detector_kwargs.get("num_classes", 3)
                self.detector = RetinaNet(self.custom_backbone, num_classes=num_classes)
            elif detector_type == "efficientdet":
                from efficientdet_pytorch import EfficientDet
                num_classes = detector_kwargs.get("num_classes", 3)
                self.detector = EfficientDet(self.custom_backbone, num_classes=num_classes)
            elif detector_type == "yolo":
                from yolov5 import YOLOv5
                num_classes = detector_kwargs.get("num_classes", 3)
                self.detector = YOLOv5(self.custom_backbone, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown detector_type: {detector_type}")

    def forward(self, x):
        if self.detector_type == "custom":
            if hasattr(self.backbone, 'model'):
                features = self.backbone.model(x)
            else:
                features = self.backbone(x)
            bbox = self.bbox_regressor(features)
            score = self.classifier(features)
            return bbox, score
        else:
            return self.detector(x)

