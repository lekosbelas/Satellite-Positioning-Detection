import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    print(f"Model created successfully with {num_classes} classes.")
    return model
