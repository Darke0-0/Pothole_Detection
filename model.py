"""
Python script to prepare FasterRCNN model.
"""
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import  FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import config

def model():
    # load the pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
                                                                 min_size=config.MIN_SIZE)
    # one class is for pot holes, and the other is background
    num_classes = 2
    # get the input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace pre-trained head with our features head
    # the head layer will classify the images based on our data input features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model