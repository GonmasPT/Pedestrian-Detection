import torchvision
import torch

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes: int, seed: int = 42):
    """
    Creates a Mask R-CNN object detection and instance segmentation model with a ResNet-50-FPN backbone.
    :params num_classes: number of classes in box_predictor and mask_predictor.
            seed: random seed value. Defaults to 42.
    :return: Mask R-CNN model.
    """
    # Load model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    # Freeze all layer in base model
    for param in model.parameters():
        param.requires_grad = False

    # Set random seed
    torch.manual_seed(seed)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the classifier with a new one, that has num_classes which is user-defined
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(
    #    in_features_mask,
    #    hidden_layer,
    #    num_classes)

    return model