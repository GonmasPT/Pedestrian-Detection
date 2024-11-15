### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch
from numpy import ndarray

from transforms import get_transform
from model import get_model_instance_segmentation
from timeit import default_timer as timer
from typing import Tuple, Dict
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


### 2. Model and transforms preparation ###

# Create EffNetB2 model
model = get_model_instance_segmentation(num_classes=2)  # len(classes) + background

# Load saved weights
model.load_state_dict(
    torch.load(
        f="models/fasterrcnn_resnet50_fpn_PennFudanPed.pth",
        map_location=torch.device("cpu"),  # load to CPU
        weights_only=True
    )
)


### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[ndarray, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    image = read_image(img)
    eval_transform = get_transform(train=False)

    # Put model into evaluation mode and turn on inference mode
    model.eval()
    with torch.inference_mode():
        x = eval_transform(image)
        # Convert RGBA -> RGB and move to device
        x = x[:3, ...].to("cpu")
        predictions = model([x, ])
        pred = predictions[0]

    # Create output image
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)  # Useless??
    image = image[:3, ...]
    filtered_scores_idx = torch.nonzero(pred["scores"] > 0.4).squeeze()
    if filtered_scores_idx.dim() == 0:
        pred_labels = [f"Pedestrian: {score:.3f}" for label, score in
                       zip(pred["labels"][filtered_scores_idx].unsqueeze(0),
                           pred["scores"][filtered_scores_idx].unsqueeze(0))]
        pred_boxes = pred["boxes"][filtered_scores_idx].long().unsqueeze(0)
    else:
        pred_labels = [f"Pedestrian: {score:.3f}" for label, score in
                       zip(pred["labels"][filtered_scores_idx], pred["scores"][filtered_scores_idx])]
        pred_boxes = pred["boxes"][filtered_scores_idx].long()
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    # masks = (pred["masks"] > 0.7).squeeze(1)
    # output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return output_image.permute(1, 2, 0).numpy(), pred_time


### 4. Gradio app ###

# Create title, description and article strings
title = "Pedestrian Detection"
description = "Faster R-CNN object detection model with a ResNet-50-FPN backbone"
article = "Adaption of PyTorch tutorial available at https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html"

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples/")]

# Create the Gradio demo
demo = gr.Interface(fn=predict,  # mapping function from input to output
                    inputs=gr.Image(type="filepath"),  # what are the inputs?
                    outputs=[gr.Image(type="numpy"),  # what are the outputs?
                             gr.Number(label="Prediction time (s)")],
                    # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()