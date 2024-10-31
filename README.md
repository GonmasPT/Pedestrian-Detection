# Object Detection Model Training and Deployment

This project provides a step-by-step guide for developing and deploying an object detection model using a pre-trained Mask R-CNN finetuned on the Penn-Fudan Database for Pedestrian Detection and Segmentation. The notebook contains code for model training, evaluation, and deployment, utilizing a simple Gradio app interface.

## Table of Contents
- [Overview](#overview)
- [Structure](#structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [Training and Evaluation](#training-and-evaluation)
  - [Deployment](#deployment)
- [Results](#results)

## Overview
The project is structured into two main sections:
1. **Development**: Includes data loading, model definition, training, and evaluation.
2. **Deployment**: Prepares the model for inference and creates a Gradio interface for user interaction.

## Structure
The key sections in the notebook are:
- **Data Loading**: Loads and displays sample images for dataset verification.
- **Custom Dataset Definition**: Defines a dataset class for PyTorch.
- **Model Initialization**: Sets up Mask R-CNN with a pre-trained ResNet-50 backbone.
- **Training Loop**: Trains the model on the dataset with specified hyperparameters.
- **Evaluation**: Visualizes model predictions on sample images.
- **Saving Model**: Saves the trained model for use in deployment.
- **Gradio Deployment**: Builds a Gradio app for interactive inference.

## Requirements
- Python (3.11.9)
- PyTorch (2.4.1)
- TorchVision (0.19.1)
- Gradio (5.0.2)
- matplotlib

## Setup

1. **Clone the Repository** (if applicable) or ensure this notebook file is accessible.
2. **Prepare the Dataset**: Place images in a folder named `data/`.
3. **Configure Gradio App**: Create a `demo/` folder and ensure it contains `app.py` and `model.py` scripts for deployment.

## Usage

### Training and Evaluation
1. **Run the Notebook**: Open `main.ipynb` and follow each section, beginning with loading and displaying an image.
2. **Train the Model**: Use the provided training loop, and adjust any parameters as needed.
3. **Save the Model**: Once training is complete, save the model to the `models/` directory.

### Deployment
1. **Prepare Deployment Files**: Place the saved model in the `demo/` directory.
2. **Launch Gradio App**:
   ```bash
   python demo/app.py
    ```
3. **Test the Interface**: Access the Gradio link provided in the terminal to test model predictions on new images.

## Results
Once deployed, the Gradio app interface will allow users to upload an image and view detected objects with bounding boxes.

## Acknowledgments
The development section of this notebook is a modified version of PyTorch's TorchVision Object Detection Finetuning Tutorial.