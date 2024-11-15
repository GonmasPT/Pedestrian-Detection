
# Pedestrian Detection Application

This project demonstrates developing a pedestrian detection app using a fine-tuned Faster R-CNN model on the Penn-Fudan dataset, achieving an AP of 0.820 and an AR of 0.852 at IoU=0.50:0.95. The application provides an interactive Gradio interface for object detection on input images.

## Table of Contents
- [Structure](#structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)

## Structure
The key sections in the application are:
- **Model Initialization**: Sets up Faster R-CNN with a pre-trained ResNet-50-FPN backbone and loads saved weights.
- **Prediction Function**: Performs predictions on input images and returns detected objects with bounding boxes and prediction times.
- **Gradio Interface**: Creates an interactive web interface for users to upload images and view detection results.

## Requirements
- PyTorch (2.4.1)
- TorchVision (0.19.1)
- Gradio (5.0.2)

## Setup

1. **Pull the Docker Image**:
   ```bash
   docker pull gonmas/pedestrian-detection:latest
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 7860:7860 -e GRADIO_SERVER_NAME=0.0.0.0 gonmas/pedestrian-detection:latest
   ```
Alternatively, if you prefer to set up the environment manually, you can clone the repository and run the application using:
   ```bash
   git clone https://github.com/GonmasPT/Pedestrian-Detection.git
   python app.py
   ```
## Usage

**Access the Interface**: Open your web browser and navigate to [http://localhost:7860](http://localhost:7860) to use the application.

## Results
Once deployed, the Gradio app interface will allow users to upload an image and view detected objects with bounding boxes and labels indicating detection confidence.

## Acknowledgments
The development of this application is based on PyTorch's TorchVision Object Detection Finetuning Tutorial, adapted to use Faster R-CNN for pedestrian detection.
