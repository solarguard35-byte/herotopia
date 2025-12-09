YOLOv8 Image Classification Training
This repository contains a Jupyter notebook for training a YOLOv8s model for image classification using the Ultralytics library. The model is fine-tuned on a custom dataset downloaded from Roboflow for a multi-class classification task (4 classes).
Overview

Model: YOLOv8s-cls (pretrained on ImageNet)
Dataset: Custom classification dataset from Roboflow (992 training images, 284 validation images, 142 test images across 4 classes)
Training Parameters:
Epochs: 10 (Note: Filename suggests 1000 epochs, but the code trains for 10; adjust as needed)
Batch Size: 16
Optimizer: Adam
Learning Rate: 0.001 (with cosine annealing)
Patience: 0 (no early stopping)

Results: Best validation top-1 accuracy: 90.8%

The notebook is designed to run in Google Colab with GPU acceleration (e.g., Tesla T4).
Dataset
The dataset is sourced from Roboflow:

Workspace: ezer-01efc
Project: classification-dataset4-zig5s
Version: 2
Format: Folder (for classification)

You can view or download the dataset here: Roboflow Dataset
Classes: 4 (specific class names are not detailed in the notebook; infer from dataset).
Requirements

Python 3.10+
Libraries:
ultralytics (for YOLOv8)
roboflow (for dataset download)
torch (with CUDA support for GPU)


Install dependencies:
textpip install ultralytics roboflow
How to Run

Clone the Repository:textgit clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Open the Notebook:
Use Google Colab: Upload YOLOv8s_classification_1000_epocs_(1) (1).ipynb to Colab.
Or locally with Jupyter: jupyter notebook YOLOv8s_classification_1000_epocs_(1) (1).ipynb

Set Up Roboflow API Key:
Replace the API key in the notebook with your own: rf = Roboflow(api_key="YOUR_API_KEY")

Run the Cells:
Execute step-by-step to check GPU, install packages, download dataset, and train the model.
Training outputs will be saved in /content/runs/classify/trainX (weights, metrics, etc.).

Customize Training:
Modify epochs, batch size, or other params in the model.train() call.
For longer training (e.g., 1000 epochs), update epochs=1000.


Training Results
From the provided run:

Final Validation Metrics:
Top-1 Accuracy: 90.8%
Top-5 Accuracy: 100%

Speed: ~0.9ms inference per image on Tesla T4.
Confusion matrix and other plots are generated in the runs/ directory.

Example training log (excerpt):
textEpoch    GPU_mem       loss  Instances       Size
1/10     0.576G      1.119         16        224
...
10/10     0.639G     0.1665         16        224
License
This project is licensed under the MIT License. The YOLOv8 model and Ultralytics library are under AGPL-3.0; refer to their documentation for usage.
Acknowledgments

Ultralytics YOLOv8
Roboflow for dataset hosting
