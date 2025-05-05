# Bone Fractures Classification using Neural Networks
A deep learning project using Convolutional Neural Networks (CNN) to classify bone fractures from X-ray images. Built for medical image analysis and portfolio demonstration.

## Overview
This project focuses on classifying bone fractures from X-ray images into 10 distinct categories using deep learning. I explore two neural network architectures: a Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN), leveraging TensorFlow and Keras to build, train, and evaluate the models. The goal is to automatically identify fracture types, which can assist in medical diagnostics.

## Dataset
- **Source**: The dataset is stored in Kaggle.
- **Structure**: It contains 1101 X-ray images across 10 subdirectories, each representing a fracture type:  
  - Avulsion fracture  
  - Comminuted fracture  
  - Fracture Dislocation  
  - Greenstick fracture  
  - Hairline Fracture  
  - Impacted fracture  
  - Longitudinal fracture  
  - Oblique fracture  
  - Pathological fracture  
  - Spiral Fracture
- **Split**: The dataset is split into 90% training (991 images) and 10% validation (110 images).

## Methodology

### 1. Data Preprocessing
- I mount Google Drive to access the dataset and load images using `utils.image_dataset_from_directory`, resizing them to 256x256 pixels and normalizing pixel values to [0, 1].
- I extract training and validation data into NumPy arrays (`x_train`, `y_train`, `x_val`, `y_val`) and one-hot encode the labels for multi-class classification.

### 2. MLP Model
- **Architecture**: I build an MLP with dense layers (300, 250, 200, 150, 100 units), using LeakyReLU activations, batch normalization, and a 10-unit softmax output layer.
- **Training**: I compile the model with the Adam optimizer (learning rate 0.0005), categorical crossentropy loss, and train it for 16 epochs with a batch size of 32.

### 3. CNN Model
- **Architecture**: I design a CNN with four `Conv2D` layers (32, 64, 128, 256 filters, 3x3 kernels, strides=2), each followed by batch normalization, LeakyReLU, and 20% dropout. The output is flattened, passed through a 100-unit dense layer, and ends with a 10-unit softmax layer.
- **Training**: I compile the CNN with the Adam optimizer (learning rate 0.0005), categorical crossentropy loss, and train it for 10 epochs with a batch size of 32, using `ModelCheckpoint` and `TensorBoard` callbacks to save the best model and log metrics.
- **Evaluation**: The CNN achieves an accuracy of **89.34%** and a loss of **0.3964** on the validation set, indicating strong performance with room for improvement.

### 4. Visualization and Analysis
- I visualize a subset of training images to confirm correct data loading and labeling.
- I generate predictions on the validation set and display predicted vs. actual labels for 10 images. The CNN correctly predicts 8 out of 10, with misclassifications (e.g., pred=Longitudinal, act=Hairline) suggesting areas for further tuning.

## Results

- **MLP Performance**: The MLP serves as a baseline but is less effective for image data due to its inability to capture spatial features, likely resulting in lower accuracy compared to the CNN.
- **CNN Performance**: The CNN achieves a validation accuracy of **89.34%** and a loss of **0.3964**, correctly classifying 8 out of 10 sampled images. Misclassifications indicate potential improvements through hyperparameter tuning, more epochs, or data augmentation.
- **Comparison**: The CNN outperforms the MLP, as expected, due to its ability to extract spatial features from X-ray images, making it more suitable for this task.

## Requirements
- Python 3.12.7  
- TensorFlow/Keras  
- NumPy  
- Matplotlib  
- Google Colab (https://colab.research.google.com/drive/1E1bzhOLwPVGLkPx4S-p8cdKOP7zQ-Ahf?usp=sharing)
