# Emotion Detection AI
A deep learning model that detects and classifies human emotions from facial expressions using a Convolutional Neural Network (CNN). Trained on 4,550 images, this model recognizes emotions such as happy, sad, angry, surprise, fear, disgust, and neutral.

## Features
*  Trained using a dataset of 4,550 labeled facial images
*  Uses CNN architecture for accurate emotion recognition
* Supports real-time emotion detection using webcam input
* Image preprocessing (resizing, grayscale conversion, normalization)
* Achieves high accuracy with data augmentation and fine-tuning
* Can be deployed as a web app or integrated into chatbots, mental health monitoring, and AI assistants

## Dataset
The dataset consists of 7 emotion classes with balanced image distribution

Preprocessed with resizing, grayscale conversion, and normalization

## Model Architecture
Convolutional Layers for feature extraction

Pooling Layers to reduce dimensionality

Fully Connected Layers for classification

Softmax Activation for predicting emotion probabilities

Optimized using Adam optimizer and cross-entropy loss
