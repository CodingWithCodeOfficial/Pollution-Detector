# Earth Search Satellite Image Classifier

A lightweight machine learning pipeline that fetches satellite image previews from the Earth Search STAC API and trains a fast TensorFlow image classifier with built-in visual analysis.

## 🚀 Overview

This project automatically:

- Queries satellite imagery using a geographic bounding box
- Downloads preview images from STAC collections (Sentinel-2 or NAIP)
- Generates weak labels using an edge-based haze proxy
- Trains a small convolutional neural network
- Produces evaluation plots and visual explanations (Grad-CAM)

It is designed to be fast, self-contained, and easy to experiment with.

## 🛠️ Key Features

- 🌍 **Earth Search STAC API integration**
- 🖼️ **Automatic image preview collection**
- 🏷️ **Weakly supervised labeling** (no manual labels required)
- 🧠 **TensorFlow CNN** with data augmentation
- ⏱️ **Early stopping** and training logs
- 📊 **Confusion matrices** and prediction galleries
- 🔥 **Grad-CAM visual explanations**
- 💾 **Auto-saved plots** and final model export

## 🧰 Technologies Used

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Data Processing:** NumPy
- **Visualization:** Matplotlib
- **Data Source:** Earth Search STAC API

## 📊 Output

The script produces:

- Training and validation curves
- Confusion matrices (raw and normalized)
- Prediction overlays and misclassification galleries
- Grad-CAM attention maps
- A saved TensorFlow model file (`saved_model`)

## 🎯 Project Purpose

This project explores how satellite imagery can be rapidly analyzed using **weak supervision** and **explainable deep learning** techniques.  
It serves as an experimental pipeline for remote sensing, environmental analysis, and ML visualization.

## 🚧 Status

**Experimental / Research Prototype**
