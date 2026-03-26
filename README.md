# vixora-radiology

AI system for chest X-ray analysis using deep learning. Classifies images (Normal/Pneumonia), outputs confidence score, and provides Grad-CAM visualization. Built with CNNs and transfer learning (ResNet, VGG, EfficientNet) with a simple web interface for fast and reliable diagnosis support.

> [!NOTE]
> Dataset used in this project: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Setup

### Create virtual environment (Python 3.11 recommended)

py -3.11 -m venv .venv  
.venv\Scripts\activate  
python --version  

### Install dependencies

python -m pip install --upgrade pip  
pip install -r requirements.txt  

## Verify installation

python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__, '| GPU:', tf.config.list_physical_devices('GPU'))"  
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"  
python -c "import streamlit; print('Streamlit OK')"  
python -c "import albumentations; print('Albumentations OK')"  

## Enable GPU (PyTorch only)

pip uninstall torch torchvision torchaudio -y  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  

> [!WARNING]
> Requirements:
> - NVIDIA GPU
> - Installed NVIDIA drivers (CUDA supported)

## Project Structure

```
    vixora-radiology/
    ├── data/
    │   └── raw/
    │       └── chest_xray/   # train / test / val
    ├── notebooks/
    ├── src/
    ├── main.py
    ├── requirements.txt
    ├── .gitignore

```