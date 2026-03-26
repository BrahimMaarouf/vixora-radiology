# vixora-radiology
AI system for chest X-ray analysis using deep learning. Classifies images (Normal/Pneumonia), outputs confidence score, and provides Grad-CAM visualization. Built with CNNs and transfer learning (ResNet, VGG, EfficientNet) with a simple web interface for fast, reliable diagnosis support.

> [!NOTE]
> Dataset we are using in this project : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia


## creating ENV & Installing requirement for vixora-radiology
 - py -3.11 -m venv .venv > we use 3.11 python version most stable So make sure U have it installed
   .venv\Scripts\activate
 - python --version
 - python -m pip install --upgrade pip
 - pip install -r requirements.txt

## Test If everything is working well
 - python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"

 - python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

 - python -c "import streamlit; print('Streamlit OK')"

 - python -c "import albumentations; print('Albumentations OK')"

### If you want gpu instead of cpu 
 - pip uninstall torch torchvision torchaudio -y
 - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
 > [!WARNING]
 > Requirements : 
    > NVIDIA GPU
    > CUDA drivers installed

## Project Current Structure

vixora-radiology/
├── data/
│   └── raw/
│       └── chest_xray/     ← train, test, val here
├── notebooks/
├── src/
├── main.py
├── requirements.txt
├── .gitignore