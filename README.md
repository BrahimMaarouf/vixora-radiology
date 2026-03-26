# vixora-radiology
AI system for chest X-ray analysis using deep learning. Classifies images (Normal/Pneumonia), outputs confidence score, and provides Grad-CAM visualization. Built with CNNs and transfer learning (ResNet, VGG, EfficientNet) with a simple web interface for fast, reliable diagnosis support.

# creating ENV & Installing requirement for vixora-radiology
 - python -m venv venv
   .\venv\Scripts\Activate.ps1
 - python -m pip install --upgrade pip
 - pip install -r requirements.txt

# Test If everything is working well
 - python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"

 - python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

 - python -c "import streamlit; print('Streamlit OK')"

 - python -c "import albumentations; print('Albumentations OK')"

> [!NOTE]
> Dataset we are using in this project : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia