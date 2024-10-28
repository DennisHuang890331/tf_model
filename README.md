# Tensorflow Model Implementation

This repository is a practice for implementing deep learning models using the TensorFlow framework.

The following models are implemented:

DeepLabV3+ [Link](https://arxiv.org/abs/1802.02611)

Masked autoencoder

Vision Transformer [Link](https://arxiv.org/abs/2010.11929)

Swin Transformer [Link](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)

# How to use

### Create Python env

```
conda create --name tf_model python=3.11.5
conda avtivate tf_model
pip3 install -r requirements.txt
```

### DeepLabV3+

```
from TensorFlow_model import deeplab_v3plus
backbone = 'resnet50' # or 'inceptionresnetv2'
model = deeplab_v3plus.build_model(backbone=backbone)
model.summary()
```

### Mask AutoEncoder

```
from TensorFlow_model import masked_autoencoder
model = masked_autoencoder.build_model()
model.summary()
```

### Vision Transformer

```
from TensorFlow_model import vision_transformer
model = vision_transformer.build_model()
model.summary()
```

### Swin Transformer

```
from TensorFlow_model import swin_transformer
model = swin_transformer.build_model()
model.summary()
```
