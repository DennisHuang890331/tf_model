# Tensorflow Model Implementation

This repository is a practice for implementing deep learning models using the TensorFlow framework.

The following models are implemented:

DeepLabV3+ [Link](https://arxiv.org/abs/1802.02611)

Masked autoencoder

Vision Transformer [Link](https://arxiv.org/abs/2010.11929)

Swin Transformer [Link](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)

RepVGG [Link](https://arxiv.org/abs/2101.03697)

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

### RepVGG

```
import tensorflow as tf

from TensorFlow_model.repvgg import create_RepVGG_A0, repvgg_layer_convert

repvggA0 = create_RepVGG_A0() # Create RepVGG layer
input = tf.keras.layers.Input((256, 512, 3))
x = repvggA0(input)
model = tf.keras.Model(input, x) # Create Model
model.summary()
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 256, 512, 3)]     0

 rep_vgg (RepVGG)            (None, 8, 16, 1280)       7851616

=================================================================
Total params: 7851616 (29.95 MB)
Trainable params: 7827968 (29.86 MB)
Non-trainable params: 23648 (92.38 KB)
_________________________________________________________________
"""

layer = model.get_layer('rep_vgg')
layer = repvgg_layer_convert(layer) # Re-Parameterization
model.layers[1] = layer
model.summary()
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 256, 512, 3)]     0

 rep_vgg (RepVGG)            (None, 8, 16, 1280)       7028384

=================================================================
Total params: 7028384 (26.81 MB)
Trainable params: 0 (0.00 Byte)
Non-trainable params: 7028384 (26.81 MB)
_________________________________________________________________
"""
```
