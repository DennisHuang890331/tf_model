# Tensorflow Model Implementation

This repository is a practice for implementing deep learning models using the TensorFlow framework.

The following models are implemented:

## Segmentation

[DeepLabV3+](https://arxiv.org/abs/1802.02611)

## Vision Transformer

[Masked autoencoder](https://arxiv.org/abs/2111.06377)

[Vision Transformer](https://arxiv.org/abs/2010.11929)

[Swin Transformer](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)

[Neighborhood Attention Transformer](https://openaccess.thecvf.com/content/CVPR2023/papers/Hassani_Neighborhood_Attention_Transformer_CVPR_2023_paper.pdf)

## Re-Parameterization

[RepVGG](https://arxiv.org/abs/2101.03697)

## Detection models

[YOLOV7](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors_CVPR_2023_paper.pdf)

# How to use

### Create Python env

```
conda create --name tf_model python=3.11.5
conda activate tf_model
pip3 install -r requirements.txt
```

### DeepLabV3+

```
from tensorflow_model import deeplab_v3plus
backbone = 'resnet50' # or 'inceptionresnetv2'
model = deeplab_v3plus.build_model(backbone=backbone)
model.summary()
```

### Mask AutoEncoder

```
from tensorflow_model import masked_autoencoder
model = masked_autoencoder.build_model()
model.summary()
```

### Vision Transformer

```
from tensorflow_model import vision_transformer
model = vision_transformer.build_model()
model.summary()
```

### Swin Transformer

```
from tensorflow_model import swin_transformer
model = swin_transformer.build_model()
model.summary()
```

### NeighborHood Attention Transdormer
```
from tensorflow_model.neighborhood_attention_transformer import NAT_Base
model = NAT_Base(input_shape=(224, 224, 3), num_classes=1000)
model.summary()
```

### RepVGG

```
import tensorflow as tf

from tensorflow_model.repvgg import create_RepVGG_A0, repvgg_layer_convert

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
### YOLOV7
```
import tensorflow as tf

from tensorflow_model.yolov7 import YOLOV7

model = YOLOV7()
# Re-Parameterize for heads
model.deploy()

"""
```
# Acknowledgment
This repository partially references the following repos:
[Keras_cv_attention_models](https://github.com/leondgarse/keras_cv_attention_models?tab=readme-ov-file#keras_cv_attention_models)