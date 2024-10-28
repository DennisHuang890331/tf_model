import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.io import loadmat


@tf.keras.utils.register_keras_serializable()
class Convolution_Block(tf.keras.layers.Layer):

    def __init__(self, num_filters=256, kernel_size=3, dilation_rate=1, padding='same', use_bias=False
                 , trainable=True, separableconv=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.use_bias = use_bias
        self.separableconv = separableconv
    
    def build(self, input_shape):
        if self.separableconv:
            self.conv = tf.keras.layers.SeparableConv2D(
                filters=self.num_filters,
                kernel_size=self.kernel_size,
                dilation_rate=self.dilation_rate,
                padding=self.padding,
                use_bias=self.use_bias,
                depthwise_initializer=tf.keras.initializers.HeNormal(),
                pointwise_initializer=tf.keras.initializers.HeNormal(),
            )
        else:
            self.conv = tf.keras.layers.Conv2D(
                filters=self.num_filters, 
                kernel_size=self.kernel_size,
                dilation_rate=self.dilation_rate,
                padding=self.padding,
                use_bias=self.use_bias,
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )
        self.BatchNormal = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.BatchNormal(x)
        x = tf.keras.activations.selu(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "dilation_rate": self.dilation_rate,
                "padding": self.padding,
                "use_bias": self.use_bias,
                "separableconv": self.separableconv
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class DilatedSpatialPyramidPooling(tf.keras.layers.Layer):

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def build(self, input_shape):
        self.conv1 = Convolution_Block(kernel_size=1, use_bias=True)
        self.conv2 = Convolution_Block(kernel_size=1, dilation_rate=1)
        self.conv3 = Convolution_Block(kernel_size=3, dilation_rate=6)
        self.conv4 = Convolution_Block(kernel_size=3, dilation_rate=12)
        self.conv5 = Convolution_Block(kernel_size=3, dilation_rate=18)
        self.conv6 = Convolution_Block(kernel_size=1)

    
    def call(self, inputs):
        dims = inputs.shape
        x = tf.keras.layers.AveragePooling2D(
            pool_size=(dims[-3], dims[-2])
        )(inputs)
        x = self.conv1(x)
        outpool = tf.keras.layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
            interpolation='bilinear'
        )(x)

        out1 = self.conv2(inputs)
        out6 = self.conv3(inputs)
        out12 = self.conv4(inputs)
        out18 = self.conv5(inputs)
        x = tf.keras.layers.Concatenate()([outpool, out1, out6, out12, out18])
        output = self.conv6(x)

        return output
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Visualization:

    def __init__(self) -> None:
        self.colormap = loadmat(
            'Dataset/CIHP/instance-level_human_parsing/human_colormap.mat'
        )['colormap']
        self.colormap = self.colormap * 100
        self.colormap = self.colormap.astype(np.uint8)

    def decode_segmetation_mask(self, mask, num_classes):
        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        for l in range(0, num_classes):
            idx = mask == l
            r[idx] = self.colormap[l, 0]
            g[idx] = self.colormap[l, 1]
            b[idx] = self.colormap[l, 2]
        rgb = np.stack([r, g, b], axis=2)
        return rgb

    def get_overlay(self, image, colored_mask):
        image = tf.keras.utils.array_to_img(image)
        image = np.array(image).astype(np.uint8)
        overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
        return overlay

    def plot_samples_matplotlib(self, display_list, figsize=(5, 3)):
        _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
        for i in range(len(display_list)):
            if display_list[i].shape[-1] == 3:
                axes[i].imshow(tf.keras.utils.array_to_img(display_list[i]))
            else:
                axes[i].imshow(display_list[i])
        plt.show()


def build_model(image_shape=(512, 512, 3), num_classes=20, 
                backbone='resnet50', rate_dropout=0.1,
                backbone_trainable=False):
    tf.keras.backend.clear_session()
    model_input = tf.keras.Input(shape=image_shape)

    if backbone == 'resnet50':
        backbone = tf.keras.applications.ResNet50(
            # weights="Checkpoints/resnets/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", 
            include_top=False, input_tensor=model_input        
        )
        x = backbone.get_layer("conv4_block6_2_relu").output
        inputb = backbone.get_layer("conv2_block3_2_relu").output
        x = DilatedSpatialPyramidPooling()(x)
        inputa = tf.keras.layers.UpSampling2D(
            size=(image_shape[0] // 4 // x.shape[1], image_shape[1] // 4 // x.shape[2]),
            interpolation='bilinear'
        )(x)
        
        inputb = Convolution_Block(num_filters=48, kernel_size=1)(inputb)

        x = tf.keras.layers.Concatenate(axis=-1)([inputa, inputb])
        x = Convolution_Block()(x)
        x = Convolution_Block()(x)
        x = tf.keras.layers.UpSampling2D(
            size=(image_shape[0] // x.shape[1], image_shape[1] // x.shape[2]),
            interpolation='bilinear'
        )(x)

        model_output = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1), padding='same')(x)
    elif backbone == 'inceptionresnetv2':
        backbone = tf.keras.applications.InceptionResNetV2(
            # weights='Checkpoints/inceptionresnetv2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False, input_tensor=model_input
        )

        low_level_feature = backbone.get_layer('activation_4').output
        for _ in range(2):
            low_level_feature = Convolution_Block()(low_level_feature)
            low_level_feature = tf.keras.layers.Conv2DTranspose(
                filters=256, kernel_size=3,
                kernel_initializer=tf.keras.initializers.HeNormal()
            )(low_level_feature)
        low_level_feature = Convolution_Block(kernel_size=1)(low_level_feature)
        if rate_dropout != 0:
            low_level_feature = tf.keras.layers.SpatialDropout2D(rate=rate_dropout)(low_level_feature)

        encoder = backbone.get_layer('block17_10_ac').output
        encoder = DilatedSpatialPyramidPooling()(encoder)
        decoder = tf.keras.layers.Conv2DTranspose(
            filters=256, kernel_size=3, kernel_initializer=tf.keras.initializers.HeNormal()
        )(encoder)

        decoder = tf.keras.layers.UpSampling2D(
            size=(4, 4), interpolation='bilinear'
        )(decoder)
        if rate_dropout != 0:
            decoder = tf.keras.layers.SpatialDropout2D(rate=rate_dropout)(decoder)

        x = tf.keras.layers.Concatenate()([low_level_feature, decoder])

        for _ in range(2):
            x = Convolution_Block()(x)
            if rate_dropout != 0:
                x = tf.keras.layers.SpatialDropout2D(rate=rate_dropout)(x)
        x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

        downsample = model_input
        downsample = Convolution_Block(num_filters=8, separableconv=False)(downsample)
        for _ in range(2):
            downsample = Convolution_Block(num_filters=8)(downsample)
        if rate_dropout != 0:
            downsample = tf.keras.layers.SpatialDropout2D(rate=rate_dropout)(downsample)

        x = tf.keras.layers.Concatenate()([downsample, x])

        for _ in range(2):
            x = Convolution_Block(num_filters=64)(x)
        
        model_output = tf.keras.layers.Conv2D(
            num_classes, kernel_size=(1, 1), padding='same')(x)

    if not backbone_trainable:
        for layer in backbone.layers:
            layer.trainable = False
    
    
    return tf.keras.Model(model_input, model_output, name='DeeplabV3Plus')



