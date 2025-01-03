import tensorflow as tf
import keras
import numpy as np
from .utils import DecodePredictions

""" Yolov7Backbone """
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.97


keras.utils.register_keras_serializable()
class ConcatStack(keras.layers.Layer): 

    def __init__(self, filters, concats=[-1, -3, -5, -6], depth=6, mid_ratio=1.0, out_channels=-1, use_additional_stack=False, activation="swish",**kwargs):
        super().__init__(**kwargs)
        concats = concats if concats is not None else [-(ii + 1) for ii in range(depth)]
        self.filters, self.concats, self.depth, self.mid_ratio, self.out_channels, self.use_additional_stack, self.activation \
            = int(filters), concats, depth, mid_ratio, out_channels, use_additional_stack, activation
        self.mid_filters = int(self.mid_ratio * self.filters)
        chennel_map = [self.filters] * 2 + [self.mid_filters] * (self.depth - 2)
        if self.out_channels > 0:
            self.out_channels = self.out_channels
        else:
            self.out_channels = 0
            for index in self.concats:
                self.out_channels += chennel_map[index]
        self.out_conv = keras.Sequential([
            keras.layers.Conv2D(int(self.out_channels), 1, 1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
            keras.layers.Activation('swish'),
        ])

        self.stack = self.build_stack()
        if use_additional_stack:
            self.another_stack = self.build_stack()
    
    def build_stack(self):
        conv_list= []
        first = keras.Sequential([
            keras.layers.Conv2D(self.filters, 1, 1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
            keras.layers.Activation(self.activation),
        ])
        conv_list.append(first)
        second = keras.Sequential([
            keras.layers.Conv2D(self.filters, 1, 1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
            keras.layers.Activation(self.activation),
        ])
        conv_list.append(second)

        for _ in range(self.depth - 2):
            conv_list.append(keras.Sequential([
                keras.layers.Conv2D(self.mid_filters, 3, 1, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(self.activation),
            ]))
        
        
        return conv_list

    
    def call_stack(self, stack, inputs):
        first = stack[0](inputs)
        second = stack[1](inputs)
        gathered = [first, second]
        for conv in stack[2:]:
            gathered.append(conv(gathered[-1]))
        x = tf.concat([gathered[index] for index in self.concats], axis=-1)
        x = self.out_conv(x)
        return x

        
    def call(self, inputs):
        x = self.call_stack(self.stack, inputs)
        if self.use_additional_stack:
            x = x + self.call_stack(self.another_stack, inputs)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'concats': self.concats,
            'depth': self.depth,
            'mid_ratio': self.mid_ratio,
            'out_channels': self.out_channels,
            'use_additional_stack': self.use_additional_stack,
            'activation': self.activation,
        })
        return config
        
keras.utils.register_keras_serializable()
class CSPDownsample(keras.layers.Layer):
    
    def __init__(self, ratio=0.5, activation="swish", **kwargs):
        super().__init__(**kwargs)
        self.ratio, self.activation = ratio, activation

    def build(self, input_shape):
        input_channel = input_shape[-1]
        hidden_ratio, out_ratio = self.ratio if isinstance(self.ratio, (list, tuple)) else (self.ratio, self.ratio)
        hidden_channel, self.out_channel = int(input_channel * hidden_ratio), int(input_channel * out_ratio)
        
        if self.out_channel == 0:
            self.pool_branch = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        else:
            self.pool_branch = keras.Sequential([
                keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
                keras.layers.Conv2D(self.out_channel, 1, 1, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(self.activation),
            ])
            self.conv_branch = keras.Sequential([
                keras.layers.Conv2D(hidden_channel, 1, 1, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(self.activation),
                keras.layers.Conv2D(self.out_channel, 3, 2, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(self.activation),
            ])

    def call(self, inputs):
        pool_branch = self.pool_branch(inputs)
        if self.out_channel == 0:
            x = pool_branch
        else:
            conv_branch = self.conv_branch(inputs)
            x = tf.concat([conv_branch, pool_branch], axis=-1)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'ratio': self.ratio,
            'activation': self.activation,
        })
        return config

# Almost same with yolor, just supporting YOLOV7_Tiny with depth=1
keras.utils.register_keras_serializable()
class ResSpatialPyramidPooling(keras.layers.Layer):

    def __init__(self, depth=2, expansion=0.5, pool_sizes=(5, 9, 13), activation="swish", **kwargs):
        super().__init__(**kwargs)
        self.depth, self.expansion, self.pool_sizes, self.activation = depth, expansion, pool_sizes, activation

    def build(self, input_shape):
        input_channel = input_shape[-1]
        hidden_channel = int(input_channel * self.expansion)
        self.short = keras.Sequential([
            keras.layers.Conv2D(hidden_channel, 1, 1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
            keras.layers.Activation(self.activation),
        ])
        if self.depth > 1:
            deep_base = [
                keras.layers.Conv2D(hidden_channel, 1, 1, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(self.activation),
                keras.layers.Conv2D(hidden_channel, 3, 1, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(self.activation),
                keras.layers.Conv2D(hidden_channel, 1, 1, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(self.activation),
            ]
        else:
            deep_base = [
                keras.layers.Conv2D(hidden_channel, 1, 1, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(self.activation),
            ]

        self.deep_base = keras.Sequential(deep_base)

        self.pool = [keras.layers.MaxPool2D(pool_size=pool_size, strides=1, padding='same') for pool_size in self.pool_sizes]

        deep = []
        for _ in range(self.depth - 1):
            deep.extend([
                keras.layers.Conv2D(hidden_channel, 1, 1, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(self.activation),
                keras.layers.Conv2D(hidden_channel, 3, 1, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(self.activation),
            ])
        if self.depth == 1:
            deep.extend([
                keras.layers.Conv2D(hidden_channel, 1, 1, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(self.activation),
            ])
        self.deep = keras.Sequential(deep)
        self.out = keras.Sequential([
            keras.layers.Conv2D(hidden_channel, 1, 1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
            keras.layers.Activation(self.activation),
        ])
    
    def call(self, inputs):
        short = self.short(inputs)
        deep = self.deep_base(inputs)
        concat = [deep]
        for pool in self.pool:
            deep = pool(deep)
            concat.append(deep)
        deep = tf.concat(concat, axis=-1)
        deep = self.deep(deep)
        out = tf.concat([deep, short], axis=-1)
        out = self.out(out)
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'depth': self.depth,
            'expansion': self.expansion,
            'pool_sizes': self.pool_sizes,
            'activation': self.activation,
        })
        return config

# Same with yolor
keras.utils.register_keras_serializable()
class FocusStem(keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding="valid", activation="swish", **kwargs):
        super().__init__(**kwargs)
        self.filters, self.kernel_size, self.strides, self.padding, self.activation = filters, kernel_size, strides, padding, activation
        self.conv = keras.Sequential([
            keras.layers.Conv2D(filters, kernel_size, strides, padding=padding, use_bias=False),
            keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
            keras.layers.Activation(activation),
        ])
   
    def call(self, inputs):
        if self.padding.lower() == "same":
            inputs = tf.pad(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]])
            patch_top_left = inputs[:, :-1:2, :-1:2]
            patch_top_right = inputs[:, :-1:2, 1::2]
            patch_bottom_left = inputs[:, 1::2, :-1:2]
            patch_bottom_right = inputs[:, 1::2, 1::2]
        else:
            patch_top_left = inputs[:, ::2, ::2]
            patch_top_right = inputs[:, ::2, 1::2]
            patch_bottom_left = inputs[:, 1::2, ::2]
            patch_bottom_right = inputs[:, 1::2, 1::2]
        nn = tf.concat([patch_top_left, patch_bottom_left, patch_top_right, patch_bottom_right], axis=-1)
        nn = self.conv(nn)
        return nn
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
        })
        return config

keras.utils.register_keras_serializable()
class Backbone(keras.layers.Layer):

    def __init__(
        self, 
        channels=[64, 128, 256, 256],
        stack_concats=[-1, -3, -5, -6],
        stack_depth=6,
        stack_out_ratio=1.0,
        use_additional_stack=False,
        stem_width=-1,  # -1 means using channels[0]
        stem_type="conv3",  # One of ["conv3", "focus", "conv1"], "focus" for YOLOV7_*6 models, "conv1" for YOLOV7_Tiny
        csp_downsample_ratios=[0, 0.5, 0.5, 0.5],
        out_features=[-3, -2, -1],
        spp_depth=2,
        activation="swish",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels, self.stack_concats, self.stack_depth = channels, stack_concats, stack_depth
        self.stack_out_ratio, self.use_additional_stack = stack_out_ratio, use_additional_stack
        self.stem_width, self.stem_type = stem_width, stem_type
        self.csp_downsample_ratios, self.out_features = csp_downsample_ratios, out_features
        self.spp_depth, self.activation = spp_depth, activation

        """ Stem """
        stem_width = stem_width if stem_width > 0 else channels[0]
        if stem_type == "focus":
            self.stem = FocusStem(stem_width, activation=activation, name="stem_")
        elif stem_type == "conv1":
            self.stem = keras.Sequential([
                keras.layers.Conv2D(stem_width, 3, 2, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(activation),
            ])
        else:
            self.stem = keras.Sequential([
                keras.layers.Conv2D(stem_width // 2, 3, 1, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(activation),
                keras.layers.Conv2D(stem_width, 3, 2, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(activation),
                keras.layers.Conv2D(stem_width, 3, 1, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(activation),
            ])

        common_kwargs = {
        "concats": stack_concats,
        "depth": stack_depth,
        "mid_ratio": 1.0,
        "use_additional_stack": use_additional_stack,
        "activation": activation,
        }

        """ blocks """
        self.network_blocks = []
        for id, (channel, csp_downsample_ratio) in enumerate(zip(channels, csp_downsample_ratios)):
            network_block = []
            stack_name = "stack{}_".format(id + 1)
            if isinstance(csp_downsample_ratio, (list, tuple)) or 0 < csp_downsample_ratio <= 1:
                network_block.append(CSPDownsample(ratio=csp_downsample_ratio, activation=activation, name=stack_name + "downsample_"))
            else:
                ds_channels = stem_width * 2 if csp_downsample_ratio <= 0 else csp_downsample_ratio
                network_block.extend([
                    keras.layers.Conv2D(ds_channels, 3, 2, padding='same', use_bias=False),
                    keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                    keras.layers.Activation(activation),
                ])
            
            out_channels = -1 if stack_out_ratio == 1 else int(channel * len(stack_concats) * stack_out_ratio)
            network_block.append(ConcatStack(channel, **common_kwargs, out_channels=out_channels, name=stack_name))

            if id == len(channels) - 1:
                # add SPPCSPC block if it's the last stack
                network_block.append(ResSpatialPyramidPooling(depth=spp_depth, activation=activation, name=stack_name + "spp_"))
            self.network_blocks.append(keras.Sequential(network_block))

    def call(self, inputs):
        features = []
        x = self.stem(inputs)
        features.append(x)
        for block in self.network_blocks:
            x = block(x)
            features.append(x)
        outputs = [features[index] for index in self.out_features]
        return outputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'stack_concats': self.stack_concats,
            'stack_depth': self.stack_depth,
            'stack_out_ratio': self.stack_out_ratio,
            'use_additional_stack': self.use_additional_stack,
            'stem_width': self.stem_width,
            'stem_type': self.stem_type,
            'csp_downsample_ratios': self.csp_downsample_ratios,
            'out_features': self.out_features,
            'spp_depth': self.spp_depth,
            'activation': self.activation,
        })
        return config

""" path aggregation fpn, using `ConcatStack` instead of `csp_stack` from yolor """
keras.utils.register_keras_serializable()
class UpSampleMerge(keras.layers.Layer):
    def __init__(self, hidden_channels, mid_ratio=0.5, concats=None, depth=6, use_additional_stack=False, activation="swish", **kwargs):
        super().__init__(**kwargs)
        self.hidden_channels, self.mid_ratio, self.concats, self.depth = hidden_channels, mid_ratio, concats, depth
        self.use_additional_stack, self.activation = use_additional_stack, activation
    
    def build(self, input_shape):
        _, H, W, _ = input_shape[0]
        self.upsample = keras.Sequential([
            keras.layers.Conv2D(input_shape[0][-1], 1, 1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
            keras.layers.Activation(self.activation),
            keras.layers.Resizing(H, W, interpolation="nearest"),
        ])
        out_channels = 0
        for shape in input_shape[1::]:
            out_channels += shape[-1]
        self.hidden_channels = self.hidden_channels if self.hidden_channels > 0 else out_channels
        self.concat = ConcatStack(self.hidden_channels, self.concats, self.depth, self.mid_ratio, \
            out_channels, self.use_additional_stack, self.activation, name="up_")

    def call(self, inputs):
        upsample = self.upsample(inputs[0])
        inputs[-1] = upsample
        x = tf.concat(inputs, axis=-1)
        x = self.concat(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_channels': self.hidden_channels,
            'mid_ratio': self.mid_ratio,
            'concats': self.concats,
            'depth': self.depth,
            'use_additional_stack': self.use_additional_stack,
            'activation': self.activation,
        })
        return config

keras.utils.register_keras_serializable()
class DownSampleMerge(keras.layers.Layer):
    def __init__(self, hidden_channels, mid_ratio=0.5, concats=None, depth=6, csp_downsample_ratio=1, \
        use_additional_stack=False, activation="swish", **kwargs):
        super().__init__(**kwargs)
        self.hidden_channels, self.mid_ratio, self.concats, self.depth = hidden_channels, mid_ratio, concats, depth
        self.csp_downsample_ratio, self.use_additional_stack, self.activation = csp_downsample_ratio, use_additional_stack, activation

    def build(self, input_shape):
        if isinstance(self.csp_downsample_ratio, (list, tuple)) or self.csp_downsample_ratio > 0:
            self.csp_downsample = CSPDownsample(ratio=self.csp_downsample_ratio, activation=self.activation, name="downsample_")
            out_channels = input_shape[0][-1] * self.csp_downsample_ratio if not isinstance(self.csp_downsample_ratio, (list, tuple)) \
                else input_shape[0][-1] * self.csp_downsample_ratio[-1]
        else:
            hidden_channels = input_shape[-1][-1]
            self.csp_downsample = keras.Sequential([
                keras.layers.Conv2D(hidden_channels, 3, 2, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(self.activation),
            ])
            out_channels = hidden_channels
        for shape in input_shape[1::]:
            out_channels += shape[-1]
        out_channels = out_channels // 2
        hidden_channels = self.hidden_channels if self.hidden_channels > 0 else out_channels
        self.concat = ConcatStack(hidden_channels, self.concats, self.depth, self.mid_ratio, out_channels, \
            self.use_additional_stack, self.activation, name="down_")
    
    def call(self, inputs):
        inputs[0] = self.csp_downsample(inputs[0])
        x = tf.concat(inputs, axis=-1)
        x = self.concat(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_channels': self.hidden_channels,
            'mid_ratio': self.mid_ratio,
            'concats': self.concats,
            'depth': self.depth,
            'csp_downsample_ratio': self.csp_downsample_ratio,
            'use_additional_stack': self.use_additional_stack,
            'activation': self.activation,
        })
        return config

keras.utils.register_keras_serializable()
class PathAggregationFPN(keras.layers.Layer):
    # yolov7                                                        # yolov7_w6
    # 51: p5 512 ---+---------------------+-> 101: out2 512         # 47: p5 512 ---┬---------------------┬-> 113: out 512
    #               v [up 256 -> concat]  ^ [down 512 -> concat]    #               ↓ [up 384 -> concat]  ↑[down 512 -> concat]
    # 37: p4 1024 -> 63: p4p5 256 -------> 88: out1 256             # 37: p4 768 --- 59: p4p5 384 ------- 103: out 384
    #               v [up 128 -> concat]  ^ [down 256 -> concat]    #               ↓ [up 256 -> concat]  ↑[down 384 -> concat]
    # 24: p3 512 --> 75: p3p4p5 128 ------+--> 75: out0 128         # 28: p3 512 --- 71: p3p4p5 256 -- 93: out 256
    #                                                               #               ↓ [up 128 -> concat]  ↑[down 256 -> concat]
    #                                                               # 19: p2 256 --- 83: p2p3p4p5 128 -----┴-> 83: out 128
    # features: [p3, p4, p5]
    def __init__(self, hidden, mid_ratio=0.5, channel_ratio=0.25, concats=None, depth=6, \
        csp_downsample_ratio=1, use_additional_stack=False, activation="swish", **kwargs):
        super().__init__(**kwargs)
        self.hidden, self.mid_ratio, self.channel_ratio, self.concats, self.depth = hidden, mid_ratio, channel_ratio, concats, depth
        self.csp_downsample_ratio, self.use_additional_stack, self.activation = csp_downsample_ratio, use_additional_stack, activation
    
    def build(self, input_shape):
        hidden_channels = self.hidden.copy() if isinstance(self.hidden, list) else self.hidden
        p_name = "p{}_".format(len(input_shape) + 2)
        self.upsamples = []
        self.upsample_merges = []
        for index, shape in enumerate(input_shape[:-1][::-1]):
            cur_p_name = "p{}".format(len(input_shape) + 1 - index)
            hidden_channel = hidden_channels.pop(0) if isinstance(hidden_channels, list) else hidden_channels
            p_name = cur_p_name + p_name
            upsample = keras.Sequential([
                keras.layers.Conv2D(int(shape[-1] * self.channel_ratio), 1, 1, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(self.activation),
            ])
            
            self.upsamples.append(upsample)
            self.upsample_merges.append(
                UpSampleMerge(int(hidden_channel), self.mid_ratio, self.concats, self.depth, self.use_additional_stack, self.activation, name=p_name)
            )

        self.downsample_merges = []
        for index, shape in enumerate(input_shape[:-1][::-1]):
            cur_name = "c3n{}_".format(index + 3)
            hidden_channel = hidden_channels.pop(0) if isinstance(hidden_channels, list) else hidden_channels
            cur_csp_downsample_ratio = self.csp_downsample_ratio.pop(0) if isinstance(self.csp_downsample_ratio, list) \
                else self.csp_downsample_ratio
            downsample = DownSampleMerge(int(hidden_channel), self.mid_ratio, self.concats, self.depth, cur_csp_downsample_ratio, \
                self.use_additional_stack, self.activation, name=cur_name)

            self.downsample_merges.append(downsample)

    def call(self, inputs):
        upsamples = [inputs[-1]]
        for id, x in enumerate(inputs[:-1][::-1]):
            x = self.upsamples[id](x)
            x = self.upsample_merges[id]([x, upsamples[-1]])
            upsamples.append(x)
            
        downsamples = [upsamples[-1]]
        for id, x in enumerate(upsamples[:-1][::-1]):
            x = self.downsample_merges[id]([downsamples[-1], x])
            downsamples.append(x)
        return downsamples
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden': self.hidden,
            'mid_ratio': self.mid_ratio,
            'channel_ratio': self.channel_ratio,
            'concats': self.concats,
            'depth': self.depth,
            'csp_downsample_ratio': self.csp_downsample_ratio,
            'use_additional_stack': self.use_additional_stack,
            'activation': self.activation,
        })
        return config

@keras.utils.register_keras_serializable()
class RepCNN(keras.layers.Layer):
    
    def __init__(self, input_dim, out_dim, activation=None, kernel_size=3, strides=1, padding='SAME', groups=1, deploy=False, **kwargs):
        super().__init__(**kwargs)
        self.deploy = deploy
        self.groups = groups
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.reparam_kernel = None
        self.reparam_biases = None
        if activation is not None:
            self.act = keras.layers.Activation(activation)
        else:
            self.act = keras.layers.Identity()
        
    def build(self, input_shape):
        
        self.rbr_identity = keras.models.Sequential([
            keras.layers.BatchNormalization(name='batch_normal')
        ]) if self.input_dim == self.out_dim and self.strides == 1 else None
        if self.rbr_identity: self.rbr_identity.build(input_shape)
        
        self.rbr_dense = keras.models.Sequential([
            keras.layers.Conv2D(
                self.out_dim, self.kernel_size, self.strides, self.padding,
                groups=self.groups, use_bias=False, name='conv2d'
            ),
            keras.layers.BatchNormalization(name='batch_normal')
        ])
        self.rbr_dense.build(input_shape)

        self.rbr_1x1 = keras.models.Sequential([
            keras.layers.Conv2D(
                self.out_dim, 1, self.strides, self.padding,
                groups=self.groups, use_bias=False, name='conv2d'
            ),
            keras.layers.BatchNormalization(name='batch_normal')
        ])
        self.rbr_1x1.build(input_shape)

    def _fuse_bn_tensor(self, sequence):
        if sequence is None:
            return 0, 0

        layer_names = [layer.name for layer in sequence.layers]
        if 'conv2d' in layer_names:
            batch_normal = sequence.get_layer('batch_normal')
            conv2d = sequence.get_layer('conv2d')
            kernel = conv2d.get_weights()[0] # weight_shape = (kernal_size 1, kernal_size 2, input_dim, output_dim)
            gamma, beta, moving_mean, moving_var = batch_normal.get_weights() # shape = (output_dim)
            eps = batch_normal.get_config()['epsilon']
        else:
            batch_normal = sequence.get_layer('batch_normal')
            if not hasattr(self, "id_tensor"):
                dim = self.input_dim // self.groups
                self.id_tensor = np.zeros((self.kernel_size, self.kernel_size, dim, dim), dtype=np.float32)
                center = self.kernel_size // 2
                for c in range(dim):
                    self.id_tensor[center, center, c, c] = 1
            kernel = self.id_tensor
            gamma, beta, moving_mean, moving_var = batch_normal.get_weights()
            eps = batch_normal.get_config()['epsilon']

        std = tf.sqrt((moving_var + eps))
        t = tf.reshape((gamma / std), (1, 1, 1, -1))
        return kernel * t, beta - moving_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        kernel1x1 = tf.pad(kernel1x1, [[1, 1], [1, 1], [0, 0], [0, 0]])
        return (
            kernel3x3 + kernel1x1 + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def fuse_repvgg_block(self):    
        if self.deploy:
            return

        self.deploy = True
        kernel, baises = self.get_equivalent_kernel_bias()
        # self.reparam_kernel = self.add_weight('reparam_kernel', kernel.shape, initializer=tf.constant_initializer(kernel.numpy()), trainable=False)
        # self.reparam_biases = self.add_weight('reparam_biases', baises.shape, initializer=tf.constant_initializer(baises.numpy()), trainable=False)
        self.reparam_biases = tf.Variable(baises, name='reparam_biases', trainable=False)
        self.reparam_kernel = tf.Variable(kernel, name='reparam_kernel', trainable=False)

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None

        self.rbr_reparam = lambda input: tf.nn.bias_add(
            tf.nn.conv2d(input, self.reparam_kernel, self.strides, self.padding), self.reparam_biases
        )

            
    def call(self, inputs):
        if hasattr(self, "rbr_reparam"): return self.act(self.rbr_reparam(inputs))
        id_out = 0 if self.rbr_identity is None else self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "deploy": self.deploy,
                "groups": self.groups,
                "input_dim": self.input_dim,
                "out_dim": self.out_dim,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "activation": self.activation,
            }
        )
        return config

""" YOLOV7Head, using Reparam Conv block """
keras.utils.register_keras_serializable()
class HeadSingle(keras.layers.Layer):
    def __init__(self, filters, use_reparam_conv_head=True, num_classes=80, regression_len=4, num_anchors=3, use_object_scores=True, activation="swish", **kwargs):
        super().__init__(**kwargs)
        self.filters, self.use_reparam_conv_head, self.num_classes = filters, use_reparam_conv_head, num_classes
        self.regression_len, self.num_anchors, self.use_object_scores, self.activation = regression_len, num_anchors, use_object_scores, activation
        if use_reparam_conv_head:
            self.rep_conv = RepCNN(filters, 3, activation=activation)
        else:
            self.conv = keras.Sequential([
                keras.layers.Conv2D(filters, 3, 1, padding='same', use_bias=False),
                keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM),
                keras.layers.Activation(activation),
            ])
        ouput_classes = num_classes + regression_len + (1 if use_object_scores else 0)
        self.output_conv = keras.layers.Conv2D(ouput_classes * num_anchors, 1, name="output_conv")
        self.reshape = keras.layers.Reshape([-1, ouput_classes], name="output_reshape")
        
    def deploy(self):
        if self.use_reparam_conv_head:
            self.rep_conv.fuse_repvgg_block()
    
    def call(self, inputs):
        x = self.rep_conv(inputs) if self.use_reparam_conv_head else self.conv(inputs)
        x = self.reshape(self.output_conv(x))
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "use_reparam_conv_head": self.use_reparam_conv_head,
                "num_classes": self.num_classes,
                "regression_len": self.regression_len,
                "num_anchors": self.num_anchors,
                "use_object_scores": self.use_object_scores,
                "activation": self.activation,
            }
        )
        return config

keras.utils.register_keras_serializable()
class YOLOV7Head(keras.layers.Layer):
    def __init__(self, use_reparam_conv_head=True, num_classes=80, regression_len=4, num_anchors=3,
                 use_object_scores=True, activation="swish", classifier_activation="sigmoid",**kwargs):
        super().__init__(**kwargs)
        self.use_reparam_conv_head, self.num_classes = use_reparam_conv_head, num_classes
        self.regression_len, self.num_anchors, self.use_object_scores = regression_len, num_anchors, use_object_scores
        self.activation, self.classifier_activation = activation, classifier_activation
        self.activation = keras.layers.Activation(classifier_activation)
    
    def build(self, input_shape):
        self.heads = []
        for shape in input_shape:
            filters = int(shape[-1] * 2)
            self.heads.append(
                HeadSingle(filters, self.use_reparam_conv_head, self.num_classes, self.regression_len, self.num_anchors, self.use_object_scores, self.activation)
            )
    def call(self, inputs):
        outputs = []
        for id, input in enumerate(inputs):
            outputs.append(self.heads[id](input))
        outputs = tf.concat(outputs, axis=1)
        return self.activation(outputs)

    def deploy(self):
        for head in self.heads:
            head.deploy()
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "use_reparam_conv_head": self.use_reparam_conv_head,
                "num_classes": self.num_classes,
                "regression_len": self.regression_len,
                "num_anchors": self.num_anchors,
                "use_object_scores": self.use_object_scores,
                "activation": self.activation,
                "classifier_activation": self.classifier_activation,
            }
        )
        return config
    


keras.utils.register_keras_serializable()
class YOLOV7(keras.Model):
    def __init__(
        self, 
        csp_channels=[64, 128, 256, 256], 
        stack_concats=[-1, -3, -5, -6], 
        stack_depth=6, 
        stack_out_ratio=1.0, 
        use_additional_stack=False, 
        stem_width=-1, 
        stem_type="conv3", 
        csp_downsample_ratios=[0, 0.5, 0.5, 0.5], 
        spp_depth=2, 
        fpn_hidden_channels=[256, 128, 256, 512], 
        fpn_channel_ratio=0.25, 
        fpn_stack_concats=None, 
        fpn_stack_depth=-1, 
        fpn_mid_ratio=0.5, 
        fpn_csp_downsample_ratio=1, 
        use_reparam_conv_head=True, 
        features_pick=[-3, -2, -1], 
        regression_len=4, 
        anchors_mode="yolor", 
        num_anchors="auto", 
        use_object_scores="auto", 
        image_shape=(640, 640, 3), 
        num_classes=80, 
        activation="swish", 
        classifier_activation="sigmoid", 
        freeze_backbone=False, 
        pretrained=None,
        model_name="yolov7", 
        pyramid_levels_min=3, 
        anchor_scale="auto", 
        rescale_mode="raw01", 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.csp_channels, self.stack_concats = csp_channels, stack_concats
        self.stack_depth, self.stack_out_ratio, self.use_additional_stack = stack_depth, stack_out_ratio, use_additional_stack
        self.stem_width, self.stem_type, self.csp_downsample_ratios = stem_width, stem_type, csp_downsample_ratios
        self.spp_depth, self.fpn_hidden_channels, self.fpn_channel_ratio = spp_depth, fpn_hidden_channels, fpn_channel_ratio
        self.fpn_stack_concats, self.fpn_stack_depth, self.fpn_mid_ratio = fpn_stack_concats, fpn_stack_depth, fpn_mid_ratio
        self.fpn_csp_downsample_ratio, self.use_reparam_conv_head = fpn_csp_downsample_ratio, use_reparam_conv_head
        self.features_pick, self.regression_len, self.anchors_mode = features_pick, regression_len, anchors_mode
        self.num_anchors, self.use_object_scores, self.image_shape = num_anchors, use_object_scores, image_shape
        self.num_classes, self.activation, self.classifier_activation = num_classes, activation, classifier_activation
        self.freeze_backbone, self.pretrained, self.name = freeze_backbone, pretrained, model_name
        self.pyramid_levels_min, self.anchor_scale, self.rescale_mode = pyramid_levels_min, anchor_scale, rescale_mode
        
      
        csp_kwargs = {"out_features": features_pick, "spp_depth": spp_depth, "activation": activation}

        self.backbone = Backbone(csp_channels, stack_concats, stack_depth, stack_out_ratio, use_additional_stack, \
                                 stem_width, stem_type, csp_downsample_ratios, name='backbone', **csp_kwargs)
            

        self.backbone.trainable = False if freeze_backbone else True
        use_object_scores, num_anchors, anchor_scale = self.get_anchors_mode_parameters(anchors_mode, use_object_scores, num_anchors, anchor_scale)
        fpn_stack_depth = fpn_stack_depth if fpn_stack_depth > 0 else stack_depth
        fpn_kwargs = {"csp_downsample_ratio": fpn_csp_downsample_ratio, "use_additional_stack": use_additional_stack, "activation": activation, "name": "pafpn_"}
        self.fpn = PathAggregationFPN(fpn_hidden_channels, fpn_mid_ratio, fpn_channel_ratio, fpn_stack_concats, fpn_stack_depth, **fpn_kwargs)
        self.head = YOLOV7Head(use_reparam_conv_head, num_classes, regression_len, num_anchors, use_object_scores, activation, classifier_activation, name="head_")

        pyramid_levels = [pyramid_levels_min, pyramid_levels_min + len(features_pick) - 1]  # -> [3, 5]
        self.post_process = DecodePredictions(image_shape ,pyramid_levels, anchors_mode, use_object_scores, anchor_scale, regression_len=regression_len, name="post_process_")
        self._model = self._build_model()

    def _build_model(self):
        inputs = keras.layers.Input(shape=self.image_shape)
        features = self.backbone(inputs)
        features = self.fpn(features)
        outputs = self.head(features)
        return keras.Model(inputs, outputs, name=self.name)
        
    def summary(self, line_length=None, positions=None, print_fn=None, expand_nested=False, show_trainable=False, layer_range=None):
        return self._model.summary(line_length, positions, print_fn, expand_nested, show_trainable, layer_range)
    
    def call(self, inputs):
        outputs = self._model(inputs)
        outputs = self.post_process(outputs)
        return outputs

    
    def get_anchors_mode_parameters(self, anchors_mode, use_object_scores="auto", num_anchors="auto", anchor_scale="auto"):
        EFFICIENTDET_MODE = "efficientdet"
        ANCHOR_FREE_MODE = "anchor_free"
        YOLOR_MODE = "yolor"
        YOLOV8_MODE = "yolov8"
        NUM_ANCHORS = {ANCHOR_FREE_MODE: 1, YOLOV8_MODE: 1, YOLOR_MODE: 3, EFFICIENTDET_MODE: 9}

        if anchors_mode == ANCHOR_FREE_MODE:
            use_object_scores = True if use_object_scores == "auto" else use_object_scores
            num_anchors = NUM_ANCHORS[anchors_mode] if num_anchors == "auto" else num_anchors
        elif anchors_mode == YOLOR_MODE:
            use_object_scores = True if use_object_scores == "auto" else use_object_scores
            num_anchors = NUM_ANCHORS[anchors_mode] if num_anchors == "auto" else num_anchors
        elif anchors_mode == YOLOV8_MODE:
            use_object_scores = False if use_object_scores == "auto" else use_object_scores
            num_anchors = NUM_ANCHORS[anchors_mode] if num_anchors == "auto" else num_anchors
        else:
            use_object_scores = False if use_object_scores == "auto" else use_object_scores
            num_anchors = NUM_ANCHORS.get(anchors_mode, NUM_ANCHORS[EFFICIENTDET_MODE]) if num_anchors == "auto" else num_anchors
            anchor_scale = 4 if anchor_scale == "auto" else anchor_scale
        return use_object_scores, num_anchors, anchor_scale

    def deploy(self):
        self.head.deploy()  # fuse reparam conv block
        self._model.trainable = False
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "csp_channels": self.csp_channels,
                "stack_concats": self.stack_concats,
                "stack_depth": self.stack_depth,
                "stack_out_ratio": self.stack_out_ratio,
                "use_additional_stack": self.use_additional_stack,
                "stem_width": self.stem_width,
                "stem_type": self.stem_type,
                "csp_downsample_ratios": self.csp_downsample_ratios,
                "spp_depth": self.spp_depth,
                "fpn_hidden_channels": self.fpn_hidden_channels,
                "fpn_channel_ratio": self.fpn_channel_ratio,
                "fpn_stack_concats": self.fpn_stack_concats,
                "fpn_stack_depth": self.fpn_stack_depth,
                "fpn_mid_ratio": self.fpn_mid_ratio,
                "fpn_csp_downsample_ratio": self.fpn_csp_downsample_ratio,
                "use_reparam_conv_head": self.use_reparam_conv_head,
                "features_pick": self.features_pick,
                "regression_len": self.regression_len,
                "anchors_mode": self.anchors_mode,
                "num_anchors": self.num_anchors,
                "use_object_scores": self.use_object_scores,
                "input_shape": self.input_shape,
                "num_classes": self.num_classes,
                "activation": self.activation,
                "classifier_activation": self.classifier_activation,
                "freeze_backbone": self.freeze_backbone,
                "pretrained": self.pretrained,
                "model_name": self.model_name,
                "pyramid_levels_min": self.pyramid_levels_min,
                "anchor_scale": self.anchor_scale,
                "rescale_mode": self.rescale_mode,
            }
        )
        return config


def YOLOV7_Tiny(
    image_shape=(416, 416, 3),
    freeze_backbone=False,
    num_classes=80,
    activation=keras.layers.LeakyReLU(negative_slope=0.1),
    classifier_activation="sigmoid",
    **kwargs,
):
    # anchors_yolov7_tiny = np.array([[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]])
    # anchors_yolor = np.array([[12,16, 19,36, 40,28], [36,75, 76,55, 72,146], [142,110, 192,243, 459,401]])
    # anchors_yolov7_tiny == np.ceil((anchors_yolor * 416 / 512)).astype('int') [TODO]
    stem_type = "conv1"
    csp_channels = [32, 64, 128, 256]
    stack_concats = [-1, -2, -3, -4]
    stack_depth = 4
    stack_out_ratio = 0.5
    csp_downsample_ratios = [0, [0, 0], [0, 0], [0, 0]]  # First 0 for conv_bn downsmaple, others [0, 0] means maxpool
    spp_depth = 1

    fpn_hidden_channels = [64, 32, 64, 128]
    fpn_mid_ratio = 1.0
    fpn_channel_ratio = 0.5
    fpn_csp_downsample_ratio = [0, 0]  # [0, 0] means using conv_bn downsmaple
    use_reparam_conv_head = False

    local_vars = {k: v for k, v in locals().items() if k != 'kwargs'}

    return YOLOV7(**local_vars, model_name="yolov7_tiny", **kwargs)



def YOLOV7_CSP(image_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, classifier_activation="sigmoid", **kwargs):
    local_vars = {k: v for k, v in locals().items() if k != 'kwargs'}
    return YOLOV7(**local_vars, model_name="yolov7_csp", **kwargs)



def YOLOV7_X(image_shape=(640, 640, 3), freeze_backbone=False, num_classes=80, classifier_activation="sigmoid", **kwargs):
    stack_concats = [-1, -3, -5, -7, -8]
    stack_depth = 8
    stem_width = 80

    fpn_stack_concats = [-1, -3, -5, -7, -8]
    fpn_mid_ratio = 1.0
    use_reparam_conv_head = False
    local_vars = {k: v for k, v in locals().items() if k != 'kwargs'}
    return YOLOV7(**local_vars, model_name="yolov7_x", **kwargs)



def YOLOV7_W6(image_shape=(1280, 1280, 3), freeze_backbone=False, num_classes=80, model_name="yolov7_w6", classifier_activation="sigmoid", **kwargs):
    csp_channels = kwargs.pop("csp_channels", [64, 128, 256, 384, 512])
    features_pick = kwargs.pop("features_pick", [-4, -3, -2, -1])
    stem_type = kwargs.pop("stem_type", "focus")
    csp_downsample_ratios = kwargs.pop("csp_downsample_ratios", [128, 256, 512, 768, 1024])  # > 1 value means using conv_bn instead of csp_downsample
    stack_out_ratio = kwargs.pop("stack_out_ratio", 0.5)

    fpn_hidden_channels = kwargs.pop("fpn_hidden_channels", [384, 256, 128, 256, 384, 512])
    fpn_channel_ratio = kwargs.pop("fpn_channel_ratio", 0.5)
    fpn_csp_downsample_ratio = kwargs.pop("fpn_csp_downsample_ratio", 0)
    use_reparam_conv_head = kwargs.pop("use_reparam_conv_head", False)

    kwargs.pop("kwargs", None)  # From other YOLOV7_*6 models
    local_vars = {k: v for k, v in locals().items() if k != 'kwargs'}
    return YOLOV7(**local_vars, **kwargs)



def YOLOV7_E6(image_shape=(1280, 1280, 3), freeze_backbone=False, num_classes=80, model_name="yolov7_e6", classifier_activation="sigmoid", **kwargs):
    stack_concats = kwargs.pop("stack_concats", [-1, -3, -5, -7, -8])
    stack_depth = kwargs.pop("stack_depth", 8)
    stem_width = kwargs.pop("stem_width", 80)
    csp_downsample_ratios = kwargs.pop("csp_downsample_ratios", [1, 1, 1, [1, 480 / 640], [1, 640 / 960]])  # different from YOLOV7_W6

    fpn_mid_ratio = kwargs.pop("fpn_mid_ratio", 0.5)
    fpn_csp_downsample_ratio = kwargs.pop("fpn_csp_downsample_ratio", [1, [1, 240 / 320], [1, 320 / 480]])  # different from YOLOV7_W6

    kwargs.pop("kwargs", None)  # From YOLOV7_E6E / YOLOV7_D6
    local_vars = {k: v for k, v in locals().items() if k != 'kwargs'}
    return YOLOV7_W6(**local_vars, **kwargs)



def YOLOV7_D6(image_shape=(1280, 1280, 3), freeze_backbone=False, num_classes=80, model_name="yolov7_d6", classifier_activation="sigmoid", **kwargs):
    stack_concats = [-1, -3, -5, -7, -9, -10]
    stack_depth = 10
    stem_width = 96
    local_vars = {k: v for k, v in locals().items() if k != 'kwargs'}
    return YOLOV7_E6(**local_vars, **kwargs)



def YOLOV7_E6E(image_shape=(1280, 1280, 3), freeze_backbone=False, model_name="yolov7_e6e", num_classes=80, classifier_activation="sigmoid", **kwargs):
    use_additional_stack = True
    local_vars = {k: v for k, v in locals().items() if k != 'kwargs'}
    return YOLOV7_E6(**local_vars, **kwargs)
