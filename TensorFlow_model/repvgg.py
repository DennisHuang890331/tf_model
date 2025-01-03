import keras
import numpy as np
import tensorflow as tf
import copy

'''
RepVGG: Making VGG-Style ConvNets Great Again
Ding, Xiaohan, et al. "Repvgg: Making vgg-style convnets great again." 
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.
'''

@keras.utils.register_keras_serializable()
class RepVGGBlock(keras.layers.Layer):
    
    def __init__(self, input_dim, out_dim, activation=True, kernel_size=3, stride=1, padding='SAME', groups=1, deploy=False, **kwargs):
        super(RepVGGBlock, self).__init__(**kwargs)
        self.deploy = deploy
        self.groups = groups
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.strides = stride
        self.padding = padding
        self.activation = activation
        self.reparam_kernel = None
        self.reparam_biases = None
        if activation:
            self.act = keras.layers.Activation('selu')
        else:
            self.act = keras.layers.Identity()
        
        self.rbr_identity = keras.models.Sequential([
            keras.layers.BatchNormalization(name='batch_normal')
        ]) if input_dim == out_dim and stride == 1 else None
        if self.rbr_identity: self.rbr_identity.build((None, None, None, input_dim))

        self.rbr_dense = keras.models.Sequential([
            keras.layers.Conv2D(
                out_dim, kernel_size, stride, padding,
                groups=groups, use_bias=False, name='conv2d'
            ),
            keras.layers.BatchNormalization(name='batch_normal')
        ])
        self.rbr_dense.build((None, None, None, input_dim))

        self.rbr_1x1 = keras.models.Sequential([
            keras.layers.Conv2D(
                out_dim, 1, stride, padding,
                groups=groups, use_bias=False, name='conv2d'
            ),
            keras.layers.BatchNormalization(name='batch_normal')
        ])
        self.rbr_1x1.build((None, None, None, input_dim))

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
        print(f"RepConv.fuse_repvgg_block")
        self.deploy = True
        kernel, baises = self.get_equivalent_kernel_bias()
        self.reparam_kernel = self.add_weight('reparam_kernel', kernel.shape, 
                                              initializer=tf.constant_initializer(kernel.numpy()), trainable=False)
        self.reparam_biases = self.add_weight('reparam_biases', baises.shape, 
                                              initializer=tf.constant_initializer(baises.numpy()), trainable=False)

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
    
    def _get_custom_L2(self):
        K3 = self.rbr_dense.get_layer('conv2d').get_weights()[0]
        K1 = self.rbr_1x1.get_layer('conv2d').get_weights()[0]
        bn_layer = self.rbr_dense.get_layer('batch_normal')
        t3 = tf.reshape(bn_layer.get_weights()[0] / tf.sqrt((bn_layer.get_weights()[3] + bn_layer.get_config()['epsilon'])), (1, 1, 1, -1))
        bn_layer = self.rbr_1x1.get_layer('batch_normal')
        t1 = tf.reshape(bn_layer.get_weights()[0] / tf.sqrt((bn_layer.get_weights()[3] + bn_layer.get_config()['epsilon'])), (1, 1, 1, -1))

        l2_loss_circle = tf.reduce_sum(K3 ** 2) - tf.reduce_sum(K3[:, :, 1:2, 1:2] ** 2)     # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = tf.reduce_sum(eq_kernel ** 2 / (t3 ** 2 + t1 ** 2))        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle
            
    def call(self, inputs, training=False):
        if training: self.add_loss(self._get_custom_L2())
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

@keras.utils.register_keras_serializable()
class RepVGG(keras.layers.Layer):

    def __init__(
        self, num_blocks, width_multiplier=None, override_groups_map=None,
        deploy=False,**kwargs
    ):
        super(RepVGG, self).__init__(**kwargs)
        assert len(width_multiplier) == 4
        self.num_blocks = num_blocks
        self.width_multiplier = width_multiplier
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(input_dim=3, out_dim=self.in_planes, kernel_size=3, stride=2, padding='SAME', deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.stages = [self.stage0, self.stage1, self.stage2, self.stage3, self.stage4]

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepVGGBlock(
                    input_dim=self.in_planes, out_dim=planes,  kernel_size=3,
                    stride=stride, padding='SAME', groups=cur_groups, deploy=self.deploy
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return keras.models.Sequential(blocks)
    
    def call(self, x):
        for stage in self.stages:
            x = stage(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_blocks": self.num_blocks,
                "width_multiplier": self.width_multiplier,
                "override_groups_map": self.override_groups_map,
                "deploy": self.deploy,
            }
        )
        return config

optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

def create_RepVGG_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5], 
                  override_groups_map=None, deploy=deploy)

def create_RepVGG_A1(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_A2(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)

def create_RepVGG_B0(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B1(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)

def create_RepVGG_B1g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B1g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)

def create_RepVGG_B2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B2g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B3(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B3g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B3g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)



func_dict = {
'RepVGG-A0': create_RepVGG_A0,
'RepVGG-A1': create_RepVGG_A1,
'RepVGG-A2': create_RepVGG_A2,
'RepVGG-B0': create_RepVGG_B0,
'RepVGG-B1': create_RepVGG_B1,
'RepVGG-B1g2': create_RepVGG_B1g2,
'RepVGG-B1g4': create_RepVGG_B1g4,
'RepVGG-B2': create_RepVGG_B2,
'RepVGG-B2g2': create_RepVGG_B2g2,
'RepVGG-B3': create_RepVGG_B3,
'RepVGG-B3g2': create_RepVGG_B3g2,
'RepVGG-B3g4': create_RepVGG_B3g4,    
}
def get_RepVGG_func_by_name(name):
    return func_dict[name]

def repvgg_layer_convert(RepVGG_layer):
    for stage in RepVGG_layer.stages:
        if isinstance(stage, RepVGGBlock):
            stage.fuse_repvgg_block()
        else:
            for layer in stage.layers:
                if hasattr(layer, 'fuse_repvgg_block'):
                    layer.fuse_repvgg_block()
    return RepVGG_layer
