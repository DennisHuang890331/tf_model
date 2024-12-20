import keras
import numpy as np
import tensorflow as tf

@keras.utils.register_keras_serializable()
class ChannelAffine(keras.layers.Layer):
    def __init__(self, use_bias=True, weight_init_value=1, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.use_bias, self.weight_init_value, self.axis = use_bias, weight_init_value, axis
        if isinstance(weight_init_value, (int, float)):
            self.ww_init = tf.initializers.Constant(weight_init_value) if weight_init_value != 1 else "ones"
        else:
            self.ww_init = weight_init_value  # Regard as built initializer
        self.bb_init = "zeros"
        self.supports_masking = False

    def build(self, input_shape):
        if self.axis == -1 or self.axis == len(input_shape) - 1:
            ww_shape = (input_shape[-1],)
        else:
            ww_shape = [1] * len(input_shape)
            axis = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
            for ii in axis:
                ww_shape[ii] = input_shape[ii]
            ww_shape = ww_shape[1:]  # Exclude batch dimension

        self.ww = self.add_weight(name="weight", shape=ww_shape, initializer=self.ww_init, trainable=True)
        if self.use_bias:
            self.bb = self.add_weight(name="bias", shape=ww_shape, initializer=self.bb_init, trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs * self.ww + self.bb if self.use_bias else inputs * self.ww

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_weights_channels_last(self):
        # channel_first -> channel_last
        weights = self.get_weights()
        if keras.backend.image_data_format() != "channels_last" and self.axis == 1:
            weights = [np.squeeze(ii) for ii in weights]
        return weights

    def set_weights_channels_last(self, weights):
        # channel_last -> channel_first
        if keras.backend.image_data_format() != "channels_last" and self.axis == 1:
            weights = [np.reshape(ii, self.ww.shape) for ii in weights]
        return self.set_weights(weights)

    def get_config(self):
        config = super().get_config()
        config.update({"use_bias": self.use_bias, "axis": self.axis})
        return config

@keras.utils.register_keras_serializable()
class StochasticDepth(keras.layers.Layer):
    """Stochastic Depth block by Dropout, arxiv: https://arxiv.org/abs/1603.09382"""
    def __init__(self, drop_rate=0, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate
    
    def build(self, input_shape):
        if self.drop_rate > 0:
            noise_shape = [None] + [1] * (len(input_shape) - 1)  # [None, 1, 1, 1]
            self.drop = keras.layers.Dropout(self.drop_rate, noise_shape=noise_shape)
        else:
            self.drop = None
    
    def call(self, inputs):
        return self.drop(inputs) if self.drop is not None else inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({"drop_rate": self.drop_rate})
        return config
    
@keras.utils.register_keras_serializable()
class MultiHeadRelativePositionalKernelBias(keras.layers.Layer):
    def __init__(self, input_height=-1, is_heads_first=False, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.input_height, self.is_heads_first, self.dilation_rate = input_height, is_heads_first, dilation_rate

    def _pad_bias_(self, indexes, total, dilation_rate):
        size = indexes.shape[0]
        bias_left = indexes[:size // 2]
        bias_right = indexes[size // 2 + 1:]
        bias_left = np.repeat(indexes[: size // 2], dilation_rate)
        bias_right = np.repeat(indexes[size // 2 + 1 :], dilation_rate)
        bias_center = np.repeat(indexes[size // 2], total - bias_left.shape[0] - bias_right.shape[0])
        return np.concatenate([bias_left, bias_center, bias_right], axis=-1)

    def build(self, input_shape):
        # input (is_heads_first=False): `[batch, height * width, num_heads, ..., size * size]`
        # input (is_heads_first=True): `[batch, num_heads, height * width, ..., size * size]`
        blocks, num_heads = (input_shape[2], input_shape[1]) if self.is_heads_first else (input_shape[1], input_shape[2])
        size = int(np.sqrt(float(input_shape[-1])))
        height = self.input_height if self.input_height > 0 else int(np.sqrt(float(blocks)))
        width = blocks // height
        pos_size = 2 * size - 1
        initializer = keras.initializers.truncated_normal(stddev=0.02)
        self.pos_bias = self.add_weight(name="positional_embedding", shape=(num_heads, pos_size * pos_size), initializer=initializer, trainable=True)

        dilation_rate = self.dilation_rate if isinstance(self.dilation_rate, (list, tuple)) else (self.dilation_rate, self.dilation_rate)
        idx_hh, idx_ww = np.arange(0, size), np.arange(0, size)
        
        coords = np.reshape(np.expand_dims(idx_hh, -1) * pos_size + idx_ww, [-1]).astype("int64")
        bias_hh = self._pad_bias_(idx_hh, total=height, dilation_rate=dilation_rate[0])
        bias_ww = self._pad_bias_(idx_ww, total=width, dilation_rate=dilation_rate[1])

        bias_hw = np.expand_dims(bias_hh, -1) * pos_size + bias_ww
        bias_coords = np.expand_dims(bias_hw, -1) + coords
        bias_coords = np.reshape(bias_coords, [-1, size**2])[::-1]  # torch.flip(bias_coords, [0])
        bias_coords_shape = [bias_coords.shape[0]] + [1] * (len(input_shape) - 4) + [bias_coords.shape[1]]

        bias_coords = np.reshape(bias_coords, bias_coords_shape)  # [height * width, 1 * n, size * size]
        self.bias_coords = tf.convert_to_tensor(bias_coords, dtype="int64")

        if not self.is_heads_first:
            self.transpose_perm = [1, 0] + list(range(2, len(input_shape) - 1))  # transpose [num_heads, height * width] -> [height * width, num_heads]

    def call(self, inputs):
        if self.is_heads_first:
            return inputs + tf.gather(self.pos_bias, self.bias_coords, axis=-1)
        else:
            return inputs + tf.transpose(tf.gather(self.pos_bias, self.bias_coords, axis=-1), self.transpose_perm)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"input_height": self.input_height, "is_heads_first": self.is_heads_first, "dilation_rate": self.dilation_rate})
        return base_config

@keras.utils.register_keras_serializable()
class CompatibleExtractPatches(keras.layers.Layer):
    def __init__(self, sizes=3, strides=2, rates=1, padding="same", compressed=True, use_conv=False, **kwargs):
        super().__init__(**kwargs)
        self.sizes, self.strides, self.rates, self.padding = sizes, strides, rates, padding
        self.compressed, self.use_conv = compressed, use_conv

        self.kernel_size = sizes[1] if isinstance(sizes, (list, tuple)) else sizes
        self.strides = strides[1] if isinstance(strides, (list, tuple)) else strides
        # dilation_rate can be 2 different values, used in DiNAT
        self.dilation_rate = (rates if len(rates) == 2 else rates[1:3]) if isinstance(rates, (list, tuple)) else (rates, rates)
        self.filters = self.kernel_size * self.kernel_size


    def build(self, input_shape):
        _, self.height, self.width, self.channel = input_shape
        if self.padding.lower() == "same":
            pad_value = self.kernel_size // 2
            self.pad_value_list = [[0, 0], [pad_value, pad_value], [pad_value, pad_value], [0, 0]]
            self.height, self.width = self.height + pad_value * 2, self.width + pad_value * 2
            self.pad_value = pad_value
        else:
            self.pad_value = 0

        if self.use_conv:
            self.conv = keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                dilation_rate=self.dilation_rate,
                padding="valid",
                use_bias=False,
                trainable=False,
                kernel_initializer=__unfold_filters_initializer__,
                name=self.name and self.name + "unfold_conv",
            )
            self.conv.build([None, *input_shape[1:-1], 1])
        else:
            self._sizes_ = [1, self.kernel_size, self.kernel_size, 1]
            self._strides_ = [1, self.strides, self.strides, 1]
            self._rates_ = [1, *self.dilation_rate, 1]
        # output_size = backend.compute_conv_output_size([self.height, self.width], self.kernel_size, self.strides, self.padding, self.dilation_rate)
        # self.output_height, self.output_width = output_size
        super().build(input_shape)

    def call(self, inputs):
        if self.pad_value > 0:
            inputs = tf.pad(inputs, self.pad_value_list)

        if self.use_conv:
            merge_channel = tf.transpose(inputs, [0, 3, 1, 2])
            merge_channel = tf.reshape(merge_channel, [-1, self.height, self.width, 1])
            conv_rr = self.conv(merge_channel)

            out = tf.reshape(conv_rr, [-1, self.channel, conv_rr.shape[1] * conv_rr.shape[2], self.filters])
            out = tf.transpose(out, [0, 2, 3, 1])  # [batch, hh * ww, kernel * kernel, channnel]
            if self.compressed:
                out = tf.reshape(out, [-1, conv_rr.shape[1], conv_rr.shape[2], self.filters * self.channel])
            else:
                out = tf.reshape(out, [-1, conv_rr.shape[1], conv_rr.shape[2], self.kernel_size, self.kernel_size, self.channel])
        else:
            out = tf.image.extract_patches(inputs, self._sizes_, self._strides_, self._rates_, "VALID")  # must be upper word VALID/SAME
            if not self.compressed:
                # [batch, hh, ww, kernel, kernel, channnel]
                out = tf.reshape(out, [-1, out.shape[1], out.shape[2], self.kernel_size, self.kernel_size, self.channel])
        return out

    def get_config(self):
        base_config = super().get_config()
        base_config.update(
            {
                "sizes": self.sizes,
                "strides": self.strides,
                "rates": self.rates,
                "padding": self.padding,
                "compressed": self.compressed,
                "use_conv": self.use_conv,
            }
        )
        return base_config

@keras.utils.register_keras_serializable()
class NeighborhoodAttention(keras.layers.Layer):
    
    def __init__(self, kernel_size=7, num_heads=1, key_dim=0, out_weight=True, qkv_bias=True, out_bias=True, dilation_rate=1, attn_dropout=0, output_dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size, self.num_heads, self.key_dim = kernel_size, num_heads, key_dim
        self.out_weight, self.qkv_bias, self.out_bias = out_weight, qkv_bias, out_bias
        self.attn_dropout, self.output_dropout = attn_dropout, output_dropout
        self.dilation_rate = dilation_rate if isinstance(dilation_rate, (list, tuple)) else (dilation_rate, dilation_rate)
        
        if self.output_dropout > 0:
            self.output_dropout_layer = keras.layers.Dropout(self.output_dropout, name=self.name and self.name + "out_drop")
        else:
            self.output_dropout_layer = None

        if self.attn_dropout > 0:
            self.attn_dropout_layer = keras.layers.Dropout(self.attn_dropout, name=self.name and self.name + "attn_drop")
        else:
            self.attn_dropout_layer = None

    def build(self, input_shape):
        _, hh, ww, cc = input_shape
        self.key_dim = self.key_dim if self.key_dim > 0 else cc // self.num_heads
        window_size = [int(self.kernel_size * ii) for ii in self.dilation_rate]
        # window_size = int(kernel_size + (kernel_size - 1) * (dilation_rate - 1))
        self.should_pad_hh, self.should_pad_ww = max(0, window_size[0] - hh), max(0, window_size[1] - ww)
        self.qkv_out = self.num_heads * self.key_dim
        self.qkv = keras.layers.Dense(self.qkv_out * 3, use_bias=self.qkv_bias, name=self.name and self.name + "qkv")
        self.compatible_extract_patch = CompatibleExtractPatches(sizes=self.kernel_size, strides=1, rates=self.dilation_rate, padding="valid", compressed=False)
        self.kernel_biases = MultiHeadRelativePositionalKernelBias(input_height=hh, dilation_rate=self.dilation_rate, name=self.name and self.name + "pos")
        self.position_kernel_biases = MultiHeadRelativePositionalKernelBias(input_height=hh, dilation_rate=self.dilation_rate, name=self.name and self.name + "pos")

        if self.out_weight:
            self.out_dense = keras.layers.Dense(cc, use_bias=self.out_bias, name=self.name and self.name + "output")
        else:
            self.out_dense = None
        

    def _replicate_padding(self, inputs, kernel_size=1, dilation_rate=1):
        padded = (kernel_size - 1) // 2
        dilation_rate = dilation_rate if isinstance(dilation_rate, (list, tuple)) else (dilation_rate, dilation_rate)

        if max(dilation_rate) == 1:  # TF NAT
            nn = tf.concat([tf.repeat(inputs[:, :1], padded, axis=1), inputs, tf.repeat(inputs[:, -1:], padded, axis=1)], axis=1)
            out = tf.concat([tf.repeat(nn[:, :, :1], padded, axis=2), nn, tf.repeat(nn[:, :, -1:], padded, axis=2)], axis=2)
        else:  # TF DiNAT
            # left = functional.repeat(functional.expand_dims(inputs[:, :dilation_rate], axis=1), padded, axis=1)
            # left = functional.reshape(left, [-1, left.shape[1] * left.shape[2], *left.shape[2:]])
            multiples = [padded if id == 1 else 1 for id in range(len(inputs.shape))]
            top = tf.tile(inputs[:, : dilation_rate[0]], multiples)
            bottom = tf.tile(inputs[:, -dilation_rate[0] :], multiples)
            top_bottom = tf.concat([top, inputs, bottom], axis=1)

            multiples = [padded if id == 2 else 1 for id in range(len(inputs.shape))]
            left = tf.tile(top_bottom[:, :, : dilation_rate[1]], multiples)
            right = tf.tile(top_bottom[:, :, -dilation_rate[1] :], multiples)
            out = tf.concat([left, top_bottom, right], axis=2)
        return out


    def call(self, inputs):
        if self.should_pad_hh or self.should_pad_ww:
            inputs = tf.pad(inputs, [[0, 0], [0, self.should_pad_hh], [0, self.should_pad_ww], [0, 0]])
        _, hh, ww, _ = inputs.shape

        qkv = self.qkv(inputs)
        query, key_value = tf.split(qkv, [self.qkv_out, self.qkv_out * 2], axis=-1)  
        query = tf.expand_dims(tf.reshape(query, [-1, hh * ww, self.num_heads, self.key_dim]), -2)  # [batch, hh * ww, num_heads, 1, key_dim]
        # key_value: [batch, height - (kernel_size - 1), width - (kernel_size - 1), kernel_size, kernel_size, key + value]

        key_value = self.compatible_extract_patch(key_value)
        key_value = self._replicate_padding(key_value, self.kernel_size, self.dilation_rate)

        key_value = tf.reshape(key_value, [-1, self.kernel_size * self.kernel_size, self.qkv_out * 2])
        key, value = tf.split(key_value, 2, axis=-1)  # [batch * block_height * block_width, K * K, key_dim]
        
        key = tf.transpose(tf.reshape(key, [-1, key.shape[1], self.num_heads, self.key_dim]), [0, 2, 3, 1])  # [batch * hh*ww, num_heads, key_dim, K * K]
        key = tf.reshape(key, [-1, hh * ww, self.num_heads, self.key_dim, self.kernel_size * self.kernel_size])  # [batch, hh*ww, num_heads, key_dim, K * K]
    
        value = tf.transpose(tf.reshape(value, [-1, value.shape[1], self.num_heads, self.key_dim]), [0, 2, 1, 3])
        value = tf.reshape(value, [-1, hh * ww, self.num_heads, self.kernel_size * self.kernel_size, self.key_dim])  # [batch, hh*ww, num_heads, K * K, key_dim]

        attention_scores = (query @ key) * 1.0 / (float(self.key_dim) ** 0.5)
        attention_scores = self.position_kernel_biases(attention_scores)
        attention_scores = tf.nn.softmax(attention_scores, -1)
        if self.attn_dropout_layer is not None:
            attention_scores = self.attn_dropout_layer(attention_scores)


        # attention_output = [batch, block_height * block_width, num_heads, 1, key_dim]
        attention_output = attention_scores @ value
        attention_output = tf.reshape(attention_output, [-1, hh, ww, self.num_heads * self.key_dim])

        if self.should_pad_hh or self.should_pad_ww:
            attention_output = attention_output[:, : hh - self.should_pad_hh, : ww - self.should_pad_ww, :]

        if self.out_weight:
            # [batch, hh, ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, hh, ww, out]
            attention_output = self.out_dense(attention_output)
        if self.output_dropout_layer is not None:
            attention_output = self.output_dropout_layer(attention_output)
        return attention_output
    
    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "kernel_size": self.kernel_size,
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "qkv_bias": self.qkv_bias,
            "out_bias": self.out_bias,
            "dilation_rate": self.dilation_rate,
            "attn_dropout": self.attn_dropout,
            "output_dropout": self.output_dropout,

        })
        return base_config

@keras.utils.register_keras_serializable()
class NeighborhoodAttentionBlock(keras.layers.Layer):

    def __init__(self, attn_kernel_size=7, num_heads=4, mlp_ratio=4, dilation_rate=1, attn_drop_rate=0, drop_rate=0, layer_scale=-1, **kwargs):
        super().__init__(**kwargs)
        self.attn_kernel_size, self.num_heads, self.mlp_ratio = attn_kernel_size, num_heads, mlp_ratio
        self.dilation_rate, self.attn_drop_rate, self.drop_rate = dilation_rate, attn_drop_rate, drop_rate
        self.layer_scale = layer_scale
        self.input_norm = keras.layers.LayerNormalization(name=self.name + "input_norm")
        self.attn = NeighborhoodAttention(attn_kernel_size, num_heads, dilation_rate=dilation_rate, attn_dropout=attn_drop_rate, name=self.name + "attn_")
        self.attn_scale_drop = LayerScalewithDropBlock(layer_scale, 1, drop_rate, axis=-1, name=self.name + "attn_")
        self.mlp_norm = keras.layers.LayerNormalization(name=self.name + "attn_norm")
        self.mlp_scale_drop = LayerScalewithDropBlock(layer_scale, 1, drop_rate, axis=-1, name=self.name + "attn_")

    def build(self, input_shape):
        self.mlp = keras.Sequential([
            keras.layers.Dense(input_shape[-1] * self.mlp_ratio, activation="gelu", name=self.name + "mlp_dense1"),
            keras.layers.Dense(input_shape[-1], activation="gelu", name=self.name + "mlp_dense2"),
        ])
            
    def call(self, inputs):
        x = self.input_norm(inputs)
        x = self.attn(x)
        x = self.attn_scale_drop(x)
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.mlp_scale_drop(x)
        return x
    
    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "attn_kernel_size": self.attn_kernel_size,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dilation_rate": self.dilation_rate,
            "attn_drop_rate": self.attn_drop_rate,
            "drop_rate": self.drop_rate,
            "layer_scale": self.layer_scale,
        })
        return base_config

@keras.utils.register_keras_serializable()
class LayerScalewithDropBlock(keras.layers.Layer):

    def __init__(self, layer_scale, residual_scale, drop_rate,  axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.layer_scale, self.residual_scale, self.drop_rate, self.axis = layer_scale, residual_scale, drop_rate, axis
        self.short = ChannelAffine(use_bias=False, weight_init_value=residual_scale, axis=axis, name=self.name + "res_gamma") if residual_scale > 0 else None
        self.deep = ChannelAffine(use_bias=False, weight_init_value=layer_scale, axis=axis, name=self.name + "gamma") if layer_scale > 0 else None
        self.drop = StochasticDepth(drop_rate, name=self.name + "drop") if drop_rate > 0 else None

    def call(self, inputs):
        short = self.short(inputs) if self.short is not None else inputs
        deep = self.deep(inputs) if self.deep is not None else inputs
        deep = self.drop(deep) if self.drop is not None else deep
        return short + deep

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "layer_scale": self.layer_scale,
            "residual_scale": self.residual_scale,
            "drop_rate": self.drop_rate,
            "axis": self.axis,
        })
        return base_config

class NeighborhoodAttentionTransformer(keras.layers.Layer):

    def __init__(
            self, 
            num_blocks=[3, 4, 6, 5],
            out_channels=[64, 128, 256, 512],
            num_heads=[2, 4, 8, 16],
            stem_width=-1,
            attn_kernel_size=7,
            use_every_other_dilations=False,  # True for DiNAT, using `dilation_rate=nn.shape[1] // attn_kernel_size` in every other attention blocks
            mlp_ratio=3,
            layer_scale=-1,
            activation="gelu",
            drop_connect_rate=0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_blocks, self.out_channels, self.num_heads = num_blocks, out_channels, num_heads
        self.stem_width, self.attn_kernel_size, self.use_every_other_dilations = stem_width, attn_kernel_size, use_every_other_dilations
        self.mlp_ratio, self.layer_scale, self.activation = mlp_ratio, layer_scale, activation, 
        self.drop_connect_rate = drop_connect_rate

        stem_width = stem_width if stem_width > 0 else out_channels[0]
        self.input_sequence = keras.Sequential([
            keras.layers.Conv2D(stem_width // 2, kernel_size=3, strides=2, use_bias=True, padding="same", name="stem_1_"),
            keras.layers.Conv2D(stem_width, kernel_size=3, strides=2, use_bias=True, padding="same", name="stem_2_"),
            keras.layers.LayerNormalization(-1, name="stem_"),
        ])
        
    def build(self, input_shape):
        """ stages """
        total_blocks = sum(self.num_blocks)
        global_block_id = 0
        _, H, W, _ = input_shape
        stages = []
        for stack_id, (num_block, out_channel, num_head) in enumerate(zip(self.num_blocks, self.out_channels, self.num_heads)):
            cur_stage = []
            stack_name = "stack{}_".format(stack_id + 1)
            if stack_id > 0:

                ds_name = stack_name + "downsample_"
                cur_stage.append(keras.layers.Conv2D(out_channel, kernel_size=3, strides=2, padding="same", name=ds_name + "_conv"))
                cur_stage.append(keras.layers.LayerNormalization(name=ds_name + "_norm"))
                H //= 2
                W //= 2
            for block_id in range(num_block):
                block_name = stack_name + "block{}_".format(block_id + 1)
                drop_rate = self.drop_connect_rate * global_block_id / total_blocks
                if self.use_every_other_dilations and block_id % 2 == 1:
                    # input 224 kernel_size 7 -> [8, 4, 2 ,1], input 384 kernel_size 7 -> [13, 6, 3 ,1], input 384 kernel_size 11 -> [8, 4, 2, 1]
                    dilation_rate = (max(1, int(H // self.attn_kernel_size)), max(1, int(W // self.attn_kernel_size)))
                else:
                    dilation_rate = 1 

                cur_stage.append(NeighborhoodAttentionBlock(
                    attn_kernel_size=self.attn_kernel_size,
                    num_heads=num_head,
                    mlp_ratio=self.mlp_ratio,
                    dilation_rate=dilation_rate,
                    drop_rate=drop_rate,
                    layer_scale=self.layer_scale,
                    name=block_name,
                ))
                global_block_id += 1
            cur_stage.append(keras.layers.LayerNormalization(name="pre_output_"))
            stages.append(cur_stage)
        self.stage1 = keras.Sequential(stages[0])
        self.stage2 = keras.Sequential(stages[1])
        self.stage3 = keras.Sequential(stages[2])
        self.stage4 = keras.Sequential(stages[3])

    
    def call(self, inputs):
        x = self.input_sequence(inputs)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
    
    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "num_blocks": self.num_blocks,
            "out_channels": self.out_channels,
            "num_heads": self.num_heads,
            "stem_width": self.stem_width,
            "attn_kernel_size": self.attn_kernel_size,
            "use_every_other_dilations": self.use_every_other_dilations,
            "mlp_ratio": self.mlp_ratio,
            "layer_scale": self.layer_scale,
            "activation": self.activation,
            "drop_connect_rate": self.drop_connect_rate,
        })
        return base_config
    
def __unfold_filters_initializer__(weight_shape, dtype="float32"):
    kernel_size = weight_shape[0]
    kernel_out = kernel_size * kernel_size
    ww = np.reshape(np.eye(kernel_out, dtype="float32"), [kernel_size, kernel_size, 1, kernel_out])
    if len(weight_shape) == 5:  # Conv3D or Conv3DTranspose
        ww = np.expand_dims(ww, 2)
    return tf.convert_to_tensor(ww)

def NAT_Mini(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape)
    x = NeighborhoodAttentionTransformer(num_blocks=[3, 4, 6, 5])(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.models.Model(inputs, outputs)
    return model

def NAT_Tiny(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape)
    x = NeighborhoodAttentionTransformer(num_blocks=[3, 4, 18, 5])(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.models.Model(inputs, outputs)
    return model

def NAT_Small(input_shape, num_classes):
    num_blocks = [3, 4, 18, 5]
    num_heads = [3, 6, 12, 24]
    out_channels = [96, 192, 384, 768]
    inputs = keras.layers.Input(shape=input_shape)
    x = NeighborhoodAttentionTransformer(
        num_blocks=num_blocks,
        out_channels=out_channels,
        num_heads=num_heads,
        mlp_ratio=2,
        layer_scale=1e-5,
    )(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.models.Model(inputs, outputs)
    return model

def NAT_Base(input_shape, num_classes):
    num_blocks = [3, 4, 18, 5]
    num_heads = [4, 8, 16, 32]
    out_channels = [128, 256, 512, 1024]
    inputs = keras.layers.Input(shape=input_shape)
    x = NeighborhoodAttentionTransformer(
        num_blocks=num_blocks,
        out_channels=out_channels,
        num_heads=num_heads,
        mlp_ratio=2,
        layer_scale=1e-5,
    )(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.models.Model(inputs, outputs)
    return model
