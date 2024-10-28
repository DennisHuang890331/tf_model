import re

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size=6, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.resize = tf.keras.layers.Reshape((-1, patch_size * patch_size * 3))

    def call(self, inputs):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )

        patches = self.resize(patches)
        return patches
    
    def show_patched_image(self, images, patches):
        idx = np.random.choice(patches.shape[0])
        print(f"Index selected: {idx}.")

        plt.figure(figsize=(4, 4))
        plt.imshow(tf.keras.utils.array_to_img(images[idx]))
        plt.axis("off")
        plt.show()

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))
        for i, patch in enumerate(patches[idx]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = tf.reshape(patch, (self.patch_size, self.patch_size, 3))
            plt.imshow(tf.keras.utils.img_to_array(patch_img))
            plt.axis("off")
        plt.show()

        # Return the index chosen to validate it outside the method.
        return idx
    
    def reconstruct_from_patch(self, patches):
        num_patches = patches.shape[0]
        n = int(np.sqrt(num_patches))
        patches = tf.reshape(patches, (num_patches, self.patch_size, self.patch_size, 3))
        rows = tf.split(patches, n, axis=0)
        rows = [tf.concat(tf.unstack(x), axis=1) for x in rows]
        reconstructed = tf.concat(rows, axis=0)
        return reconstructed
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'patch_size': self.patch_size,
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class PatchEncoder(tf.keras.layers.Layer):
    
    def __init__(self, patch_size=6, projection_dim=128, mask_proportion=0.75, downstream=False, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion
        self.downstream = downstream

        # This is a trainable mask token initialized randomly from a normal distribution.
        self.mask_token = tf.Variable(
            tf.random.normal([1, self.patch_size * self.patch_size * 3]), trainable=True
        )
    
    def build(self, input_shape):
        (_, self.num_patches, self.patch_area) = input_shape
        
        # Create the projection layer for the patches.
        self.projection = tf.keras.layers.Dense(self.projection_dim)
        
        # Create the position embbeding.
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim
        )
        
        # Number of patches that will be masked.
        self.num_mask = int(self.mask_proportion * self.num_patches)

    def call(self, inputs):
        # Get position embedding.
        batch_size = tf.shape(inputs)[0]
        potitions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embeddings = self.position_embedding(potitions[tf.newaxis, ...])
        pos_embeddings = tf.tile(
            pos_embeddings, [batch_size, 1, 1]
        ) # (Batch_size, num_patches, projection_dim)

        # Embed the patches.
        patch_embeddings = (self.projection(inputs) + pos_embeddings) # (Batch_size, num_patches, projection_dim)

        if self.downstream:
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)
            # The encoder input is the unmasked patch embeddings. Here we gather
            # all the patches that should be unmasked.
            unmasked_embeddings = tf.gather(
                patch_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)


            # Get the unmasked and masked position embeddings. We will need them
            # for the decoder.
            unmasked_positions = tf.gather(
                pos_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)
            masked_positions = tf.gather(
                pos_embeddings, mask_indices, axis=1, batch_dims=1
            )  # (B, mask_numbers, projection_dim)
            
            # Repeat the mask token number of mask times.
            # Mask tokens replace the masks of the image.
            mask_tokens = tf.repeat(self.mask_token, repeats=self.num_mask, axis=0)
            mask_tokens = tf.repeat(
                mask_tokens[tf.newaxis, ...], repeats=batch_size, axis=0
            )

            # Get the masked embeddings for the tokens.
            masked_embeddings = self.projection(mask_tokens) + masked_positions

            return (
                unmasked_embeddings,  # Input to the encoder.
                masked_embeddings,  # First part of input to the decoder.
                unmasked_positions,  # Added to the encoder outputs.
                mask_indices,  # The indices that were masked.
                unmask_indices,  # The indices that were unmaksed.
            )

    def get_random_indices(self, batch_size):
        # Create random indices from a uniform distribution and then split
        # it into mask and unmask indices.
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_patches)), axis=1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices
    
    def generate_masked_image(self, patches, unmask_indices):
        # Choose a random patch and it corresponding unmask index.
        idx = np.random.choice(patches.shape[0])
        patch = patches[idx]
        unmask_index = unmask_indices[idx]

        # Build a numpy array of same shape as patch.
        new_patch = np.zeros_like(patch)

        # Iterate of the new_patch and plug the unmasked patches.
        count = 0
        for i in range(unmask_index.shape[0]):
            new_patch[unmask_index[i]] = patch[unmask_index[i]]
        return new_patch, idx
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'patch_size': self.patch_size,
                'projection_dim': self.projection_dim,
                'mask_proportion': self.mask_proportion,
                'downstream': self.downstream,
            }
        )
        return config

@tf.keras.utils.register_keras_serializable()
class MaskedAutoencoder(tf.keras.Model):

    def __init__(self, train_augmentation_model, test_augmentation_model, patch_layer, 
                 patch_encoder, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.train_augmentation_model = train_augmentation_model
        self.test_augmentation_model = test_augmentation_model
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=True):
        if training:
            x = self.train_augmentation_model(inputs)
            x = self.patch_layer(x)
            self.patch_encoder.downstream = False
            (
                unmasked_embeddings,
                masked_embeddings,
                unmasked_positions,
                mask_indices,
                unmask_indices,
            ) = self.patch_encoder(x)
            encoder_outputs = self.encoder(unmasked_embeddings)
            # Create the decoder inputs.
            encoder_outputs = encoder_outputs + unmasked_positions
            decoder_inputs = tf.concat([encoder_outputs, masked_embeddings], axis=1)
            # Decode the inputs.
            decoder_outputs = self.decoder(decoder_inputs)
            decoder_patches = self.patch_layer(decoder_outputs)
            y_pre = tf.gather(decoder_patches, mask_indices, axis=1, batch_dims=1)
        
            return y_pre
  

    def calculate_loss(self, images, test=False):
        # Augment the input images.
        if test:
            augmented_images = self.test_augmentation_model(images)
        else:
            augmented_images = self.train_augmentation_model(images)
        
        # Patch the augmented images.
        patches = self.patch_layer(augmented_images)
        # Encode the patches.
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.patch_encoder(patches)

        # Pass the unmaksed patche to the encoder.
        encoder_outputs = self.encoder(unmasked_embeddings)
        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = tf.concat([encoder_outputs, masked_embeddings], axis=1)
        # Decode the inputs.
        decoder_outputs = self.decoder(decoder_inputs)
        decoder_patches = self.patch_layer(decoder_outputs)

        y_true = tf.gather(patches, mask_indices, axis=1, batch_dims=1)
        y_pre = tf.gather(decoder_patches, mask_indices, axis=1, batch_dims=1)
        # Compute the total loss.
        total_loss = self.compiled_loss(y_true, y_pre)

        return total_loss, y_true, y_pre
    
    def train_step(self, images):
        with tf.GradientTape() as tape:
            total_loss, y_true, y_pre = self.calculate_loss(images)
        
        # Apply gradients.
        train_var = [
            self.train_augmentation_model.trainable_variables,
            self.patch_layer.trainable_variables,
            self.patch_encoder.trainable_variables,
            self.encoder.trainable_variables,
            self.decoder.trainable_variables,
        ]
        grads = tape.gradient(total_loss, train_var)
        tvlist = []
        for (grad, var) in zip(grads, train_var):
            for g, v in zip(grad, var):
                tvlist.append((g, v))
        self.optimizer.apply_gradients(tvlist)

        # Report progress.
        self.compiled_metrics.update_state(y_true, y_pre)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, images):
        total_loss, y_true, y_pre = self.calculate_loss(images, test=True)

        # Update the trackers.
        self.compiled_metrics.update_state(y_true, y_pre)
        return {m.name: m.result() for m in self.metrics}

def get_train_augmentation_model(image_shape=(32, 32, 3), crop_size=48):

    inputs = tf.keras.layers.Input(image_shape)
    x = tf.keras.layers.Rescaling(1/255.0)(inputs)
    x = tf.keras.layers.Resizing(image_shape[0] + 20, image_shape[0] + 20)(x)
    x = tf.keras.layers.RandomCrop(crop_size, crop_size)(x)
    ouputs = tf.keras.layers.RandomFlip('horizontal')(x)
    
    return tf.keras.Model(inputs, ouputs, name='train_data_augmentation')

def get_test_augmentation_model(image_shape=(32, 32, 3), crop_size=48):
    inputs = tf.keras.layers.Input(image_shape)
    x = tf.keras.layers.Rescaling(1/255.0)(inputs)
    ouputs = tf.keras.layers.Resizing(crop_size, crop_size)(x)
    return tf.keras.Model(inputs, ouputs, name='test_data_augmentation')

def mlp(x, dropout_rate, hidden_units):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

def create_encoder(num_heads=4, num_layers=6, projection_dim=128):
    inputs = tf.keras.layers.Input((None, projection_dim))
    x = inputs

    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = tf.keras.layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=0.1)

        # Skip connection 2.
        x = tf.keras.layers.Add()([x3, x2])

    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return tf.keras.Model(inputs, outputs, name="mae_encoder")

def create_decoder(num_layers=2, num_heads=4, image_size=48, num_patches=(48//6)**2, projection_dim=128):
    inputs = tf.keras.layers.Input((num_patches, projection_dim))
    x = tf.keras.layers.Dense(projection_dim)(inputs)

    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = tf.keras.layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=0.1)

        # Skip connection 2.
        x = tf.keras.layers.Add()([x3, x2])

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Flatten()(x)
    pre_final = tf.keras.layers.Dense(units=image_size * image_size * 3, activation="sigmoid")(x)
    outputs = tf.keras.layers.Reshape((image_size, image_size, 3))(pre_final)

    return tf.keras.Model(inputs, outputs, name="mae_decoder")

def save_masked_autoencoder(path, model):
    assert re.match(r'.*h5', path).group(0), "Error in model path. {path}"
    model.save_weights(path)
    return model

def load_model(path, input_size=(32,32,3), patch_size=6, projection_dim=128, mask_proportion=0.75,
                            num_head=4, enc_layer=6, dec_layer=2, image_size=48):
    assert re.match(r'.*h5', path).group(0), "Error in model path. {path}"

    train_augmentation_model = get_train_augmentation_model(input_size, image_size)
    test_augmentation_model = get_test_augmentation_model(input_size, image_size)
    patch_layer = Patches()
    patch_encoder = PatchEncoder(patch_size=patch_size, projection_dim=projection_dim,
                                 mask_proportion=mask_proportion)
    encoder = create_encoder(num_heads=num_head, num_layers=enc_layer,
                             projection_dim=projection_dim)
    decoder = create_decoder(num_layers=dec_layer, num_heads=num_head, image_size=image_size,
                             num_patches=(image_size // patch_size) ** 2, projection_dim=projection_dim)

    mae_model = MaskedAutoencoder(
        train_augmentation_model=train_augmentation_model,
        test_augmentation_model=test_augmentation_model,
        patch_layer=patch_layer,
        patch_encoder=patch_encoder,
        encoder=encoder,
        decoder=decoder,
    )
    mae_model.build(input_shape=(None, 32,32,3))
    mae_model.load_weights(path)
    mae_model.summary()
    
    return mae_model

def build_model(input_size=(32,32,3), patch_size=6, projection_dim=128, mask_proportion=0.75,
                            num_head=4, enc_layer=6, dec_layer=2, image_size=48):

    train_augmentation_model = get_train_augmentation_model(input_size, image_size)
    test_augmentation_model = get_test_augmentation_model(input_size, image_size)
    patch_layer = Patches()
    patch_encoder = PatchEncoder(patch_size=patch_size, projection_dim=projection_dim,
                                 mask_proportion=mask_proportion)
    encoder = create_encoder(num_heads=num_head, num_layers=enc_layer,
                             projection_dim=projection_dim)
    decoder = create_decoder(num_layers=dec_layer, num_heads=num_head, image_size=image_size,
                             num_patches=(image_size // patch_size) ** 2, projection_dim=projection_dim)

    mae_model = MaskedAutoencoder(
        train_augmentation_model=train_augmentation_model,
        test_augmentation_model=test_augmentation_model,
        patch_layer=patch_layer,
        patch_encoder=patch_encoder,
        encoder=encoder,
        decoder=decoder,
    )
    mae_model.build(input_shape=(None, 32,32,3))
    mae_model.summary()

    return mae_model

