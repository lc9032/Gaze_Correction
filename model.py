from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, ReLU, Conv2DTranspose, Activation, LayerNormalization, Add, Concatenate, BatchNormalization
import tensorflow as tf
import tf_slim as slim

import matplotlib.pyplot as plt
import numpy as np


# def lrelu(x, alpha=0.2):
#     return tf.nn.leaky_relu(x, alpha)

# def instance_norm(x, name='instance_norm'):
#     return LayerNormalization(axis=[1, 2], name=name)(x)

class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(InstanceNormalization, self).__init__(name=name)
        self.layer_norm = LayerNormalization(axis=[1, 2])

    def call(self, x):
        return self.layer_norm(x)

# class Discriminator(tf.keras.Model):
#     def __init__(self, params):
#         super(Discriminator, self).__init__()
#         self.layers_num = 5
#         self.channel = 64
#         self.image_size = params.image_size
#         self.conv_layers = []

#         # Initialize convolutional layers
#         for i in range(self.layers_num):
#             filters = self.channel * (2 ** i)
#             self.conv_layers.append(
#                 tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, padding='same', use_bias=True, name=f'conv_{i}')
#             )

#         filter_size = int(self.image_size / (2 ** self.layers_num))

#         # self.conv_logit_gan = tf.keras.layers.Conv2D(1, kernel_size=filter_size, strides=1, padding='valid', use_bias=False, name='conv_logit_gan')
#         # self.conv_logit_reg = tf.keras.layers.Conv2D(2, kernel_size=filter_size, strides=1, padding='valid', use_bias=False, name='conv_logit_reg')
#         self.conv_logit_gan = Conv2D(1, kernel_size=2, strides=1, padding='valid', use_bias=False)
#         self.conv_logit_reg = Conv2D(2, kernel_size=2, strides=1, padding='valid', use_bias=False)


#     def call(self, x_init, training=False):
#         x = x_init

#         for conv in self.conv_layers:
#             x = conv(x)
#             x = lrelu(x)

#         x_gan = self.conv_logit_gan(x)
#         x_reg = self.conv_logit_reg(x)
#         x_reg = tf.reshape(x_reg, [-1, 2])

#         return x_gan, x_reg

# class ConvolutionBlock(tf.keras.layers.Layer):
#     def __init__(self, filters):
#         super(ConvolutionBlock, self).__init__()
#         self.conv1 = Conv2D(filters, kernel_size=3, padding='same', use_bias=False)
#         self.norm1 = BatchNormalization()
#         self.relu = ReLU()
#         self.conv2 = Conv2D(filters, kernel_size=3, padding='same', use_bias=False)
#         self.norm2 = BatchNormalization()
    
#     def call(self, x):
#         x1 = self.conv1(x)
#         x1 = self.norm1(x1)
#         x1 = self.relu(x1)
#         x1 = self.conv2(x1)
#         x1 = self.norm2(x1)
#         x1 = self.relu(x1)
#         return x1

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # Define the convolution blocks as shown in the architecture
        self.conv1 = self._conv_block(64, (3, 3), (1, 1))
        self.conv2 = self._conv_block(128, (3, 3), (1, 1))
        self.conv3 = self._conv_block(256, (3, 3), (1, 1))
        
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.upconv1 = self._upconv_block(128, (3, 3), (2, 2))
        self.upconv2 = self._upconv_block(64, (3, 3), (2, 2))
        self.upconv3 = self._upconv_block(64, (3, 3), (2, 2))
        
        self.final_conv = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='tanh')
        
    def _conv_block(self, filters, kernel_size, strides):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
    
    def _upconv_block(self, filters, kernel_size, strides):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
    
    def call(self, input_image, target_angle):
        # Expand target_angle to match the spatial dimensions of input_image
        batch_size = tf.shape(input_image)[0]
        height = tf.shape(input_image)[1]
        width = tf.shape(input_image)[2]
        
        target_angle = tf.reshape(target_angle, (batch_size, 1, 1, 2))
        target_angle = tf.tile(target_angle, [1, height, width, 1])
        
        x = tf.concat([input_image, target_angle], axis=-1)
        
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        output_image = self.final_conv(x)
        return output_image

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    # def __init__(self):
    #     super(Generator, self).__init__()
    #     self.initial_channels = 32
    #     # self.target_angle_tile = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 64, 32, 1]))
    #     self.target_angle_tile = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 32, 64, 1]))
    #     self.concat = Concatenate(axis=-1)

    #     # Define convolution blocks and pooling layers
    #     self.conv_block1 = ConvolutionBlock(32)
    #     self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    #     self.conv_block2 = ConvolutionBlock(64)
    #     self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    #     self.conv_block3 = ConvolutionBlock(128)
    #     self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    #     self.conv_block4 = ConvolutionBlock(256)

    #     # Define up-convolution blocks
    #     self.upconv1 = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)
    #     self.conv_block5 = ConvolutionBlock(128)
    #     self.upconv2 = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False)
    #     self.conv_block6 = ConvolutionBlock(64)
    #     self.upconv3 = Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', use_bias=False)
    #     self.conv_block7 = ConvolutionBlock(32)

    #     # Define the final layers
    #     self.vector_field_conv = Conv2D(2, kernel_size=3, padding='same', use_bias=False, activation='sigmoid')
    #     self.brightness_map_conv = Conv2D(1, kernel_size=3, padding='same', use_bias=False, activation='sigmoid')
    #     self.final_conv = Conv2D(3, kernel_size=3, padding='same', use_bias=False, activation='sigmoid')
    
    # def warp_image(self, image, vector_field):
    #     """ Warp the image according to the vector field. """
    #     batch_size, height, width, _ = tf.shape(image)
    #     grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
    #     grid_x = tf.cast(grid_x, tf.float32)
    #     grid_y = tf.cast(grid_y, tf.float32)
    #     grid_x = tf.expand_dims(grid_x, axis=0)
    #     grid_y = tf.expand_dims(grid_y, axis=0)
    #     grid_x = tf.tile(grid_x, [batch_size, 1, 1])
    #     grid_y = tf.tile(grid_y, [batch_size, 1, 1])
    #     vector_x = vector_field[:, :, :, 0]
    #     vector_y = vector_field[:, :, :, 1]
    #     warped_x = grid_x + vector_x
    #     warped_y = grid_y + vector_y
    #     warped_x = tf.clip_by_value(warped_x, 0, tf.cast(width - 1, tf.float32))
    #     warped_y = tf.clip_by_value(warped_y, 0, tf.cast(height - 1, tf.float32))
    #     warped_coords = tf.stack([warped_y, warped_x], axis=-1)
    #     warped_image = tf.gather_nd(image, tf.cast(warped_coords, tf.int32), batch_dims=1)
    #     return warped_image

    # def adjust_brightness(self, image, brightness_map):
    #     """ Adjust the brightness of the image according to the brightness map. """
    #     return image * brightness_map

    # # @tf.function
    # def call(self, input_image, target_angle):
    #     # Target angle processing
    #     target_angle_tiled = self.target_angle_tile(target_angle)
    #     x = self.concat([input_image, target_angle_tiled])
    #     print(f'Shape after concatenating angles: {x.shape}')

    #     # Convolution and pooling layers
    #     x = self.conv_block1(x)
    #     print(f'Shape after conv_block1: {x.shape}')
    #     x = self.pool1(x)
    #     print(f'Shape after pool1: {x.shape}')
        
    #     x = self.conv_block2(x)
    #     print(f'Shape after conv_block2: {x.shape}')
    #     x = self.pool2(x)
    #     print(f'Shape after pool2: {x.shape}')
        
    #     x = self.conv_block3(x)
    #     print(f'Shape after conv_block3: {x.shape}')
    #     x = self.pool3(x)
    #     print(f'Shape after pool3: {x.shape}')
        
    #     x = self.conv_block4(x)
    #     print(f'Shape after conv_block4: {x.shape}')

    #     # Up-convolution layers
    #     x = self.upconv1(x)
    #     print(f'Shape after upconv1: {x.shape}')
    #     x = self.conv_block5(x)
    #     print(f'Shape after conv_block5: {x.shape}')
        
    #     x = self.upconv2(x)
    #     print(f'Shape after upconv2: {x.shape}')
    #     x = self.conv_block6(x)
    #     print(f'Shape after conv_block6: {x.shape}')
        
    #     x = self.upconv3(x)
    #     print(f'Shape after upconv3: {x.shape}')
    #     x = self.conv_block7(x)
    #     print(f'Shape after conv_block7: {x.shape}')

    #     # # Vector field and brightness map
    #     vector_field = self.vector_field_conv(x)
    #     brightness_map = self.brightness_map_conv(x)
    #     print(f'Shape of vector field: {vector_field.shape}')
    #     print(f'Shape of brightness map: {brightness_map.shape}')

    #     # # Output image processing (warp and adjust brightness can be implemented separately)
    #     # warped_image = self.warp_image(input_image, vector_field)
    #     # output_image = self.adjust_brightness(warped_image, brightness_map)

    #     # output_image = self.final_conv(x)

    #     # return output_image
    #     return vector_field, brightness_map


    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    # def __init__(self):
    #     super(Generator, self).__init__()

    #     # Encoder layers
    #     self.encoder = [
    #         layers.Conv2D(32, kernel_size=4, strides=2, padding='same', activation='relu'),
    #         layers.BatchNormalization(),
    #         layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu'),
    #         layers.BatchNormalization(),
    #         layers.Conv2D(128, kernel_size=4, strides=2, padding='same', activation='relu'),
    #         layers.BatchNormalization(),
    #         layers.Conv2D(256, kernel_size=4, strides=2, padding='same', activation='relu'),
    #         layers.BatchNormalization()
    #     ]

    #     # Middle layers
    #     self.middle = [
    #         layers.Conv2D(512, kernel_size=4, strides=2, padding='same', activation='relu'),
    #         layers.BatchNormalization(),
    #         layers.Conv2D(512, kernel_size=4, strides=2, padding='same', activation='relu'),
    #         layers.BatchNormalization()
    #     ]

    #     # Decoder layers
    #     self.decoder = [
    #         layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', activation='relu'),
    #         layers.BatchNormalization(),
    #         layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu'),
    #         layers.BatchNormalization(),
    #         layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu'),
    #         layers.BatchNormalization(),
    #         layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', activation='relu')
    #     ]

    #     self.final_layer = layers.Conv2DTranspose(3, kernel_size=4, strides=1, padding='same', activation='tanh')


    # def call(self, input_, angles):
    #     x = input_

    #     # Ensure the angles tensor is compatible
    #     batch_size = tf.shape(x)[0]
    #     height, width = tf.shape(x)[1], tf.shape(x)[2]
    #     angles_expanded = tf.reshape(angles, (batch_size, 1, 1, angles.shape[-1]))
    #     angles_expanded = tf.tile(angles_expanded, [1, height, width, 1])
    #     x = tf.concat([x, angles_expanded], axis=-1)
    #     print(f"Shape after concatenating angles: {x.shape}")

    #     # Encoder
    #     print(f"Input shape: {x.shape}")
    #     for layer in self.encoder:
    #         x = layer(x)
    #         print(f"Shape after encoder layer {layer.name}: {x.shape}")

    #     # Middle
    #     # for layer in self.middle:
    #     #     x = layer(x)
    #     #     print(f"Shape after middle layer {layer.name}: {x.shape}")

    #     # Decoder
    #     for layer in self.decoder:
    #         x = layer(x)
    #         print(f"Shape after decoder layer {layer.name}: {x.shape}")

    #     x = self.final_layer(x)
    #     print(f"Shape after final layer: {x.shape}")

    #     return x
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    # def __init__(self, channel=64, num_bottleneck_blocks=30):
    #     super(Generator, self).__init__()
    #     self.channel = channel
    #     self.num_bottleneck_blocks = num_bottleneck_blocks

    #     # Define the layers
    #     self.input_conv = Conv2D(channel, kernel_size=7, strides=1, padding='same', use_bias=False)
    #     self.input_norm = InstanceNormalization()
    #     self.relu = ReLU()

    #     self.encoder_convs = []
    #     self.encoder_norms = []
    #     for i in range(2):
    #         channel *= 2
    #         self.encoder_convs.append(Conv2D(channel, kernel_size=4, strides=2, padding='same', use_bias=False))
    #         self.encoder_norms.append(InstanceNormalization())

    #     self.bottleneck_convs_a = []
    #     self.bottleneck_norms_a = []
    #     self.bottleneck_convs_b = []
    #     self.bottleneck_norms_b = []
    #     for i in range(num_bottleneck_blocks):
    #         self.bottleneck_convs_a.append(Conv2D(channel, kernel_size=3, strides=1, padding='same', use_bias=False))
    #         self.bottleneck_norms_a.append(InstanceNormalization())
    #         self.bottleneck_convs_b.append(Conv2D(channel, kernel_size=3, strides=1, padding='same', use_bias=False))
    #         self.bottleneck_norms_b.append(InstanceNormalization())

    #     self.decoder_convs = []
    #     self.decoder_norms = []
    #     for i in range(2):
    #         channel //= 2
    #         self.decoder_convs.append(Conv2DTranspose(channel, kernel_size=4, strides=2, padding='same', use_bias=False))
    #         self.decoder_norms.append(InstanceNormalization())

    #     self.output_conv = Conv2D(3, kernel_size=7, strides=1, padding='same', use_bias=False)
    #     self.tanh = tf.keras.activations.tanh

    # def call(self, input_, angles):
    #     style_dim = angles.shape[-1]

    #     angles_reshaped = tf.reshape(angles, [-1, 1, 1, style_dim])
    #     angles_tiled = tf.tile(angles_reshaped, [1, tf.shape(input_)[1], tf.shape(input_)[2], 1])
    #     x = tf.concat([input_, angles_tiled], axis=3)

    #     # input layer
    #     x = self.input_conv(x)
    #     x = self.input_norm(x)
    #     x = self.relu(x)

    #     # encoder
    #     for conv, norm in zip(self.encoder_convs, self.encoder_norms):
    #         x = conv(x)
    #         x = norm(x)
    #         x = self.relu(x)

    #     # bottleneck
    #     for conv_a, norm_a, conv_b, norm_b in zip(self.bottleneck_convs_a, self.bottleneck_norms_a, self.bottleneck_convs_b, self.bottleneck_norms_b):
    #         x_a = conv_a(x)
    #         x_a = norm_a(x_a)
    #         x_a = self.relu(x_a)
    #         x_b = conv_b(x_a)
    #         x_b = norm_b(x_b)
    #         x = Add()([x, x_b])

    #     # decoder
    #     for conv, norm in zip(self.decoder_convs, self.decoder_norms):
    #         x = conv(x)
    #         x = norm(x)
    #         x = self.relu(x)

    #     x = self.output_conv(x)
    #     x = self.tanh(x)

    #     return x
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    # def __init__(self):
    #     super(Generator, self).__init__()
    #     self.channel = 64

    #     self.conv_input = Conv2D(self.channel, kernel_size=7, strides=1, padding='same', use_bias=False, name='conv2d_input')
    #     self.in_input = LayerNormalization(axis=[1, 2], name='in_input')

    #     self.encoder_layers = []
    #     for i in range(2):
    #         self.encoder_layers.append(
    #             Conv2D(self.channel * 2, kernel_size=4, strides=2, padding='same', use_bias=False, name=f'conv2d_{i}')
    #         )
    #         self.encoder_layers.append(
    #             LayerNormalization(axis=[1, 2], name=f'in_conv_{i}')
    #         )
    #         self.channel *= 2

    #     self.bottleneck_layers = []
    #     for i in range(6):
    #         self.bottleneck_layers.append(
    #             Conv2D(self.channel, kernel_size=3, strides=1, padding='same', use_bias=False, name=f'conv_res_a_{i}')
    #         )
    #         self.bottleneck_layers.append(
    #             LayerNormalization(axis=[1, 2], name=f'in_res_a_{i}')
    #         )
    #         self.bottleneck_layers.append(
    #             Conv2D(self.channel, kernel_size=3, strides=1, padding='same', use_bias=False, name=f'conv_res_b_{i}')
    #         )
    #         self.bottleneck_layers.append(
    #             LayerNormalization(axis=[1, 2], name=f'in_res_b_{i}')
    #         )

    #     self.decoder_layers = []
    #     for i in range(2):
    #         self.decoder_layers.append(
    #             Conv2DTranspose(self.channel // 2, kernel_size=4, strides=2, padding='same', use_bias=False, name=f'deconv_{i}')
    #         )
    #         self.decoder_layers.append(
    #             LayerNormalization(axis=[1, 2], name=f'in_decon_{i}')
    #         )
    #         self.channel //= 2

    #     self.conv_output = Conv2D(3, kernel_size=7, strides=1, padding='same', use_bias=False, name='output')

    # def call(self, input_, angles, training=False):
    #     style_dim = angles.shape[-1]

    #     angles_reshaped = tf.reshape(angles, [-1, 1, 1, style_dim])
    #     angles_tiled = tf.tile(angles_reshaped, [1, tf.shape(input_)[1], tf.shape(input_)[2], 1])
    #     x = tf.concat([input_, angles_tiled], axis=3)

    #     x = self.conv_input(x)
    #     x = self.in_input(x)
    #     # x = relu(x)
    #     x = tf.nn.relu(x)

    #     for i in range(2):
    #         x = self.encoder_layers[2 * i](x)
    #         x = self.encoder_layers[2 * i + 1](x)
    #         # x = relu(x)
    #         x = tf.nn.relu(x)

    #     for i in range(6):
    #         x_a = self.bottleneck_layers[4 * i](x)
    #         x_a = self.bottleneck_layers[4 * i + 1](x_a)
    #         # x_a = relu(x_a)
    #         x_a = tf.nn.relu(x_a)
    #         x_b = self.bottleneck_layers[4 * i + 2](x_a)
    #         x_b = self.bottleneck_layers[4 * i + 3](x_b)
    #         x = x + x_b

    #     for i in range(2):
    #         x = self.decoder_layers[2 * i](x)
    #         x = self.decoder_layers[2 * i + 1](x)
    #         # x = relu(x)
    #         x = tf.nn.relu(x)

    #     x = self.conv_output(x)
    #     # x = tanh(x)
    #     x = tf.nn.tanh(x)

    #     return x
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

# def vgg_16(inputs):
#     end_points = {}
#     net = inputs

#     # Block 1
#     net = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(net)
#     net = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(net)
#     end_points['conv1_2'] = net
#     net = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(net)

#     # Block 2
#     net = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(net)
#     net = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(net)
#     end_points['conv2_2'] = net
#     net = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(net)

#     # Block 3
#     net = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(net)
#     net = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(net)
#     net = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(net)
#     end_points['conv3_3'] = net
#     net = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(net)

#     # Block 4
#     net = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(net)
#     net = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(net)
#     net = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(net)
#     end_points['conv4_3'] = net
#     net = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(net)

#     # Block 5
#     net = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(net)
#     net = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(net)
#     net = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(net)
#     end_points['conv5_3'] = net
#     net = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(net)

#     return net, end_points


class GazeRedirectGAN(tf.keras.Model):
    # def __init__(self, generator, discriminator, vgg_model):
    def __init__(self, generator):
        super(GazeRedirectGAN, self).__init__()
        self.generator = generator
        # self.discriminator = discriminator
        # self.vgg_model = vgg_model

    # def compile(self, gen_optimizer, disc_optimizer, loss_fn, perceptual_loss_fn):
    def compile(self, gen_optimizer, loss_fn):
        super(GazeRedirectGAN, self).compile()
        self.gen_optimizer = gen_optimizer
        # self.disc_optimizer = disc_optimizer
        self.loss_fn = loss_fn
        # self.perceptual_loss_fn = perceptual_loss_fn

    def train_step(self, data):
        (img_t, gaze_target), (img, gaze_real) = data
        with tf.GradientTape(persistent=True) as tape:
            output_image = self.generator(img, gaze_target)
            corr_loss = self.loss_fn(img_t, output_image)

            recon_image = self.generator(output_image, gaze_real)
            recon_loss = self.loss_fn(img, recon_image)


        corr_gradients = tape.gradient(corr_loss, self.generator.trainable_variables)
        recon_gradients = tape.gradient(recon_loss, self.generator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(corr_gradients, self.generator.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(recon_gradients, self.generator.trainable_variables))

        return {"corr_loss": corr_loss, "recon_loss": recon_loss}

    # def train_step(self, data):
    #     # x_real, (gaze_real, gaze_target) = data
    #     (x_t, gaze_target), (x_real, gaze_real) = data


    #     target_angle = tf.random.normal([1, 1, 1, 2])

    #     with tf.GradientTape() as tape:
    #         # Generate images
    #         x_fake = self.generator(x_real, target_angle, training=True)
    #         vector_field, brightness_map = self.generator(x_real, target_angle, training=True)
    #         output_image = adjust_brightness(warp_image(input_image, vector_field), brightness_map)
        
    #         # x_reconstructed = self.generator(x_fake, gaze_real, training=True)

    #         #####MYCODESTARTHERE###
    #         # gen_loss = self.loss_fn(x_real, x_reconstructed)

    #         # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #         # print(x_fake.shape)
    #         # print(x_t.shape)
    #         corr_loss = self.loss_fn(x_fake, x_t)
    #         recon_loss = 0#self.loss_fn(x_reconstructed, x_real)
    #         #####MYCODEENDHERE#####

    #         # Discriminator predictions
    #         # real_output, real_gaze = self.discriminator(x_real, training=True)
    #         # fake_output, fake_gaze = self.discriminator(x_fake, training=True)

    #         # Calculate the losses
    #         # gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output) + self.perceptual_loss_fn(x_fake, x_real)
    #         # disc_loss_real = self.loss_fn(tf.ones_like(real_output), real_output)
    #         # disc_loss_fake = self.loss_fn(tf.zeros_like(fake_output), fake_output)
    #         # disc_loss = (disc_loss_real + disc_loss_fake) / 2

    #         # Gaze estimation loss
    #         # gaze_loss = tf.reduce_mean(tf.square(gaze_real - real_gaze)) + tf.reduce_mean(tf.square(gaze_target - fake_gaze))
    #         # total_disc_loss = disc_loss + gaze_loss

    #     # Calculate gradients
    #     gen_gradients = tape.gradient(corr_loss, self.generator.trainable_variables)
    #     # disc_gradients = tape.gradient(total_disc_loss, self.discriminator.trainable_variables)

    #     # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #     # print(corr_loss)
    #     # print(gen_gradients)

    #     # Apply gradients
    #     self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
    #     # self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

    #     #####MYCODESTARTHERE###
    #     # gen_gradients = tape.gradient(recon_loss, self.generator.trainable_variables)
    #     # self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
    #     #####MYCODEENDHERE#####

    #     #####TESTING#####
    #     # gen_loss = recon_loss#####
    #     # disc_loss = 0####
    #     # gaze_loss = 0####
    #     #####TESTING#####

    #     # return {"gen_loss": gen_loss, "disc_loss": disc_loss, "gaze_loss": gaze_loss}
    #     return {"corr_loss": corr_loss, "recon_loss": recon_loss}

    def call(self, inputs, training=False):
        x_real, gaze_target = inputs
        return self.generator(x_real, gaze_target, training=training)

    
