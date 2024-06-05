from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, ReLU, Conv2DTranspose, Activation, LayerNormalization, Add, Concatenate, BatchNormalization
import tensorflow as tf
import tf_slim as slim

import matplotlib.pyplot as plt
import numpy as np


class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding='same'):
        super(ConvBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size, strides, padding)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, kernel_size, strides, padding)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.conv3 = layers.Conv2D(filters, kernel_size, strides, padding)
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.ReLU()
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        xr = self.relu1(x)

        x = self.conv2(xr)
        x = self.bn2(x)
        x = self.relu2(x)

        x = layers.add([x, xr])

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        return x

class UpConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=2, padding='same'):
        super(UpConvBlock, self).__init__()
        self.conv1 = layers.Conv2DTranspose(filters, kernel_size, strides, padding)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        # self.conv2 = layers.Conv2DTranspose(filters, kernel_size, strides, padding)
        # self.bn2 = layers.BatchNormalization()
        # self.relu2 = layers.ReLU()
        # self.conv3 = layers.Conv2DTranspose(filters, kernel_size, strides, padding)
        # self.bn3 = layers.BatchNormalization()
        # self.relu3 = layers.ReLU()
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu2(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # # x = layers.add([x, inputs])
        # x = self.relu3(x)
        return x

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        # Define the convolution blocks as shown in the architecture
        self.conv1 = ConvBlock(32)
        self.conv2 = ConvBlock(64)#self._conv_block(64, (3, 3), (1, 1))
        self.conv3 = ConvBlock(128)#self._conv_block(128, (3, 3), (1, 1))
        self.conv4 = ConvBlock(256)#self._conv_block(256, (3, 3), (1, 1))
        self.conv5 = ConvBlock(512)#self._conv_block(256, (3, 3), (1, 1))
        self.conv6 = ConvBlock(256)#self._conv_block(256, (3, 3), (1, 1))
        self.conv7 = ConvBlock(32)#self._conv_block(256, (3, 3), (1, 1))
        
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.upconv1 = UpConvBlock(128)#self._upconv_block(128, (3, 3), (2, 2))
        self.upconv2 = UpConvBlock(64)#self._upconv_block(64, (3, 3), (2, 2))
        self.upconv3 = UpConvBlock(32)#self._upconv_block(64, (3, 3), (2, 2))
        self.upconv4 = UpConvBlock(32)
        
        self.final_conv = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='tanh')


        # Final layers for producing flow field and brightness map
        # self.flow_conv = layers.Conv2D(2, kernel_size=4, strides=1, padding='same', activation=None)
        # self.brightness_conv = layers.Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')
        
    
    def call(self, input_image, target_angle):
        # Expand target_angle to match the spatial dimensions of input_image
        batch_size = tf.shape(input_image)[0]
        height = tf.shape(input_image)[1]
        width = tf.shape(input_image)[2]
        
        target_angle = tf.reshape(target_angle, (batch_size, 1, 1, 2))
        target_angle = tf.tile(target_angle, [1, height, width, 1])
        
        x = tf.concat([input_image, target_angle], axis=-1)

        x1 = self.conv1(x)
        p1 = self.pool1(x1)

        # print('1st polling', p1.shape)

        x2 = self.conv2(p1)
        p2 = self.pool2(x2)

        # print('2nd polling', p2.shape)

        x3 = self.conv3(p2)
        p3 = self.pool3(x3)

        # print('3rd polling', p3.shape)

        x4 = self.conv4(p3)
        p4 = self.pool4(x4)

        # print('4th polling', p4.shape)

        x = self.conv5(p4)
        x = self.upconv1(x)

        # print('1st up-conv', x.shape)

        x = tf.concat([x, p3], axis=-1)
        x = self.conv6(x)
        x = self.upconv2(x)

        # print('2nd up-conv', x.shape)

        x = tf.concat([x, p2], axis=-1)
        x = self.conv7(x)

        # print('3rd up-conv', x.shape)


        x = self.upconv3(x)
        x = self.upconv4(x)

        x = self.final_conv(x)


        # print('final_conv', x.shape)

        flow_x, brightness_x = tf.split(x, num_or_size_splits=[2, 1], axis=-1)
        # print('flow_x', flow_x.shape)
        # print('brightness_x', brightness_x.shape)

        # Flow field and brightness map
        # flow_field = self.flow_conv(x)
        # print("flow_field:", flow_field.shape)

        brightness_map = tf.math.sigmoid(brightness_x, name=None)#self.brightness_conv(brightness_x)
        # print("brightness_map:", brightness_map.shape)


        # Warp the input image using the flow field
        warped_image = self.warp(input_image, flow_x)

        # print("warped_image:" , warped_image)

        # Adjust brightness using the brightness map
        output_image = self.adjust_brightness(warped_image, brightness_map)

        # print("output_image:" , output_image)


        return x


    def warp(self, input_image, flow_field):
        batch_size, height, width, channels = tf.shape(input_image)[0], tf.shape(input_image)[1], tf.shape(input_image)[2], tf.shape(input_image)[3]
        # width_f = tf.cast(width, tf.float32)
        # height_f = tf.cast(height, tf.float32)
        
        # Generate a grid of coordinates
        grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
        grid_x = tf.cast(grid_x, tf.float32)
        grid_y = tf.cast(grid_y, tf.float32)

        grid = tf.stack([grid_y, grid_x], axis=-1)
        grid = tf.expand_dims(grid, axis=0)
        grid = tf.tile(grid, [batch_size, 1, 1, 1])
        
        # Apply flow field to the grid
        new_coords = grid + flow_field
        
        # Normalize coordinates to [-1, 1] range
        # new_coords = new_coords / tf.constant([height_f - 1, width_f - 1]) * 2.0 - 1.0
        new_coords = new_coords / tf.constant([32.0 - 1.0, 64.0 - 1.0]) * 2.0 - 1.0
        
        # Reshape new_coords to be compatible with tf.image.resize
        new_coords = tf.stack([new_coords[..., 1], new_coords[..., 0]], axis=-1)
        
        # Perform bilinear sampling
        # warped_image = tf.keras.layers.Lambda(
        #     lambda args: tf.keras.backend.resize_images(args[0], height, width, data_format='channels_last', interpolation='bilinear')
        # )([input_image, new_coords])
        warped_image = tf.keras.layers.Lambda(
            lambda args: tf.keras.backend.resize_images(args[0], 32, 64, data_format='channels_last', interpolation='bilinear')
        )([input_image, new_coords])
        
        return warped_image

    # def warp(self, input_image, flow_field):
    #     # batch_size, height, width, channels = tf.shape(input_image)
    #     batch_size, height, width, channels = tf.shape(input_image)[0], tf.shape(input_image)[1], tf.shape(input_image)[2], tf.shape(input_image)[3]
    #     flow_field = tf.image.resize(flow_field, (height, width))  # Resize flow field to match input image size
    #     grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
    #     grid_x = tf.cast(grid_x, tf.float32)
    #     grid_y = tf.cast(grid_y, tf.float32)

    #     grid_x = tf.expand_dims(grid_x, axis=0)  # Add batch dimension
    #     grid_y = tf.expand_dims(grid_y, axis=0)  # Add batch dimension
    #     grid_x = tf.expand_dims(grid_x, axis=-1)  # Add channel dimension
    #     grid_y = tf.expand_dims(grid_y, axis=-1)  # Add channel dimension

    #     flow_x, flow_y = tf.split(flow_field, 2, axis=-1)
    #     warped_grid_x = grid_x + flow_x
    #     warped_grid_y = grid_y + flow_y

    #     width_f = tf.cast(width, tf.float32)
    #     height_f = tf.cast(height, tf.float32)

    #     warped_grid_x = tf.clip_by_value(warped_grid_x, 0, width_f - 1)
    #     warped_grid_y = tf.clip_by_value(warped_grid_y, 0, height_f - 1)

    #     # Flatten the grid coordinates
    #     warped_grid_x_flat = tf.reshape(warped_grid_x, [-1])
    #     warped_grid_y_flat = tf.reshape(warped_grid_y, [-1])
        
    #     # Calculate flattened indices for gather operation
    #     warped_indices = tf.cast(warped_grid_y_flat * width_f + warped_grid_x_flat, tf.int32)
        
    #     # Flatten the input image
    #     input_image_flat = tf.reshape(input_image, [-1, channels])
        
    #     # Gather the warped image pixels
    #     warped_image_flat = tf.gather(input_image_flat, warped_indices)
        
    #     # Reshape the warped image back to the original dimensions
    #     warped_image = tf.reshape(warped_image_flat, [batch_size, height, width, channels])

    #     return warped_image

    def adjust_brightness(self, warped_image, brightness_map):
        brightness_map = tf.image.resize(brightness_map, (tf.shape(warped_image)[1], tf.shape(warped_image)[2]))
        adjusted_image = warped_image * brightness_map
        return adjusted_image


class GazeRedirectGAN(tf.keras.Model):
    def __init__(self, generator):
        super(GazeRedirectGAN, self).__init__()
        self.generator = generator


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

            L_total = 0.8 * corr_loss + 0.2 * recon_loss

        # corr_gradients = tape.gradient(corr_loss, self.generator.trainable_variables)
        # recon_gradients = tape.gradient(recon_loss, self.generator.trainable_variables)

        # self.gen_optimizer.apply_gradients(zip(corr_gradients, self.generator.trainable_variables))
        # self.gen_optimizer.apply_gradients(zip(recon_gradients, self.generator.trainable_variables))

        # Compute gradients for the total loss
        gradients = tape.gradient(L_total, self.generator.trainable_variables)

        # Apply gradients
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        return {"lc": corr_loss, "lr": recon_loss, "lt": L_total}

    def call(self, inputs, training=False):
        x_real, gaze_target = inputs
        return self.generator(x_real, gaze_target, training=training)

    
