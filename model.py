from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, ReLU, Conv2DTranspose, Activation, LayerNormalization, Add, Concatenate, BatchNormalization
import tensorflow.keras.backend as K
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
# import tf_slim as slim

# import matplotlib.pyplot as plt
# import numpy as np

import tensorflow_addons as tfa

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model


from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam


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
        # self.bn1 = layers.BatchNormalization()
        # self.relu1 = layers.ReLU()
        
    def call(self, inputs):
        x = self.conv1(inputs)
        # x = self.bn1(x)
        # x = self.relu1(x)
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
        
        # self.final_conv = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='tanh')
        self.final_conv = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='linear')

        # # Final layers for producing flow field and brightness map
        # self.flow_conv = layers.Conv2D(2, kernel_size=4, strides=1, padding='same', activation=None)
        # self.brightness_conv = layers.Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')




        # self.conv_en0 = layers.Conv2D(32, kernel_size=7, strides=1, padding='same')
        # self.bn_en0 = layers.BatchNormalization()
        self.relu_en0 = layers.ReLU()

        # self.conv_en1 = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')
        # self.bn_en1 = layers.BatchNormalization()
        # self.relu_en1 = layers.ReLU()

        # self.conv_en2 = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')
        # self.bn_en2 = layers.BatchNormalization()
        # self.relu_en2 = layers.ReLU()

        # self.conv_en3 = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')
        # self.bn_en3 = layers.BatchNormalization()
        # self.relu_en3 = layers.ReLU()

        # self.conv_b = ConvBlock(256)

        # self.conv_de0 = UpConvBlock(128, kernel_size=4, strides=2, padding='same')
        # self.bn_de0 = layers.BatchNormalization()
        # self.relu_de0 = layers.ReLU()

        # self.conv_de1 = UpConvBlock(64, kernel_size=4, strides=2, padding='same')
        # self.bn_de1 = layers.BatchNormalization()
        # self.relu_de1 = layers.ReLU()
        # self.conv_de2 = UpConvBlock(32, kernel_size=4, strides=2, padding='same')
        # self.bn_de2 = layers.BatchNormalization()
        # self.relu_de2 = layers.ReLU()

        # self.final_conv = layers.Conv2D(3, kernel_size=7, strides=1, padding='same')
        
    
    def call(self, input_image, pose, target_angle, landmarks):

        # Expand target_angle to match the spatial dimensions of input_image
        batch_size = tf.shape(input_image)[0]
        height = tf.shape(input_image)[1]
        width = tf.shape(input_image)[2]

        pose = tf.reshape(pose, (batch_size, 1, 1, 1))
        pose = tf.tile(pose, [1, height, width, 1])
        
        target_angle = tf.reshape(target_angle, (batch_size, 1, 1, 2))
        target_angle = tf.tile(target_angle, [1, height, width, 1])

        landmarks_reshaped_x, landmarks_reshaped_y = tf.split(landmarks, num_or_size_splits=2, axis=2)
        landmarks_reshaped_x = tf.reshape(landmarks_reshaped_x, (batch_size, 1, 1, 11))
        landmarks_reshaped_x = tf.tile(landmarks_reshaped_x, [1, height, width, 1])

        landmarks_reshaped_y = tf.reshape(landmarks_reshaped_y, (batch_size, 1, 1, 11))
        landmarks_reshaped_y = tf.tile(landmarks_reshaped_y, [1, height, width, 1])
        
        x = tf.concat([input_image, pose, target_angle, landmarks_reshaped_x, landmarks_reshaped_y], axis=-1)

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

        # Flow field and brightness map
        # flow_field = self.flow_conv(x)
        # print("flow_field:", flow_field.shape)
        flow_x = tf.tanh(flow_x)
        # flow_x = self.relu_en0(flow_x)
        brightness_map = tf.math.sigmoid(brightness_x, name=None)#self.brightness_conv(brightness_x)

        # Warp the input image using the flow field
        warped_image = self.warp(input_image, flow_x, landmarks)
        
        # Adjust brightness using the brightness map
        output_image = self.adjust_brightness(warped_image, brightness_map)

        return output_image


# Epoch 1/100
# 1st polling (None, 16, 32, 32)
# 2nd polling (None, 8, 16, 64)
# 3rd polling (None, 4, 8, 128)
# 4th polling (None, 2, 4, 256)
# 1st up-conv (None, 4, 8, 128)
# 2nd up-conv (None, 8, 16, 64)
# 3rd up-conv (None, 8, 16, 32)
# upconv4 (None, 32, 64, 32)
# final_conv (None, 32, 64, 3)


    # def call(self, input_image, pose, target_angle, landmarks):
    #     # Expand target_angle to match the spatial dimensions of input_image
    #     batch_size = tf.shape(input_image)[0]
    #     height = tf.shape(input_image)[1]
    #     width = tf.shape(input_image)[2]

    #     pose = tf.reshape(pose, (batch_size, 1, 1, 1))
    #     pose = tf.tile(pose, [1, height, width, 1])
        
    #     target_angle = tf.reshape(target_angle, (batch_size, 1, 1, 2))
    #     target_angle = tf.tile(target_angle, [1, height, width, 1])

    #     landmarks_reshaped_x, landmarks_reshaped_y = tf.split(landmarks, num_or_size_splits=2, axis=2)
    #     landmarks_reshaped_x = tf.reshape(landmarks_reshaped_x, (batch_size, 1, 1, 11))
    #     landmarks_reshaped_x = tf.tile(landmarks_reshaped_x, [1, height, width, 1])

    #     landmarks_reshaped_y = tf.reshape(landmarks_reshaped_y, (batch_size, 1, 1, 11))
    #     landmarks_reshaped_y = tf.tile(landmarks_reshaped_y, [1, height, width, 1])
        
    #     x = tf.concat([input_image, pose, target_angle, landmarks_reshaped_x, landmarks_reshaped_y], axis=-1)


    #     # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #     # print('Input Shape:', x.shape)

    #     x = self.conv_en0(x)
    #     x = self.bn_en0(x)
    #     x = self.relu_en0(x)
    #     # x = MaxPooling2D((2, 2))(x)

    #     # print('After conv_en0:', x.shape)

    #     x = self.conv_en1(x)
    #     x = self.bn_en1(x)
    #     x = self.relu_en1(x)
    #     # x = MaxPooling2D((2, 2))(x)

    #     # print('After conv_en1:', x.shape)

    #     x = self.conv_en2(x)
    #     x = self.bn_en2(x)
    #     x = self.relu_en2(x)
    #     # x = MaxPooling2D((2, 2))(x)

    #     x = self.conv_en3(x)
    #     x = self.bn_en3(x)
    #     x = self.relu_en3(x)
    #     # x = MaxPooling2D((2, 2))(x)

    #     # print('After conv_en2:', x.shape)

    #     for i in range(6):
    #         x = self.conv_b(x)

    #         # print(f'After conv_b {i+1}:', x.shape)


    #     x = self.conv_de0(x)
    #     x = self.bn_de0(x)
    #     x = self.relu_de0(x)

    #     x = self.conv_de1(x)
    #     x = self.bn_de1(x)
    #     x = self.relu_de1(x)

    #     # print('After conv_de1:', x.shape)

    #     x = self.conv_de2(x)
    #     x = self.bn_de2(x)
    #     x = self.relu_de2(x)

    #     # print('After conv_de2:', x.shape)

    #     # print(x.shape)

    #     x = self.final_conv(x)

    #     # print('After final_conv:', x.shape)

    #     x = tf.tanh(x)

    #     return x






    def remove_duplicate_points(self, source_points, dest_points):
        """
        Remove duplicate points from source_points and the corresponding points in dest_points.
        
        Args:
            source_points (tf.Tensor): The source points tensor of shape [batch_size, num_points, 2].
            dest_points (tf.Tensor): The destination points tensor of shape [batch_size, num_points, 2].
        
        Returns:
            tuple: Two tensors representing the unique source and destination points.
        """
        def unique_points(points, other_points):

            reshaped_points = tf.reshape(points, [-1, 2])
            reshaped_other_points = tf.reshape(other_points, [-1, 2])

            flattened_points = tf.strings.reduce_join(tf.as_string(reshaped_points), axis=1, separator=',')
            
            # Find unique values
            unique_points, idx = tf.unique(flattened_points)


            first_occurrence_indices = tf.math.unsorted_segment_min(tf.range(tf.size(flattened_points)), idx, tf.size(unique_points))
            
            # Split back to original shape
            unique_points = tf.strings.split(unique_points, ',')
            
            # Convert strings to numbers
            unique_points = tf.strings.to_number(unique_points)
            
            # Reshape to original shape
            unique_points = tf.reshape(unique_points, [-1, 2])
            
            # Gather corresponding destination points
            unique_dest_points = tf.gather(reshaped_other_points, first_occurrence_indices)

            max_points = 8
            if unique_points.shape[0]:
                max_points = min(unique_points.shape[0], unique_dest_points.shape[0])

            if max_points is not None:
                unique_points = unique_points[:max_points]
                unique_dest_points = unique_dest_points[:max_points]
                padding = [[0, max_points - tf.shape(unique_points)[0]], [0, 0]]
                unique_points = tf.pad(unique_points, padding)
                unique_dest_points = tf.pad(unique_dest_points, padding)

            return unique_points, unique_dest_points
        
        unique_source_points, unique_dest_points = tf.map_fn(
            lambda x: unique_points(x[0], x[1]), 
            (source_points, dest_points), 
            dtype=(tf.float32, tf.float32)
        )

        return unique_source_points, unique_dest_points



    # def keep_top_difference_points(self, source_points, dest_points, top_k = 16):
    #     """
    #     Keep only the top_k pairs of points with the largest differences between dest_points and source_points.
        
    #     Args:
    #         source_points (tf.Tensor): The source points tensor of shape [batch_size, num_points, 2].
    #         dest_points (tf.Tensor): The destination points tensor of shape [batch_size, num_points, 2].
    #         top_k (int): The number of pairs to keep based on the largest differences.
        
    #     Returns:
    #         tuple: Two tensors with only the top_k pairs of points.
    #     """
    #     def filter_points_in_batch(source, dest):
    #         difference = tf.abs(dest - source)
    #         difference_sum = tf.reduce_sum(difference, axis=-1)

    #         # Get the top_k indices based on the differences
    #         _, top_k_indices = tf.nn.top_k(difference_sum, k=top_k, sorted=True)

    #         # Select the top_k points based on the indices
    #         top_source_points = tf.gather(source, top_k_indices)
    #         top_dest_points = tf.gather(dest, top_k_indices)

    #         return top_source_points, top_dest_points

    #     filtered_source_points, filtered_dest_points = tf.map_fn(
    #         lambda x: filter_points_in_batch(x[0], x[1]), 
    #         (source_points, dest_points), 
    #         dtype=(tf.float32, tf.float32)
    #     )

    #     return filtered_source_points, filtered_dest_points

    def keep_top_and_last_difference_points(self, source_points, dest_points, top_k = 8, last_k = 8):
        """
        Keep the top_k and last_k pairs of points based on the differences between dest_points and source_points.
        
        Args:
            source_points (tf.Tensor): The source points tensor of shape [batch_size, num_points, 2].
            dest_points (tf.Tensor): The destination points tensor of shape [batch_size, num_points, 2].
            top_k (int): The number of pairs to keep based on the largest differences.
            last_k (int): The number of pairs to keep based on the smallest differences.
        
        Returns:
            tuple: Two tensors with only the top_k and last_k pairs of points.
        """
        def filter_points_in_batch(source, dest):
            difference = tf.abs(dest - source)
            difference_sum = tf.reduce_sum(difference, axis=-1)

            # Get the top_k and last_k indices based on the differences
            top_k_values, top_k_indices = tf.nn.top_k(difference_sum, k=top_k, sorted=True)
            last_k_values, last_k_indices = tf.nn.top_k(-difference_sum, k=last_k, sorted=True)

            # Select the top_k and last_k points based on the indices
            top_source_points = tf.gather(source, top_k_indices)
            top_dest_points = tf.gather(dest, top_k_indices)
            last_source_points = tf.gather(source, last_k_indices)
            last_dest_points = tf.gather(dest, last_k_indices)

            # Concatenate the top and last points
            combined_source_points = tf.concat([top_source_points, last_source_points], axis=0)
            combined_dest_points = tf.concat([top_dest_points, last_dest_points], axis=0)


            # if max_points is not None:
            #     unique_points = unique_points[:max_points]
            #     unique_dest_points = unique_dest_points[:max_points]
            #     padding = [[0, max_points - tf.shape(unique_points)[0]], [0, 0]]
            #     unique_points = tf.pad(unique_points, padding)
            #     unique_dest_points = tf.pad(unique_dest_points, padding)

            return combined_source_points, combined_dest_points

        filtered_source_points, filtered_dest_points = tf.map_fn(
            lambda x: filter_points_in_batch(x[0], x[1]), 
            (source_points, dest_points), 
            dtype=(tf.float32, tf.float32)
        )

        return filtered_source_points, filtered_dest_points



    def warp(self, input_image, flow_field, landmarks):

        batch_size, height, width, channels = tf.shape(input_image)[0], tf.shape(input_image)[1], tf.shape(input_image)[2], tf.shape(input_image)[3]

        # Creating source_points manually
        y_indices = tf.linspace(0.0, 32.0 - 1.0, height)
        x_indices = tf.linspace(0.0, 64.0 - 1.0, width)
        y_grid, x_grid = tf.meshgrid(y_indices, x_indices, indexing='ij')
        source_points = tf.stack([y_grid, x_grid], axis=-1)  # shape: (height, width, 2)
        source_points = tf.reshape(source_points, [-1, 2])   # shape: (height * width, 2)
        source_points = tf.expand_dims(source_points, axis=0)  # shape: (1, height * width, 2)
        source_points = tf.tile(source_points, [batch_size, 1, 1])  # shape: (batch_size, height * width, 2)

        # print(source_points)

        # source_points = flow_field[..., :32, :]
        dest_points = flow_field[..., :64, :]
        # source_points = flow_field[..., :1, :]
        # dest_points = flow_field[..., 1:2, :]

        warped_image = input_image

        # source_points = tf.reshape(source_points, [batch_size, -1, 2])
        dest_points = tf.reshape(dest_points, [batch_size, -1, 2])

        # source_points = (source_points + 1.0) / 2.0
        source_points_y = source_points[..., 0]
        source_points_x = source_points[..., 1]
        # source_points = tf.stack([source_points_y, source_points_x], axis=-1)

        # dest_points = (dest_points + 1.0) / 2.0
        dest_points_y = source_points_y + dest_points[..., 0] * (3.0)
        dest_points_x = source_points_x + dest_points[..., 1] * (6.0)
        dest_points = tf.stack([dest_points_y, dest_points_x], axis=-1)

        # source_points, dest_points = self.remove_close_points(source_points,dest_points)
        source_points, dest_points = self.keep_top_and_last_difference_points(source_points,dest_points)

        new_source_points = tf.constant([[[0.0, 0.0], [31.0, 0.0], [0.0, 63.0], [31.0, 63.0],
                                          [16.0, 0.0], [0.0, 31.0], [31.0, 31.0], [16.0, 63.0]]], dtype=tf.float32)
        new_dest_points = tf.constant([[[0.0, 0.0], [31.0, 0.0], [0.0, 63.0], [31.0, 63.0],
                                        [16.0, 0.0], [0.0, 31.0], [31.0, 31.0], [16.0, 63.0]]], dtype=tf.float32)
        new_source_points = tf.tile(new_source_points, [batch_size, 1, 1])
        new_dest_points = tf.tile(new_dest_points, [batch_size, 1, 1])

        landmarks = tf.reverse(landmarks, axis=[-1])

        source_points = tf.concat([new_source_points, source_points], axis=1)
        # source_points = tf.concat([landmarks, source_points], axis=1)
        dest_points = tf.concat([new_dest_points, dest_points], axis=1)
        # dest_points = tf.concat([landmarks, dest_points], axis=1)

        source_points,dest_points = self.remove_duplicate_points(source_points, dest_points)
        dest_points,source_points = self.remove_duplicate_points(dest_points, source_points)

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(source_points)
        print(dest_points)
        print(landmarks)

        # for _ in (0,2):
        warped_image, _ = tfa.image.sparse_image_warp(warped_image, source_points, dest_points)

        return warped_image












    
    # def warp(self, images, vector_fields, landmarks):
    #     """
    #     Warps a batch of images according to the given batch of vector fields.
        
    #     Args:
    #     images: A tensor of shape (batch, 32, 64, 3) representing the batch of eye images.
    #     vector_fields: A tensor of shape (batch, 32, 64, 2) representing the batch of displacement vectors.
        
    #     Returns:
    #     warped_images: A tensor of shape (batch, 32, 64, 3) representing the batch of warped eye images.
    #     """
    #     batch_size, height, width, channels = images.shape
    #     vector_fields = tf.cast(vector_fields, tf.float32)
        
    #     # Create a grid of coordinates corresponding to the image
    #     x, y = tf.meshgrid(tf.range(width), tf.range(height))
    #     x = tf.cast(x, tf.float32)
    #     y = tf.cast(y, tf.float32)
    #     grid = tf.stack([x, y], axis=-1)  # shape: (32, 64, 2)
        
    #     def warp_single_image(image_and_field):
    #         image, vector_field = image_and_field
    #         new_coords = grid + vector_field*0.5
            
    #         # Flatten coordinates
    #         flattened_coords = tf.reshape(new_coords, [-1, 2])
            
    #         # Clip coordinates to be within valid range
    #         x_new = tf.clip_by_value(flattened_coords[:, 0], 0, width - 1)
    #         y_new = tf.clip_by_value(flattened_coords[:, 1], 0, height - 1)

    #         sampled_indices = tf.stack([y_new, x_new], axis=-1)
    #         gathered_pixels = tf.gather_nd(image, tf.cast(sampled_indices, tf.int32))
    #         sampled_image = tf.reshape(gathered_pixels, (height, width, channels))
            
    #         return sampled_image
        
    #     # Use tf.map_fn to apply the warp to each image in the batch
    #     warped_images = tf.map_fn(warp_single_image, (images, vector_fields), dtype=tf.float32)
        
    #     return warped_images


    def adjust_brightness(self, warped_image, brightness_map):
        brightness_map = tf.image.resize(brightness_map, (tf.shape(warped_image)[1], tf.shape(warped_image)[2]))
        adjusted_image = warped_image * (brightness_map*0.2+0.8)
        return adjusted_image




class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_en1 = layers.Conv2D(32, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv_en2 = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv_en3 = layers.Conv2D(128, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv_en4 = layers.Conv2D(256, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv_en5 = layers.Conv2D(512, kernel_size=4, strides=2, padding='same', activation='relu')

        # filter_size = int(2048 / 2**5)
        # self.conv_en6 = layers.Conv2D(1, kernel_size=filter_size, strides=1, padding='same', use_bias=False)
        # self.conv_en7 = layers.Conv2D(2, kernel_size=filter_size, strides=1, padding='same', use_bias=False)

        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(3, activation='linear')

        self.relu = layers.ReLU()

    def call(self, img_input, pose_input):
        batch_size = tf.shape(img_input)[0]
        height = tf.shape(img_input)[1]
        width = tf.shape(img_input)[2]
        
        pose = tf.reshape(pose_input, (batch_size, 1, 1, 1))
        pose = tf.tile(pose, [1, height, width, 1])

        # Concatenate image and pose inputs
        concatenated = Concatenate()([img_input, pose])

        # Convolutional layers for feature extraction

        x = self.conv_en1(concatenated)

        x = self.conv_en2(x)
        x = self.conv_en3(x)
        x = self.conv_en4(x)
        x = self.conv_en5(x)

        x = Flatten()(x)

        # Dense layers for regression
        x = self.dense1(x)
        x = self.dense2(x)
        
        # Output layer for gaze prediction
        x = self.dense3(x)

        x_reg, x_gan = tf.split(x, num_or_size_splits=[2, 1], axis=-1)

        x_reg = Lambda(lambda x: x * 20.0)(x_reg)  # Scaling to range [-20, 20]

        # model = tf.keras.Model(inputs=[img_input, pose_input], outputs=outputs, name='gaze_prediction_model')

        x_reg = tf.reshape(x_reg, (batch_size, 1, 2))

        # x_gan = self.relu(x_gan)
        x_gan = tf.tanh(x_gan)

        return x_gan, x_reg

    # def predict_gaze(self, img, pose):
    #     # img, pose, _ = preprocess_data(img, pose, 0, 0)  # Only need to preprocess the image and pose
    #     # img = tf.expand_dims(img, axis=0)  # Add batch dimension
    #     # pose = tf.expand_dims(pose, axis=0)  # Add batch dimension
    #     # prediction = model.predict([img, pose])
    #     prediction = self.model([img, pose])
    #     return prediction

    # def gaze_loss(self, image1, pose1, image2, pose2):
    #     # Predict gaze directions for both images
    #     pred_gaze1 = self.predict_gaze(image1, pose1)
    #     pred_gaze2 = self.predict_gaze(image2, pose2)

    #     # Calculate the loss (Mean Squared Error between the two gaze directions)
    #     loss = tf.reduce_mean(tf.square(pred_gaze1 - pred_gaze2))
    #     loss = 0.01 * loss

    #     return loss




class GazeRedirectGAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(GazeRedirectGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

        # self.gaze_prediction = GazePrediction()

    def compile(self, gen_optimizer, disc_optimizer, loss_fn):
        super(GazeRedirectGAN, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.loss_fn = loss_fn
        # self.perceptual_loss_fn = perceptual_loss_fn

    def train_step(self, data):
        (img_t, p_t, gaze_target, landmarks_t), (img, p, gaze_real, landmarks) = data
        with tf.GradientTape(persistent=True) as tape:
            output_image = self.generator(img, p, gaze_target, landmarks)
            corr_loss = self.loss_fn(img_t, output_image)
            # gaze_loss = self.gaze_prediction.gaze_loss(img_t, p_t, output_image, p_t)

            recon_image = self.generator(output_image, p_t, gaze_real, landmarks_t)
            recon_loss = self.loss_fn(img, recon_image)
            # gaze_r_loss = self.gaze_prediction.gaze_loss(img, p, recon_image, p)

            lc = corr_loss #+ gaze_loss
            lr = recon_loss #+ gaze_r_loss
            L_total = 0.8 * (lc) + 0.2 * (lr)

            ##############################################################################################
            gan_real, gaze_real_p = self.discriminator(img, p, training=True)
            gan_fake, gaze_fake_p = self.discriminator(output_image, p_t, training=True)

            # gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output) + self.loss_fn(output_image, img)
            # disc_loss_real = self.loss_fn(tf.ones_like(real_output), real_output)
            # disc_loss_fake = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            # disc_loss = (disc_loss_real + disc_loss_fake) / 2

            # gaze_loss = tf.reduce_mean(tf.square(gaze_real - real_gaze)) + tf.reduce_mean(tf.square(gaze_target - fake_gaze))

            # total_disc_loss = disc_loss + gaze_loss


            #---
            # batch_size = tf.shape(img)[0]
            # eps = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
            # interpolated = eps * img + (1. - eps) * output_image
            # gan_inter, _ = self.discriminator(interpolated, p, training=True)
            # grad = tf.gradients(gan_inter, interpolated)[0]
            # slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
            # gp = tf.reduce_mean(tf.square(slopes - 1.))
            # d_loss = (-tf.reduce_mean(gan_real) + tf.reduce_mean(gan_fake) + 10. * gp)
            # g_loss = -tf.reduce_mean(gan_fake)

            # reg_loss_d = self.loss_fn(gaze_real, reg_real)
            # reg_loss_g = self.loss_fn(gaze_target, reg_fake)

            # # g_loss = (self.g_loss + 5.0 * self.reg_g_loss + 50.0 * self.recon_loss + 100.0 * self.s_loss + 100.0 * self.c_loss)
            # # d_loss = self.d_loss + 5.0 * self.reg_d_loss

            # L_total = L_total + g_loss + reg_loss_g
            # d_loss = d_loss + reg_loss_d


            d_loss = 2.0*self.loss_fn(gaze_real, gaze_real_p) - 0.5*tf.reduce_mean(gan_real) + 0.5*tf.reduce_mean(gan_fake)

            L_total = 80.0*L_total + 0.2*self.loss_fn(gaze_target, gaze_fake_p) - 0.4*tf.reduce_mean(gan_fake)
            ##############################################################################################


        # corr_gradients = tape.gradient(corr_loss, self.generator.trainable_variables)
        # recon_gradients = tape.gradient(recon_loss, self.generator.trainable_variables)

        # self.gen_optimizer.apply_gradients(zip(corr_gradients, self.generator.trainable_variables))
        # self.gen_optimizer.apply_gradients(zip(recon_gradients, self.generator.trainable_variables))

        # Compute gradients for the total loss
        gradients = tape.gradient(L_total, self.generator.trainable_variables)

        disc_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)

        # Apply gradients
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        return {"lc": lc, "lr": lr, "lt": L_total, "gr": tf.reduce_mean(gan_real), "gf": tf.reduce_mean(gan_fake)}


    def call(self, inputs, training=False):
        x_real, gaze_target = inputs
        return self.generator(x_real, gaze_target, training=training)


# class GazePrediction:
#     def __init__(self, model_path='gaze_prediction_model.h5'):
#         self.model = self.build_model()
#         self.model.load_weights(model_path)

#     def build_model(self, input_shape=(32, 64, 3)):
#         # Image input
#         img_input = Input(shape=input_shape, name='img_input')
#         pose_input = Input(shape=(1,), name='pose_input')

#         # Reshape and tile pose to match image input dimensions
#         pose = Lambda(lambda p: tf.tile(tf.reshape(p, (-1, 1, 1, 1)), [1, input_shape[0], input_shape[1], 1]))(pose_input)

#         # Concatenate image and pose inputs
#         concatenated = Concatenate()([img_input, pose])

#         # Convolutional layers for feature extraction
#         x = Conv2D(32, (3, 3), activation='relu', padding='same')(concatenated)
#         x = MaxPooling2D((2, 2))(x)

#         x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#         x = MaxPooling2D((2, 2))(x)

#         x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#         x = MaxPooling2D((2, 2))(x)

#         x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
#         x = MaxPooling2D((2, 2))(x)

#         x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
#         x = MaxPooling2D((2, 2))(x)

#         x = Flatten()(x)

#         # Dense layers for regression
#         x = Dense(128, activation='relu')(x)
#         x = Dense(64, activation='relu')(x)
        
#         # Output layer for gaze prediction
#         outputs = Dense(2, activation='linear')(x)  # Outputs in range [-1, 1]
#         outputs = Lambda(lambda x: x * 20.0)(outputs)  # Scaling to range [-20, 20]

#         model = tf.keras.Model(inputs=[img_input, pose_input], outputs=outputs, name='gaze_prediction_model')
#         return model

#     def predict_gaze(self, img, pose):
#         # img, pose, _ = preprocess_data(img, pose, 0, 0)  # Only need to preprocess the image and pose
#         # img = tf.expand_dims(img, axis=0)  # Add batch dimension
#         # pose = tf.expand_dims(pose, axis=0)  # Add batch dimension
#         # prediction = model.predict([img, pose])
#         prediction = self.model([img, pose])
#         return prediction

#     def gaze_loss(self, image1, pose1, image2, pose2):
#         # Predict gaze directions for both images
#         pred_gaze1 = self.predict_gaze(image1, pose1)
#         pred_gaze2 = self.predict_gaze(image2, pose2)

#         # Calculate the loss (Mean Squared Error between the two gaze directions)
#         loss = tf.reduce_mean(tf.square(pred_gaze1 - pred_gaze2))
#         loss = 0.01 * loss

#         return loss