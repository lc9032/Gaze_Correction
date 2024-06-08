from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, ReLU, Conv2DTranspose, Activation, LayerNormalization, Add, Concatenate, BatchNormalization
import tensorflow as tf
# import tf_slim as slim

# import matplotlib.pyplot as plt
# import numpy as np

import tensorflow_addons as tfa


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
        
    
    def call(self, input_image, target_angle, landmarks):
        # Expand target_angle to match the spatial dimensions of input_image
        batch_size = tf.shape(input_image)[0]
        height = tf.shape(input_image)[1]
        width = tf.shape(input_image)[2]
        
        target_angle = tf.reshape(target_angle, (batch_size, 1, 1, 2))
        target_angle = tf.tile(target_angle, [1, height, width, 1])

        landmarks_reshaped_x, landmarks_reshaped_y = tf.split(landmarks, num_or_size_splits=2, axis=2)
        landmarks_reshaped_x = tf.reshape(landmarks_reshaped_x, (batch_size, 1, 1, 11))
        landmarks_reshaped_x = tf.tile(landmarks_reshaped_x, [1, height, width, 1])

        landmarks_reshaped_y = tf.reshape(landmarks_reshaped_y, (batch_size, 1, 1, 11))
        landmarks_reshaped_y = tf.tile(landmarks_reshaped_y, [1, height, width, 1])
        
        x = tf.concat([input_image, target_angle, landmarks_reshaped_x, landmarks_reshaped_y], axis=-1)

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
        flow_x = tf.tanh(flow_x)

        brightness_map = tf.math.sigmoid(brightness_x, name=None)#self.brightness_conv(brightness_x)
        # print("brightness_map:", brightness_map.shape)


        # Warp the input image using the flow field
        warped_image = self.warp(input_image, flow_x, landmarks)
        
        # print("!!!!!!!!!!!!!!")
        # print("warped_image:" , warped_image)

        # Adjust brightness using the brightness map
        output_image = self.adjust_brightness(warped_image, brightness_map)

        # print("output_image:" , output_image)


        return output_image

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
        dest_points_y = source_points_y + dest_points[..., 0] * (1.0)
        dest_points_x = source_points_x + dest_points[..., 1] * (4.0)
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
        (img_t, gaze_target, landmarks_t), (img, gaze_real, landmarks) = data
        with tf.GradientTape(persistent=True) as tape:
            output_image = self.generator(img, gaze_target, landmarks)
            corr_loss = self.loss_fn(img_t, output_image)


            recon_image = self.generator(output_image, gaze_real, landmarks_t)
            recon_loss = self.loss_fn(img, recon_image)

            L_total = 0.99 * corr_loss + 0.01 * recon_loss

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

    
