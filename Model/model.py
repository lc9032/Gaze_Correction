
import tensorflow as tf # type: ignore
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.layers import  Concatenate, Flatten, Dense, Lambda# type: ignore

from Model.transformation import Transformation
import numpy as np # type: ignore
import cv2 # type: ignore


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
class AttentionLayer(layers.Layer):
    def __init__(self, filters):
        super(AttentionLayer, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=1)
        self.conv2 = layers.Conv2D(filters, kernel_size=1)
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, inputs):
        f = self.conv1(inputs)
        g = self.conv2(inputs)
        h = layers.add([f, g])
        h = self.sigmoid(h)
        return layers.multiply([inputs, h])
    
    
class UpConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=2, padding='same'):
        super(UpConvBlock, self).__init__()
        self.conv1 = layers.Conv2DTranspose(filters, kernel_size, strides, padding)
        # self.bn1 = layers.BatchNormalization()
        # self.relu1 = layers.ReLU()
        #############################################################################
        # super(UpConvBlock, self).__init__()
        # self.upsample = layers.UpSampling2D(size=strides)
        # self.conv = layers.Conv2D(filters, kernel_size, padding=padding)
        # self.bn = layers.BatchNormalization()
        # self.relu = layers.ReLU()
        
    def call(self, inputs):
        x = self.conv1(inputs)
        # x = self.bn1(x)
        # x = self.relu1(x)
        #############################################################################
        # x = self.upsample(inputs)
        # x = self.conv(x)
        # x = self.bn(x)
        # x = self.relu(x)

        return x

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.trans = Transformation()
        # Define the convolution blocks as shown in the architecture
        self.conv1 = ConvBlock(32)
        self.conv2 = ConvBlock(64)
        self.conv3 = ConvBlock(128)
        self.conv4 = ConvBlock(256)
        self.conv5 = ConvBlock(512)
        self.conv6 = ConvBlock(256)
        self.conv7 = ConvBlock(32)
        
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.upconv1 = UpConvBlock(128)
        self.upconv2 = UpConvBlock(64)
        self.upconv3 = UpConvBlock(32)
        self.upconv4 = UpConvBlock(32)

        # self.attention1 = AttentionLayer(128)
        # self.attention2 = AttentionLayer(256)
        # self.attention3 = AttentionLayer(512)
        
        # self.final_conv = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='tanh')
        # self.final_conv = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation=None)
        self.final_conv = tf.keras.layers.Conv2D(4, (3, 3), padding='same', activation='linear')

        # # Final layers for producing flow field and brightness map
        # self.flow_conv = layers.Conv2D(2, kernel_size=4, strides=1, padding='same', activation=None)
        # self.brightness_conv = layers.Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')

        self.relu_en0 = layers.ReLU()
        
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
        # landmarks_reshaped_x = tf.reshape(landmarks_reshaped_x, (batch_size, 1, 1, 11))
        landmarks_reshaped_x = tf.reshape(landmarks_reshaped_x, (batch_size, 1, 1, 6))
        landmarks_reshaped_x = tf.tile(landmarks_reshaped_x, [1, height, width, 1])

        # landmarks_reshaped_y = tf.reshape(landmarks_reshaped_y, (batch_size, 1, 1, 11))
        landmarks_reshaped_y = tf.reshape(landmarks_reshaped_y, (batch_size, 1, 1, 6))
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

        # p3 = self.attention1(p3)

        x4 = self.conv4(p3)
        p4 = self.pool4(x4)
        # print('4th polling', p4.shape)

        # p4 = self.attention2(p4)

        x = self.conv5(p4)
        x = self.upconv1(x)
        # print('1st up-conv', x.shape)

        # x = self.attention3(x)

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

        flow_x, brightness_x = tf.split(x, num_or_size_splits=[2, 2], axis=-1)

        # Flow field and brightness map
        # flow_field = self.flow_conv(x)
        flow_x = tf.tanh(flow_x)
        # brightness_map = tf.math.sigmoid(brightness_x, name=None)#self.brightness_conv(brightness_x)
        brightness_map = tf.nn.softmax(brightness_x)

        # Warp the input image using the flow field
        warped_image = self.trans.apply_transformation(flow_x, input_image, 3)
        
        # Adjust brightness using the brightness map
        # output_image = self.adjust_brightness(warped_image, brightness_map)
        output_image = self.apply_lcm(warped_image, brightness_map)
        # output_image = warped_image

        # return output_image
        return flow_x, brightness_map

    # def adjust_brightness(self, warped_image, brightness_map):
    #     brightness_map = tf.image.resize(brightness_map, (tf.shape(warped_image)[1], tf.shape(warped_image)[2]))
    #     adjusted_image = (warped_image) + (1 - warped_image)*(1-brightness_map)*0.75
        
    #     return adjusted_image
    
    def apply_lcm(self, batch_img, light_weight):
        img_wgts, pal_wgts = tf.split(light_weight, [1,1], 3)
        img_wgts = tf.tile(img_wgts, [1,1,1,3])
        pal_wgts = tf.tile(pal_wgts, [1,1,1,3])
        palette = tf.ones(tf.shape(batch_img), dtype = tf.float32)
        ret = tf.add(tf.multiply(batch_img, img_wgts), tf.multiply(palette, pal_wgts))
        return ret

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_en1 = layers.Conv2D(32, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv_en2 = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv_en3 = layers.Conv2D(128, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv_en4 = layers.Conv2D(256, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv_en5 = layers.Conv2D(512, kernel_size=4, strides=2, padding='same', activation='relu')

        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(5, activation='linear')

        # Layers to process landmarks
        self.landmark_dense1 = Dense(128, activation='relu')
        self.landmark_dense2 = Dense(64, activation='relu')

        self.relu = layers.ReLU()

    def call(self, img_input, pose_input, landmarks):
        batch_size = tf.shape(img_input)[0]
        height = tf.shape(img_input)[1]
        width = tf.shape(img_input)[2]

        # Process landmarks
        landmarks_processed = self.landmark_dense1(landmarks)
        landmarks_processed = self.landmark_dense2(landmarks_processed)
        landmarks_processed = Flatten()(landmarks_processed)
        
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

        x = Concatenate()([x, landmarks_processed])

        # Dense layers for regression
        x = self.dense1(x)
        x = self.dense2(x)
        
        # Output layer for gaze prediction
        x = self.dense3(x)

        x_dir, x_reg, x_gan = tf.split(x, num_or_size_splits=[2, 2, 1], axis=-1)

        # x_reg = Lambda(lambda x: x * 20.0)(x_reg)  # Scaling to range [-20, 20]

        x_reg = tf.reshape(x_reg, (batch_size, 1, 2))
        x_dir = tf.reshape(x_dir, (batch_size, 1, 2))

        x_gan = tf.tanh(x_gan)
        x_reg = self.relu(x_reg)
        x_dir = tf.tanh(x_dir)

        return x_gan, x_reg, x_dir


class GazeRedirectGAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(GazeRedirectGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.trans = Transformation()

    def compile(self, gen_optimizer, disc_optimizer, loss_fn):
        super(GazeRedirectGAN, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # (img_t, p_t, gaze_target, landmarks_t), (img, p, gaze_real, landmarks), weightMap, weightMapGaze = data
        (img_t, p_t, gaze_target, landmarks_t, weightMapEyeball_t, mg_t), (img, p, gaze_real, landmarks, weightMapEyeball, mg), weightMap = data

        # # Create the weight map for a single image
        # weight_map = self.create_gaussian_weight_map(img.shape[1:3])
        # # weight_map = self.create_weight_map_from_landmarks_tf(landmarks, sigma=5)
        # # weight_map_t = self.create_weight_map_from_landmarks(landmarks_t, sigma=5)
        # # weight_map = np.maximum(weight_map_r, weight_map_t)

        # # Convert to tensor
        # weight_map = tf.convert_to_tensor(weight_map, dtype=tf.float32)

        # # Expand dimensions to match batch size and channels
        # weight_map = tf.expand_dims(weight_map, axis=-1)  # Shape [height, width, 1]
        # weight_map = tf.expand_dims(weight_map, axis=0)   # Shape [1, height, width, 1]

        # # Tile the weight map across the batch
        # batch_size = tf.shape(img)[0]
        # weight_map = tf.tile(weight_map, [batch_size, 1, 1, tf.shape(img)[-1]])  # Shape [batch_size, height, width, channels]

        with tf.GradientTape(persistent=True) as tape:
            corr_flow, corr_brightness_map = self.generator(img, p, gaze_target, landmarks)
            warped_image = self.trans.apply_transformation(corr_flow, img, 3)
            output_image = self.trans.apply_lcm(warped_image, corr_brightness_map)

            # corr_loss = self.loss_fn(img_t, output_image)
            # corr_loss = corr_loss + self.weighted_mse_loss(img_t, output_image, weightMap)
            # corr_loss = corr_loss + self.weighted_mse_loss(img_t, output_image, weightMapGaze)

            cl_m = self.loss_fn(img_t, output_image) #self.loss_fn(img_t, output_image)
            cl_e = self.weighted_mse_loss(img_t, warped_image, weightMap)
            cl_g = self.weighted_mse_loss(img_t, warped_image, weightMapEyeball) + self.weighted_mse_loss(img_t, warped_image, weightMapEyeball_t)
            cl_es = self.eyeball_structure_loss(weightMapEyeball, corr_flow, img)
            #cl_ner = 0#cl_ner = self.non_eye_region_flow_loss(weightMap, corr_flow, img)
            # cl_gf = self.gaze_flow_consistency_loss(corr_flow, mg, mg_t, weightMapEyeball)

            # corr_loss = cl_m + cl_e + cl_g + cl_es #+ cl_ner
            corr_loss = cl_m + 1.0*cl_e + 4.0*cl_g + 1.0*cl_es #+ 0.8*cl_gf

            recon_flow, recon_brightness_map = self.generator(output_image, p_t, gaze_real, landmarks_t)
            warped_image_r = self.trans.apply_transformation(recon_flow, output_image, 3)
            recon_image = self.trans.apply_lcm(warped_image_r, recon_brightness_map)
            # recon_loss = self.loss_fn(img, recon_image)
            # recon_loss = recon_loss + self.weighted_mse_loss(img, recon_image, weightMap)
            # recon_loss = recon_loss + self.weighted_mse_loss(img, recon_image, weightMapGaze)

            rl_m = self.loss_fn(img, recon_image) #self.loss_fn(img, recon_image)
            rl_e = self.weighted_mse_loss(img, warped_image_r, weightMap)
            rl_g = self.weighted_mse_loss(img, warped_image_r, weightMapEyeball) + self.weighted_mse_loss(img, warped_image_r, weightMapEyeball_t)
            recon_loss = rl_m + 1.0*rl_e + 1.0*rl_g

            gan_real, gaze_real_p, gaze_real_dir = self.discriminator(img, p, landmarks, training=True)
            gan_fake, gaze_fake_p, gaze_fake_dir = self.discriminator(output_image, p_t, landmarks_t, training=True)

            gaze_real_p = gaze_real_p*gaze_real_dir
            gaze_fake_p = gaze_fake_p*gaze_fake_dir

            gan_real = 1.0*tf.reduce_mean(gan_real)
            gan_fake = 1.0*tf.reduce_mean(gan_fake)
            d_gaze_mse = 1.0*self.loss_fn(gaze_real, gaze_real_p)
            d_loss = d_gaze_mse - gan_real + gan_fake

            gaze_mse = self.loss_fn(gaze_target, gaze_fake_p)

            # L_total = 800.0*L_total #+ 2.0*self.loss_fn(gaze_target, gaze_fake_p) + 1.0*(-1 - tf.reduce_mean(gan_fake))

            L_total = 0.8 * (corr_loss) + 0.2 * (recon_loss) + 0.01*gaze_mse + 0.01*(-1 - gan_fake)

        # Compute gradients for the total loss
        gradients = tape.gradient(L_total, self.generator.trainable_variables)

        disc_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)

        # Apply gradients
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        # return {"lcr": 0.8 * (lc) + 0.2 * (lr), "ggr": 2.0*self.loss_fn(gaze_target, gaze_fake_p), "glf": 1.0*(-1 - tf.reduce_mean(gan_fake)), "lt": L_total, "dl": d_loss}
        return {"cl_m": cl_m, "cl_e": cl_e, "cl_g": cl_g, "cl_es": cl_es, "cl_gf": cl_gf, "lt": L_total, "g": gaze_mse, "gf": gan_fake}

    
    # def loss(self, img_ori, img_pred, img_trg, flow, weightMapEyeball, weightMapEyeball_t, weightMap):
    #     l_img = self.l2_loss(img_trg, img_pred)
    #     l_e = self.weighted_mse_loss(img_trg, img_pred, weightMap)
    #     l_g = self.weighted_mse_loss(img_trg, img_pred, weightMapEyeball) + self.weighted_mse_loss(img_trg, img_pred, weightMapEyeball_t)
    #     l_es = self.eyeball_structure_loss(weightMapEyeball, flow, img_ori)

    #     loss = l_img + l_e + l_g + l_es


    def l2_loss(self, y_true, y_pred):
        """
        Compute the L2 loss between the true and predicted values.
        
        Args:
        - y_true (tensor): The ground truth values.
        - y_pred (tensor): The predicted values.

        Returns:
        - L2 loss (tensor): The L2 loss value.
        """
        return tf.reduce_sum(tf.square(y_true - y_pred))
    
    def weighted_mse_loss(self, y_true, y_pred, weight_map):
        # Calculate the Mean Squared Error (MSE) between true and predicted values
        mse = tf.square(y_true - y_pred)
        
        # Apply the weight map
        weighted_mse = mse * weight_map
        
        # Calculate the mean of the weighted MSE
        loss = tf.reduce_mean(weighted_mse)
        # loss = tf.reduce_sum(weighted_mse)
        return loss
    
    # def eyeball_structure_loss(self, weightMap, flow):
    #     """
    #     Calculate the loss to retain the eyeball structure during gaze redirection.
        
    #     Args:
    #     - weightMap (tensor): Binary mask indicating eyeball region, shape (batch_size, height, width, 1)
    #     - flow (tensor): Flow field tensor, shape (batch_size, height, width, 2)
    #     - eyelid_mask (tensor, optional): Binary mask indicating eyelid region, same shape as weightMap. Default is None.

    #     Returns:
    #     - loss (tensor): Calculated loss for retaining eyeball structure.
    #     """
        
    #     # Assuming flow has shape (batch_size, height, width, 2)
    #     flow_x, flow_y = tf.split(flow, num_or_size_splits=2, axis=-1)

    #     # Compute the gradients of the flow field with respect to x and y
    #     flow_x_dx = tf.abs(tf.image.sobel_edges(flow_x)[..., 0])
    #     flow_y_dy = tf.abs(tf.image.sobel_edges(flow_y)[..., 1])
        
    #     # Combine gradients
    #     flow_gradient_magnitude = flow_x_dx + flow_y_dy
        
    #     # Calculate the loss as specified in the equation
    #     loss = weightMap * flow_gradient_magnitude
        
    #     # Sum over all pixels and batch
    #     loss = tf.reduce_sum(loss)
        
    #     return loss

    # def non_eye_region_flow_loss(self, eye_region_weightMap, flow):
    #     """
    #     Loss to enforce that flow is zero outside the eye region.

    #     Args:
    #     - eye_region_weightMap (tensor): Binary mask indicating eye region, shape (batch_size, height, width, 1)
    #     - flow (tensor): Flow field tensor, shape (batch_size, height, width, 2)

    #     Returns:
    #     - loss (tensor): Calculated loss for non-eye regions to enforce zero flow.
    #     """
    #     non_eye_region_mask = 1.0 - eye_region_weightMap  # Mask for regions outside the eye

    #     # Square the flow values
    #     flow_x, flow_y = tf.split(flow, num_or_size_splits=2, axis=-1)
    #     flow_magnitude = tf.square(flow_x) + tf.square(flow_y)

    #     # Apply the non-eye region mask
    #     loss = non_eye_region_mask * flow_magnitude

    #     # Sum over all pixels and batch
    #     loss = tf.reduce_sum(loss)

    #     return loss

    # def gaze_flow_consistency_loss(self, flow, mg, mg_t, weightMapEyeball):
    #     """
    #     Calculate the loss that ensures the flow in the eyeball region is consistent with the expected gaze shift.
        
    #     Args:
    #     - flow (tensor): Flow field tensor, shape (batch_size, height, width, 2)
    #     - mg (tensor): Middle gaze point in the original image, shape (batch_size, 2)
    #     - mg_t (tensor): Middle gaze point in the target image, shape (batch_size, 2)
    #     - weightMapEyeball (tensor): Binary mask indicating the eyeball region, shape (batch_size, height, width, 1)

    #     Returns:
    #     - loss (tensor): Calculated loss for ensuring flow consistency with gaze shift.
    #     """
        
    #     # Calculate the expected flow based on the difference between middle gaze points
    #     expected_flow = tf.expand_dims(mg_t - mg, axis=1)  # Shape: (batch_size, 1, 2)

    #     # Compute the actual flow in the eyeball region by averaging flow within the region
    #     flow_x, flow_y = tf.split(flow, num_or_size_splits=2, axis=-1)
    #     flow_avg_x = tf.reduce_sum(flow_x * weightMapEyeball, axis=[1, 2]) / (tf.reduce_sum(weightMapEyeball, axis=[1, 2]) + 1e-6)
    #     flow_avg_y = tf.reduce_sum(flow_y * weightMapEyeball, axis=[1, 2]) / (tf.reduce_sum(weightMapEyeball, axis=[1, 2]) + 1e-6)
    #     flow_avg = tf.stack([flow_avg_x, flow_avg_y], axis=-1)  # Shape: (batch_size, 2)

    #     # Compute the difference between the expected flow and actual flow
    #     loss = tf.reduce_mean(tf.square(flow_avg - expected_flow))

    #     return loss

    def gaze_flow_consistency_loss(self, flow, mg, mg_t, weightMapEyeball):
        """
        Calculate the loss that ensures the flow in the eyeball region is consistent with the expected gaze shift.
        
        Args:
        - flow (tensor): Flow field tensor, shape (batch_size, height, width, 2)
        - mg (tensor): Middle gaze point in the original image, shape (batch_size, 2)
        - mg_t (tensor): Middle gaze point in the target image, shape (batch_size, 2)
        - weightMapEyeball (tensor): Binary mask indicating the eyeball region, shape (batch_size, height, width, 1)

        Returns:
        - loss (tensor): Calculated loss for ensuring flow consistency with gaze shift.
        """
        # Calculate the expected flow based on the difference between middle gaze points
        expected_flow = mg_t - mg  # Shape: (batch_size, 2)
        
        # Reshape expected flow to match the flow field's shape
        shape = tf.shape(flow)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        expected_flow = tf.reshape(expected_flow, [batch_size, 1, 1, 2])
        expected_flow = tf.tile(expected_flow, [1, height, width, 1])

        # Mask the flow to focus only on the eyeball region
        weightMapEyeball = tf.cast(weightMapEyeball, tf.float32)
        flow_in_eyeball = flow * weightMapEyeball
        
        # Calculate the flow discrepancy
        flow_diff = flow_in_eyeball - expected_flow
        
        # Calculate the loss as the L2 norm of the flow discrepancy
        loss = tf.reduce_mean(tf.square(flow_diff))

        return loss


    def eyeball_structure_loss(self, weightMap, flow, ori_img):
        """
        Calculate the loss to retain the eyeball structure during gaze redirection.
        
        Args:
        - weightMap (tensor): Binary mask indicating eyeball region, shape (batch_size, height, width, 1)
        - flow (tensor): Flow field tensor, shape (batch_size, height, width, 2)
        - ori_img (tensor): Original image before transformation

        Returns:
        - loss (tensor): Calculated loss for retaining eyeball structure.
        """
        # Calculate TV (dFlow(p)/dx  + dFlow(p)/dy)
        TV_flow = self.TVloss(flow)

        # calculate the (1-D(p))
        img_gray = tf.reduce_mean(ori_img, axis=3, keepdims=True)
        ones = tf.ones_like(img_gray)
        bright = ones - img_gray

        # calculate the F_e(p)
        weights = tf.multiply(bright, weightMap)
        TV_eye = tf.multiply(weights, TV_flow)

        # Sum over all pixels and batch
        # loss = tf.reduce_sum(TV_eye)
        loss = tf.reduce_mean(TV_eye)
        
        return loss

    def TVloss(self, flow):
        """
        Compute the Total Variation (TV) loss for the flow field.
        
        Args:
        - flow (tensor): Flow field tensor, shape (batch_size, height, width, 2)

        Returns:
        - TV (tensor): Total variation loss.
        """
        # flow_x, flow_y = tf.split(flow, num_or_size_splits=2, axis=-1)
        # flow_x_dx = tf.abs(flow_x[:, 1:, :] - flow_x[:, :-1, :])
        # flow_y_dy = tf.abs(flow_y[:, :, 1:] - flow_y[:, :, :-1])
        # TV = tf.reduce_sum(flow_x_dx) + tf.reduce_sum(flow_y_dy)


        dinputs_dx = flow[:, :-1, :, :] - flow[:, 1:, :, :]
        dinputs_dy = flow[:, :, :-1, :] - flow[:, :, 1:, :]
        dinputs_dx = tf.pad(dinputs_dx, [[0, 0], [0, 1], [0, 0], [0, 0]], "CONSTANT")
        dinputs_dy = tf.pad(dinputs_dy, [[0, 0], [0, 0], [0, 1], [0, 0]], "CONSTANT")
        
        tot_var = tf.add(tf.abs(dinputs_dx), tf.abs(dinputs_dy))
        TV = tf.reduce_sum(tot_var, axis=3, keepdims=True)

        return TV

    def non_eye_region_flow_loss(self, eye_region_weightMap, flow, ori_img):
        """
        Loss to enforce that flow is zero outside the eye region.

        Args:
        - eye_region_weightMap (tensor): Binary mask indicating eye region, shape (batch_size, height, width, 1)
        - flow (tensor): Flow field tensor, shape (batch_size, height, width, 2)
        - ori_img (tensor): Original image before transformation

        Returns:
        - loss (tensor): Calculated loss for non-eye regions to enforce zero flow.
        """
        non_eye_region_mask = 1.0 - eye_region_weightMap  # Mask for regions outside the eye

        # Square the flow values
        flow_x, flow_y = tf.split(flow, num_or_size_splits=2, axis=-1)
        flow_magnitude = tf.square(flow_x) + tf.square(flow_y)

        # Apply the non-eye region mask
        loss = non_eye_region_mask * flow_magnitude

        # Sum over all pixels and batch
        loss = tf.reduce_sum(loss)

    
    # def create_gaussian_weight_map(self, image_shape, eye_center = (24,32), sigma=10):
    #     """
    #     Creates a Gaussian weight map centered around the eye.
    #     image_shape: (height, width)
    #     eye_center: (x, y) coordinates of the eye center
    #     sigma: Standard deviation of the Gaussian distribution
    #     """
    #     y, x = np.meshgrid(np.arange(image_shape[0]), np.arange(image_shape[1]), indexing='ij')
    #     distance = (x - eye_center[0])**2 + (y - eye_center[1])**2
    #     weight_map = np.exp(-distance / (2 * sigma**2))
        
    #     # Normalize the weight map so that it ranges from 0 to 1
    #     weight_map = weight_map / np.max(weight_map)
        
    #     return weight_map
    
    
    def call(self, inputs, training=False):
        x_real, gaze_target = inputs
        return self.generator(x_real, gaze_target, training=training)
