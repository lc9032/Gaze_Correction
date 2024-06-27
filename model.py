
import tensorflow as tf # type: ignore
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.layers import  Concatenate, Flatten, Dense, Lambda# type: ignore
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2D, ReLU, Conv2DTranspose, Activation, LayerNormalization, Add, BatchNormalization# type: ignore
# from tensorflow.keras.models import Model # type: ignore

from transformation import Transformation


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
        
        # self.final_conv = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='tanh')
        # self.final_conv = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation=None)
        self.final_conv = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='linear')

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
        flow_x = tf.tanh(flow_x)
        brightness_map = tf.math.sigmoid(brightness_x, name=None)#self.brightness_conv(brightness_x)

        # Warp the input image using the flow field
        warped_image = self.trans.apply_transformation(flow_x, input_image, 3)
        
        # Adjust brightness using the brightness map
        output_image = self.adjust_brightness(warped_image, brightness_map)

        return output_image

    def adjust_brightness(self, warped_image, brightness_map):
        brightness_map = tf.image.resize(brightness_map, (tf.shape(warped_image)[1], tf.shape(warped_image)[2]))
        adjusted_image = warped_image * (brightness_map*1.0+0.0)
        return adjusted_image

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

        x_reg = tf.reshape(x_reg, (batch_size, 1, 2))

        x_gan = tf.tanh(x_gan)

        return x_gan, x_reg


class GazeRedirectGAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(GazeRedirectGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, gen_optimizer, disc_optimizer, loss_fn):
        super(GazeRedirectGAN, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        (img_t, p_t, gaze_target, landmarks_t), (img, p, gaze_real, landmarks) = data
        with tf.GradientTape(persistent=True) as tape:
            output_image = self.generator(img, p, gaze_target, landmarks)
            corr_loss = self.loss_fn(img_t, output_image)

            recon_image = self.generator(output_image, p_t, gaze_real, landmarks_t)
            recon_loss = self.loss_fn(img, recon_image)

            lc = corr_loss
            lr = recon_loss
            L_total = 0.8 * (lc) + 0.2 * (lr)

            gan_real, gaze_real_p = self.discriminator(img, p, training=True)
            gan_fake, gaze_fake_p = self.discriminator(output_image, p_t, training=True)

            d_loss = 2.0*self.loss_fn(gaze_real, gaze_real_p) - 0.5*tf.reduce_mean(gan_real) + 0.5*tf.reduce_mean(gan_fake)
            L_total = 400.0*L_total + 0.2*self.loss_fn(gaze_target, gaze_fake_p) - 0.8*tf.reduce_mean(gan_fake)

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
