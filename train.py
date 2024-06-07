import pickle
import os

import tensorflow as tf

from model import Generator, GazeRedirectGAN

import matplotlib.pyplot as plt


import numpy as np
# file_path = './training_inputs_x/left_data.pkl'
file_path = './training_inputs_x/right_data.pkl'


def load_pickle_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def preprocess_data(img, p, h, v, img_t, p_t, h_t, v_t):
    
    img = tf.cast(img, tf.float32)
    img_t = tf.cast(img_t, tf.float32)
    
    # Normalize the images to [-1, 1]
    img = (img / 127.5) - 1.0
    img_t = (img_t / 127.5) - 1.0

    # h = tf.expand_dims(h, axis=-1)
    # v = tf.expand_dims(v, axis=-1)
    # h_t = tf.expand_dims(h_t, axis=-1)
    # v_t = tf.expand_dims(v_t, axis=-1)
    h = tf.expand_dims(tf.cast(h, tf.float32), axis=-1)
    v = tf.expand_dims(tf.cast(v, tf.float32), axis=-1)
    h_t = tf.expand_dims(tf.cast(h_t, tf.float32), axis=-1)
    v_t = tf.expand_dims(tf.cast(v_t, tf.float32), axis=-1)

    gaze_real = tf.stack([h, v], axis=-1)
    gaze_target = tf.stack([h_t, v_t], axis=-1)

    return (img_t, gaze_target), (img, gaze_real)

def create_dataset(data, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((data['img'], data['p'], data['h'], data['v'], data['img_t'], data['p_t'], data['h_t'], data['v_t']))
    dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def perceptual_loss_fn(x, y):
    x_features = vgg_model(x)
    y_features = vgg_model(y)
    return tf.reduce_mean(tf.abs(x_features - y_features))


batch_size = 32
data = load_pickle_data(file_path)
train_dataset = create_dataset(data, batch_size)


# Example usage
# input_shape = (224, 224, 3)
# params = type('Params', (object,), {'image_size': 224})

generator = Generator()

# discriminator = Discriminator(params)
# vgg_model, _ = vgg_16(tf.keras.Input(shape=input_shape))

# generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9)


gan_model = GazeRedirectGAN(generator)#, discriminator, vgg_model)
gan_model.compile(
    gen_optimizer=generator_optimizer,
    # disc_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn = tf.keras.losses.MeanSquaredError()
    # loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)#,
    # perceptual_loss_fn=lambda x, y: tf.reduce_mean(tf.abs(vgg_model(x) - vgg_model(y)))
)

# Define checkpoint directory and checkpoint objects
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator=gan_model.generator,
                                #  discriminator=gan_model.discriminator,
                                 gen_optimizer=gan_model.gen_optimizer#,
                                #  disc_optimizer=gan_model.disc_optimizer
                                 )
# checkpoint = tf.train.Checkpoint(gan_model=gan_model, gen_optimizer=gan_model.gen_optimizer, disc_optimizer=gan_model.disc_optimizer)

# checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=100)

# Restore the latest checkpoint if it exists
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print(f'Restored from checkpoint: {checkpoint_manager.latest_checkpoint}')
else:
    print('No checkpoint found, starting from scratch.')

# Define a callback to save checkpoints
class CheckpointSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        checkpoint_manager.save()
        print(f'\nCheckpoint saved at epoch {epoch + 1}')

# Train the GAN model with checkpoint saving
epochs = 100

gan_model.fit(train_dataset, epochs=epochs, callbacks=[CheckpointSaver()])


#########################################################################################################################################
# # # import matplotlib.pyplot as plt

# # def display_generated_image(generator, dataset):
# # Extract a random image and corresponding target from the dataset
# for (img_t, gaze_target), (img, gaze_real) in train_dataset.take(1):
#     break

# img = tf.cast(img, tf.float32)
# img_t = tf.cast(img_t, tf.float32)

# gaze_target = tf.cast(gaze_target, tf.float32)
# gaze_real = tf.cast(gaze_real, tf.float32)


# # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# # print(img.shape)
# # print(gaze_target.shape)
# # print(img)
# # print(gaze_target)

# # Generate the target image using the generator

# print(gaze_target)
# generated_image = generator(img, gaze_target, training=False)

# # Convert images to the range [0, 1] for displaying
# img = (img + 1.0) / 2.0
# img_t = (img_t + 1.0) / 2.0
# generated_image = (generated_image + 1.0) / 2.0

# # Plot the original, target, and generated images
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# axes[0].imshow(img[0].numpy())
# axes[0].set_title("Original Image")
# axes[0].axis('off')

# axes[1].imshow(img_t[0].numpy())
# axes[1].set_title("Target Image")
# axes[1].axis('off')

# axes[2].imshow(generated_image[0].numpy())
# axes[2].set_title("Generated Image")
# axes[2].axis('off')

# plt.savefig('./result.png')
# plt.close(fig)

##############################################################################################################


# Function to load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(32, 64)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)

    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

# Function to unnormalize the image
def unnormalize_image(image):
    image = (image + 1) / 2 * 255
    return tf.cast(image, tf.uint8)

# Load the test image

test_image_path = './preprocessing_dataset/0029/right/0029_2m_0P_0V_-15H.jpg'
test_image = load_and_preprocess_image(test_image_path)

test_image = tf.cast(test_image, tf.float32)
# test_image = (test_image / 127.5) - 1.0

test_image = tf.expand_dims(test_image, axis=0)  # Add batch dimension

tar_img_ph = './preprocessing_dataset/0029/right/0029_2m_0P_0V_0H.jpg'
tar_img = load_and_preprocess_image(tar_img_ph)
tar_img = tf.cast(tar_img, tf.float32)



# Example target gaze direction
gaze_target = np.array([[0.0, 0.0]])  # Adjust as needed
gaze_target = tf.convert_to_tensor(gaze_target, dtype=tf.float32)

gaze_target = tf.expand_dims(gaze_target, axis=0)

# Define and compile your generator model as needed
# generator = Generator()
# Load your trained weights if not already done
# checkpoint.restore(checkpoint_manager.latest_checkpoint)

# Generate the output image

# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# print(test_image.shape)
# print(gaze_target.shape)
# print(test_image)
# print(gaze_target)

# print(gaze_target)
generated_image = generator(test_image, gaze_target, training=False)

test_image = (test_image + 1.0) / 2.0
generated_image = (generated_image + 1.0) / 2.0

tar_img = (tar_img + 1.0) / 2.0
# generated_image = tf.squeeze(generated_image, axis=0).numpy()  # Remove batch dimension


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
axes[0].imshow(test_image[0].numpy())
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(tar_img.numpy())
axes[1].set_title("Target Image")
axes[1].axis('off')

axes[2].imshow(generated_image[0].numpy())
axes[2].set_title("Generated Image")
axes[2].axis('off')

plt.savefig('./result.png')
plt.close(fig)
