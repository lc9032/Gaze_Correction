import pickle
import os

import tensorflow as tf

from model import Generator, GazeRedirectGAN, Discriminator

import matplotlib.pyplot as plt

import cv2

import numpy as np
file_path_l = './training_inputs_COL/left_data.pkl'
file_path_r = './training_inputs_COL/right_data.pkl'


def load_pickle_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def preprocess_data(img, p, h, v, img_t, p_t, h_t, v_t, landmarks, landmarks_t):
    
    img = tf.cast(img, tf.float32)
    img_t = tf.cast(img_t, tf.float32)
    
    # Normalize the images to [-1, 1]
    img = (img / 127.5) - 1.0
    img_t = (img_t / 127.5) - 1.0

    # h = tf.expand_dims(h, axis=-1)
    # v = tf.expand_dims(v, axis=-1)
    # h_t = tf.expand_dims(h_t, axis=-1)
    # v_t = tf.expand_dims(v_t, axis=-1)
    p = tf.expand_dims(tf.cast(p, tf.float32), axis=-1)
    h = tf.expand_dims(tf.cast(h, tf.float32), axis=-1)
    v = tf.expand_dims(tf.cast(v, tf.float32), axis=-1)
    p_t = tf.expand_dims(tf.cast(p_t, tf.float32), axis=-1)
    h_t = tf.expand_dims(tf.cast(h_t, tf.float32), axis=-1)
    v_t = tf.expand_dims(tf.cast(v_t, tf.float32), axis=-1)

    gaze_real = tf.stack([h, v], axis=-1)
    gaze_target = tf.stack([h_t, v_t], axis=-1)

    landmarks = tf.cast(landmarks, tf.float32)  # Convert eye landmarks to tensor
    landmarks_t = tf.cast(landmarks_t, tf.float32) 

    return (img_t, p_t, gaze_target, landmarks_t), (img, p, gaze_real, landmarks)

def create_dataset(data, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((data['img'], data['p'], data['h'], data['v'], data['img_t'], data['p_t'], data['h_t'], data['v_t'], data['landmarks'], data['landmarks_t']))
    dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# def perceptual_loss_fn(x, y):
#     x_features = vgg_model(x)
#     y_features = vgg_model(y)
#     return tf.reduce_mean(tf.abs(x_features - y_features))


batch_size = 64
data_l = load_pickle_data(file_path_l)
# data_r = load_pickle_data(file_path_r)
# data = {**data_l, **data_r}

train_dataset = create_dataset(data_l, batch_size)


# Example usage
# input_shape = (224, 224, 3)
# params = type('Params', (object,), {'image_size': 224})

generator = Generator()

discriminator = Discriminator()
# vgg_model, _ = vgg_16(tf.keras.Input(shape=input_shape))

# generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9)


gan_model = GazeRedirectGAN(generator, discriminator)#, discriminator, vgg_model)
gan_model.compile(
    gen_optimizer=generator_optimizer,
    disc_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    # loss_fn = vgg_model.perceptual_loss
    loss_fn = tf.keras.losses.MeanSquaredError()
    # loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)#,
    # perceptual_loss_fn=lambda x, y: tf.reduce_mean(tf.abs(vgg_model(x) - vgg_model(y)))
)

# Define checkpoint directory and checkpoint objects
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator=gan_model.generator,
                                 discriminator=gan_model.discriminator,
                                 gen_optimizer=gan_model.gen_optimizer,
                                 disc_optimizer=gan_model.disc_optimizer
                                 )
# checkpoint = tf.train.Checkpoint(gan_model=gan_model, gen_optimizer=gan_model.gen_optimizer, disc_optimizer=gan_model.disc_optimizer)

# checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)

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
# # import matplotlib.pyplot as plt

# def display_generated_image(generator, dataset):
# Extract a random image and corresponding target from the dataset

batch_size = 32
test_data = load_pickle_data('./training_inputs_COL/left_data.pkl')
test_dataset = create_dataset(test_data, batch_size)

for (img_t, p_t, gaze_target, landmarks_t), (img, p, gaze_real, landmarks) in test_dataset.take(1):
    break

img = tf.cast(img, tf.float32)
img_t = tf.cast(img_t, tf.float32)

gaze_target = tf.cast(gaze_target, tf.float32)
gaze_real = tf.cast(gaze_real, tf.float32)


# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# print(img.shape)
# print(gaze_target.shape)
# print(img)
# print(gaze_target)

# Generate the target image using the generator

# print(gaze_target)
generated_image = generator(img, p, gaze_target, landmarks, training=False)

# Convert images to the range [0, 1] for displaying
img = (img + 1.0) / 2.0
img_t = (img_t + 1.0) / 2.0
generated_image = (generated_image + 1.0) / 2.0

# Plot the original, target, and generated images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img[0].numpy())
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(img_t[0].numpy())
axes[1].set_title("Target Image")
axes[1].axis('off')

axes[2].imshow(generated_image[0].numpy())
axes[2].set_title("Generated Image")
axes[2].axis('off')

plt.savefig('./result.png')
plt.close(fig)



# def predict_gaze(model, img, pose):
#     img, pose, _ = preprocess_data(img, pose, 0, 0)  # Only need to preprocess the image and pose
#     img = tf.expand_dims(img, axis=0)  # Add batch dimension
#     pose = tf.expand_dims(pose, axis=0)  # Add batch dimension
#     prediction = model.predict([img, pose])
#     return prediction


# # Load and preprocess your input image
# # input_image = cv2.imread('./preprocessing_dataset_COL/0009/left/0009_2m_0P_-10V_10H.jpg')  # Load your image using OpenCV
# # pose = 0.0 
# prediction = discriminator(img, p, training=False)
# print('Predicted gaze:', prediction)


# print(gaze_real)