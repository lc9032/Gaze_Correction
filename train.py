import pickle
import os

import tensorflow as tf

from model import Generator, GazeRedirectGAN, Discriminator
from processingDataset import ProcessingDataset

import matplotlib.pyplot as plt

import cv2

import numpy as np
file_path_l = './training_inputs_COL/left_data.pkl'
file_path_r = './training_inputs_COL/right_data.pkl'



def train():
    process_dataset = ProcessingDataset()

    batch_size = 64
    data_l = process_dataset.load_pickle_data(file_path_l)
    data_r = process_dataset.load_pickle_data(file_path_r)
    data = {**data_l, **data_r}

    train_dataset = process_dataset.create_dataset(data, batch_size)

    generator = Generator()

    discriminator = Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9)

    gan_model = GazeRedirectGAN(generator, discriminator)#, discriminator, vgg_model)
    gan_model.compile(
        gen_optimizer=generator_optimizer,
        disc_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss_fn = tf.keras.losses.MeanSquaredError()
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
    epochs = 1

    gan_model.fit(train_dataset, epochs=epochs, callbacks=[CheckpointSaver()])


    #########################################################################################################################################
    # batch_size = 32
    # test_data = process_dataset.load_pickle_data('./training_inputs_COL/left_data.pkl')
    # test_dataset = process_dataset.create_dataset(test_data, batch_size)

    # for (img_t, p_t, gaze_target, landmarks_t), (img, p, gaze_real, landmarks) in test_dataset.take(1):
    #     break

    # img = tf.cast(img, tf.float32)
    # img_t = tf.cast(img_t, tf.float32)

    # gaze_target = tf.cast(gaze_target, tf.float32)
    # gaze_real = tf.cast(gaze_real, tf.float32)

    # # Generate the target image using the generator

    # # print(gaze_target)
    # generated_image = generator(img, p, gaze_target, landmarks, training=False)

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


train()