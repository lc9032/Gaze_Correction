import pickle
import os

import tensorflow as tf

from model import Generator, GazeRedirectGAN, Discriminator
from processingDataset import ProcessingDataset

import matplotlib.pyplot as plt

import cv2

import numpy as np


class Train():
    def __init__(self):
        self.file_path_l = './training_inputs_COL/left_data.pkl'
        self.file_path_r = './training_inputs_COL/right_data.pkl'
        self.epochs = 1

    def run(self):
        process_dataset = ProcessingDataset()

        batch_size = 64
        data_l = process_dataset.load_pickle_data(self.file_path_l)
        data_r = process_dataset.load_pickle_data(self.file_path_r)
        data = {**data_l, **data_r}

        train_dataset = process_dataset.create_dataset(data, batch_size)

        generator = Generator()
        discriminator = Discriminator()

        gan_model = GazeRedirectGAN(generator, discriminator)
        gan_model.compile(
            gen_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9),
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

        gan_model.fit(train_dataset, epochs=self.epochs, callbacks=[CheckpointSaver()])


train = Train()
train.run()
print("Training Completed!!!")