import pickle
import os

import tensorflow as tf

from model import Generator, GazeRedirectGAN, Discriminator

import matplotlib.pyplot as plt

import numpy as np

from processingDataset import ProcessingDataset

file_path_l = './training_inputs_COL/left_data.pkl'
file_path_r = './training_inputs_COL/right_data.pkl'


def test():
    process_dataset = ProcessingDataset()

    # Define the generator and discriminator (assuming they are implemented as classes)
    generator = Generator()
    discriminator = Discriminator()

    # Define the optimizers (same as used during training)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # Load the checkpoint
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(generator=generator,
                                    discriminator=discriminator,
                                    gen_optimizer=generator_optimizer,
                                    disc_optimizer=discriminator_optimizer
                                    )

    # Restore the latest checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Load the test image
    eye_landmarks = process_dataset.read_landmarks_from_txt('./preprocessing_dataset_CelebA/0/info/left/anna-jackson-2.txt')
    # eye_landmarks = process_dataset.read_landmarks_from_txt('./preprocessing_dataset_COL/0024/info/left/0024_2m_0P_0V_-15H.txt')
    landmarks = tf.cast(eye_landmarks, tf.float32)

    landmarks = tf.expand_dims(landmarks, axis=0)

    test_image_path = './preprocessing_dataset_CelebA/0/left/anna-jackson-2.jpg'
    # test_image_path = './preprocessing_dataset_COL/0024/left/0024_2m_0P_0V_-15H.jpg'
    test_image = process_dataset.load_and_preprocess_image(test_image_path)

    test_image = tf.cast(test_image, tf.float32)

    test_image = tf.expand_dims(test_image, axis=0)  # Add batch dimension

    # Example target gaze direction
    gaze_target = np.array([[0.0, 0.0]])  # Adjust as needed
    gaze_target = tf.convert_to_tensor(gaze_target, dtype=tf.float32)

    gaze_target = tf.expand_dims(gaze_target, axis=0)

    pose = 0.0
    pose = tf.convert_to_tensor(pose, dtype=tf.float32)
    pose = tf.expand_dims(pose, axis=0)

    # Generate the output image

    generated_image = generator(test_image, pose, gaze_target, landmarks, training=False)

    test_image = (test_image + 1.0) / 2.0
    generated_image = (generated_image + 1.0) / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
    axes[0].imshow(test_image[0].numpy())
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(generated_image[0].numpy())
    axes[1].set_title("Generated Image")
    axes[1].axis('off')

    plt.savefig('./result_CA.png')
    plt.close(fig)


test()