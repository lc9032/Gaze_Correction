import pickle
import os

import tensorflow as tf

from model import Generator, GazeRedirectGAN, Discriminator

import matplotlib.pyplot as plt

import cv2

import numpy as np


class ProcessingDataset:
    def __init__(self):
        pass

    def load_pickle_data(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def preprocess_data(self, img, p, h, v, img_t, p_t, h_t, v_t, landmarks, landmarks_t):
        
        img = tf.cast(img, tf.float32)
        img_t = tf.cast(img_t, tf.float32)
        
        # Normalize the images to [-1, 1]
        img = (img / 127.5) - 1.0
        img_t = (img_t / 127.5) - 1.0

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

    def create_dataset(self, data, batch_size=32):
        dataset = tf.data.Dataset.from_tensor_slices((data['img'], data['p'], data['h'], data['v'], data['img_t'], data['p_t'], data['h_t'], data['v_t'], data['landmarks'], data['landmarks_t']))
        dataset = dataset.map(self.preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset


    def read_landmarks_from_txt(self, file_path):
        landmarks = []
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(int, line.strip().split(','))
                landmarks.append((x, y))
        return landmarks

    # Function to load and preprocess the image
    def load_and_preprocess_image(self, image_path, target_size=(32, 64)):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, target_size)

        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        return image

    # Function to unnormalize the image
    def unnormalize_image(self, image):
        image = (image + 1) / 2 * 255
        return tf.cast(image, tf.uint8)