
import os
import time
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import Generator, Discriminator
from facial_landmark import facial_landmark

image_width = 64
image_height = 32

# gen_model_ex = Generator()
facial_landmark_ex = facial_landmark()

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

# Function to preprocess the input image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_width, image_height))
    image = (image / 127.5) - 1
    image = np.expand_dims(image, axis=0)
    return image

# Function to postprocess the output image
def postprocess_image(image):
    image = (image[0] + 1) * 127.5
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

# Start the video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break


    original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # original_frame = cv2.resize(original_frame, (256, 256))
    
    # Resize the frame to 256x256 for processing
    face_landmarks = facial_landmark_ex.face_landmark(frame)

    if face_landmarks:    
        left_eye_region, right_eye_region, left_eye_landmarks, right_eye_landmarks = facial_landmark_ex.extract_eye_regions(frame, face_landmarks[0])
        # eye_regions_OUT = cv2.resize(eye_regions, (256, 256))
    
        # Preprocess the captured frame
        input_image_l = preprocess_image(left_eye_region)
        input_image_r = preprocess_image(right_eye_region)
        
        # Generate the output image using the generator
        p = 0.0
        gaze_target = np.array([[0.0, 0.0]])  # Adjust as needed
        gaze_target = tf.convert_to_tensor(gaze_target, dtype=tf.float32)
        gaze_target = tf.expand_dims(gaze_target, axis=0)
        landmarks_l = tf.cast(left_eye_landmarks, tf.float32) 
        landmarks_r = tf.cast(right_eye_landmarks, tf.float32)
        landmarks_l = tf.expand_dims(landmarks_l, axis=0)
        landmarks_r = tf.expand_dims(landmarks_r, axis=0)
        prediction_l = generator(input_image_l, p, gaze_target, landmarks_l, training=False)
        prediction_r = generator(input_image_r, p, gaze_target, landmarks_r, training=False)
        
        # Postprocess the generated image
        output_image_l = postprocess_image(prediction_l)
        output_image_r = postprocess_image(prediction_r)
        
        # Concatenate the original frame and output image for display
        # display_image = np.concatenate((eye_regions_OUT, output_image), axis=1)

        #################################
        display_image2 = facial_landmark_ex.replace_eye_regions(original_frame, face_landmarks[0], output_image_l, output_image_r)
        #################################
        
        # Display the result
        cv2.imshow('Input Image (Left) | Generated Image (Right)', cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR))
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()