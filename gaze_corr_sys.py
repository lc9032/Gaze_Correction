
import os
import time
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import Generator, Discriminator
from facial_landmark import facial_landmark

import mediapipe as mp

image_width = 64
image_height = 32

class GazeCorrSys():
    def __init__(self):
        self.checkpoint_dir = './training_checkpoints'

    # Function to preprocess the input image
    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_width, image_height))
        image = (image / 127.5) - 1
        image = np.expand_dims(image, axis=0)
        return image

    # Function to postprocess the output image
    def postprocess_image(self, image):
        image = (image[0] + 1) * 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def loadCheckPoint(self):
        # Define the generator and discriminator (assuming they are implemented as classes)
        generator = Generator()
        discriminator = Discriminator()

        # Define the optimizers (same as used during training)
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        # Load the checkpoint
        # checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        checkpoint = tf.train.Checkpoint(generator=generator,
                                        discriminator=discriminator,
                                        gen_optimizer=generator_optimizer,
                                        disc_optimizer=discriminator_optimizer
                                        )

        # Restore the latest checkpoint
        checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

        return generator
    
    def extract_landmarks(self, face_landmarks, img_w, img_h):
        face_2d = []
        face_3d = []
        nose_2d = None
        nose_3d = None

        for idx, lm in enumerate(face_landmarks.landmark):
        # for idx, lm in enumerate(face_landmarks):
            if idx in {33, 263, 1, 61, 291, 199}:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

        return np.array(face_2d, dtype=np.float64), np.array(face_3d, dtype=np.float64), nose_2d, nose_3d

    def calculate_head_pose(self, face_2d, face_3d, img_w, img_h):
        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1]])
        distortion_matrix = np.zeros((4, 1), dtype=np.float64)
        success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
        
        rmat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360
        
        return x, y, z, rotation_vec, translation_vec, cam_matrix, distortion_matrix

    def draw_annotations(self, image, x, y, z):
        # nose_3d_projection, _ = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix)
        # p1 = (int(nose_2d[0]), int(nose_2d[1]))
        # p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
        
        # cv2.line(image, p1, p2, (255, 0, 0), 3)
        # cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        return image

    def run(self):
        # gen_model_ex = Generator()
        facial_landmark_ex = facial_landmark()

        generator = self.loadCheckPoint()

        # Start the video capture
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open video capture.")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_landmarks = facial_landmark_ex.face_landmark(frame)

            if face_landmarks:    
                left_eye_region, right_eye_region, left_eye_landmarks, right_eye_landmarks = facial_landmark_ex.extract_eye_regions(frame, face_landmarks[0])
            
                # Preprocess the captured frame
                input_image_l = self.preprocess_image(left_eye_region)
                input_image_r = self.preprocess_image(right_eye_region)
                

                img_h, img_w, img_c = output_frame.shape
                face_2d, face_3d, nose_2d, nose_3d = self.extract_landmarks(face_landmarks[0], img_w, img_h)
                x, y, z, rotation_vec, translation_vec, cam_matrix, distortion_matrix = self.calculate_head_pose(face_2d, face_3d, img_w, img_h)

                # Generate the output image using the generator
                p = y
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
                output_image_l = self.postprocess_image(prediction_l)
                output_image_r = self.postprocess_image(prediction_r)

                facial_landmark_ex.replace_eye_regions(output_frame, face_landmarks[0], output_image_l, output_image_r)
                self.draw_annotations(output_frame, x ,y ,z)
                
            # Display the result
            cv2.imshow('Input Image (Left) | Generated Image (Right)', cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
            
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and close windows
        cap.release()
        cv2.destroyAllWindows()


gaze_corr_sys = GazeCorrSys()
gaze_corr_sys.run()