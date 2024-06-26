
# import os
# import time
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
import cv2 # type: ignore
# import matplotlib.pyplot as plt # type: ignore

from model import Generator, Discriminator
from facial_landmark import facial_landmark

# import mediapipe as mp # type: ignore

from threading import Thread, Lock
import multiprocessing as mp
from queue import Queue

image_width = 64
image_height = 32

class GazeCorrSys():
    def __init__(self):
        self.checkpoint_dir = './training_checkpoints'
        self.frame_queue = Queue(maxsize=10)
        self.output_queue = Queue(maxsize=10)
        self.lock = Lock()
        self.generator = self.loadCheckPoint()

    # Function to preprocess the input image
    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_width, image_height))
        image = (image / 127.5) - 1
        image = np.expand_dims(image, axis=0)
        return image

    # Function to postprocess the output image
    def postprocess_image(self, image):
        image = (image + 1) * 127.5
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

    # def gazeRedir(self):
    #     pass

    def inference_thread(self):
        while True:
            if not self.frame_queue.empty():
                frame, left_eye_region, right_eye_region, left_eye_landmarks, right_eye_landmarks, pose = self.frame_queue.get()
                input_image_l = self.preprocess_image(left_eye_region)
                input_image_r = self.preprocess_image(right_eye_region)
                input_images = np.concatenate([input_image_l, input_image_r], axis=0)

                gaze_target = np.array([[0.0, 0.0]])  # Adjust as needed
                gaze_target = tf.convert_to_tensor(gaze_target, dtype=tf.float32)
                gaze_target = tf.expand_dims(gaze_target, axis=0)
                gaze_targets = tf.tile(gaze_target, [2, 1, 1])

                landmarks_l = tf.cast(left_eye_landmarks, tf.float32) 
                landmarks_r = tf.cast(right_eye_landmarks, tf.float32)
                landmarks_batch = tf.concat([tf.expand_dims(landmarks_l, axis=0), tf.expand_dims(landmarks_r, axis=0)], axis=0)
                # landmarks_l = tf.expand_dims(landmarks_l, axis=0)
                # landmarks_r = tf.expand_dims(landmarks_r, axis=0)

                predictions = self.generator(input_images, pose, gaze_targets, landmarks_batch, training=False)
                
                output_image_l = self.postprocess_image(predictions[0])
                output_image_r = self.postprocess_image(predictions[1])

                self.output_queue.put((frame, output_image_l, output_image_r))
    
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
                input_images = np.concatenate([input_image_l, input_image_r], axis=0)
                

                img_h, img_w, _ = output_frame.shape
                face_2d, face_3d, _, _ = self.extract_landmarks(face_landmarks[0], img_w, img_h)
                x, y, z, _, _, _, _ = self.calculate_head_pose(face_2d, face_3d, img_w, img_h)

                # Generate the output image using the generator
                pose_values = [y, y]  
                pose = tf.constant(pose_values, shape=(2, 1), dtype=tf.float32)

                gaze_target = np.array([[0.0, 0.0]])  # Adjust as needed
                gaze_target = tf.convert_to_tensor(gaze_target, dtype=tf.float32)
                gaze_target = tf.expand_dims(gaze_target, axis=0)
                gaze_targets = tf.tile(gaze_target, [2, 1, 1])

                landmarks_l = tf.cast(left_eye_landmarks, tf.float32) 
                landmarks_r = tf.cast(right_eye_landmarks, tf.float32)
                landmarks_batch = tf.concat([tf.expand_dims(landmarks_l, axis=0), tf.expand_dims(landmarks_r, axis=0)], axis=0)

                predictions = generator(input_images, pose, gaze_targets, landmarks_batch, training=False)
                
                # Postprocess the generated image
                output_image_l = self.postprocess_image(predictions[0])
                output_image_r = self.postprocess_image(predictions[1])


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


if __name__ == '__main__':
    gaze_corr_sys = GazeCorrSys()
    gaze_corr_sys.run()

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################


# import cv2
# import numpy as np
# import tensorflow as tf
# from threading import Thread, Lock, Condition
# from multiprocessing import Process, Queue
# from model import Generator, Discriminator
# from facial_landmark import facial_landmark
# import signal
# import os
# import time
# import gc

# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# image_width = 64
# image_height = 32

# class GazeCorrSys():
#     def __init__(self):
#         self.checkpoint_dir = './training_checkpoints'
#         self.frame_queue = Queue(maxsize=10)  # Buffer more frames for processing
#         self.output_queue = Queue(maxsize=10)
#         self.lock = Lock()
#         self.condition = Condition(self.lock)
#         self.stop_flag = False

#     # Function to preprocess the input image
#     def preprocess_image(self, image):
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image, (image_width, image_height))
#         image = (image / 127.5) - 1
#         image = np.expand_dims(image, axis=0)
#         return image

#     # Function to postprocess the output image
#     def postprocess_image(self, image):
#         image = (image[0] + 1) * 127.5
#         image = np.clip(image, 0, 255).astype(np.uint8)
#         return image

#     def loadCheckPoint(self):
#         # Define the generator and discriminator (assuming they are implemented as classes)
#         generator = Generator()
#         discriminator = Discriminator()

#         # Define the optimizers (same as used during training)
#         generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
#         discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#         # Load the checkpoint
#         checkpoint = tf.train.Checkpoint(generator=generator,
#                                          discriminator=discriminator,
#                                          gen_optimizer=generator_optimizer,
#                                          disc_optimizer=discriminator_optimizer)

#         # Restore the latest checkpoint
#         checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

#         return generator
    
#     def extract_landmarks(self, face_landmarks, img_w, img_h):
#         face_2d = []
#         face_3d = []
#         nose_2d = None
#         nose_3d = None

#         for idx, lm in enumerate(face_landmarks.landmark):
#             if idx in {33, 263, 1, 61, 291, 199}:
#                 x, y = int(lm.x * img_w), int(lm.y * img_h)
#                 if idx == 1:
#                     nose_2d = (lm.x * img_w, lm.y * img_h)
#                     nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

#                 face_2d.append([x, y])
#                 face_3d.append([x, y, lm.z])

#         return np.array(face_2d, dtype=np.float64), np.array(face_3d, dtype=np.float64), nose_2d, nose_3d

#     def calculate_head_pose(self, face_2d, face_3d, img_w, img_h):
#         focal_length = 1 * img_w
#         cam_matrix = np.array([[focal_length, 0, img_h / 2],
#                                [0, focal_length, img_w / 2],
#                                [0, 0, 1]])
#         distortion_matrix = np.zeros((4, 1), dtype=np.float64)
#         success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
        
#         rmat, _ = cv2.Rodrigues(rotation_vec)
#         angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
#         x = angles[0] * 360
#         y = angles[1] * 360
#         z = angles[2] * 360
        
#         return x, y, z, rotation_vec, translation_vec, cam_matrix, distortion_matrix

#     def draw_annotations(self, image, x, y, z):
#         cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         return image

#     def capture_frames(self):
#         cap = cv2.VideoCapture(0)

#         if not cap.isOpened():
#             print("Error: Could not open video capture.")
#             return

#         while not self.stop_flag:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             with self.condition:
#                 if self.frame_queue.full():
#                     self.frame_queue.get()
#                 self.frame_queue.put(frame)
#                 self.condition.notify_all()

#         cap.release()

#     def process_frame(self, frame, generator):
#         facial_landmark_ex = facial_landmark()
#         # generator = self.loadCheckPoint()

#         output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_landmarks = facial_landmark_ex.face_landmark(frame)

#         if face_landmarks:    
#             left_eye_region, right_eye_region, left_eye_landmarks, right_eye_landmarks = facial_landmark_ex.extract_eye_regions(frame, face_landmarks[0])
        
#             # Preprocess the captured frame
#             input_image_l = self.preprocess_image(left_eye_region)
#             input_image_r = self.preprocess_image(right_eye_region)
            
#             img_h, img_w, img_c = output_frame.shape
#             face_2d, face_3d, nose_2d, nose_3d = self.extract_landmarks(face_landmarks[0], img_w, img_h)
#             x, y, z, rotation_vec, translation_vec, cam_matrix, distortion_matrix = self.calculate_head_pose(face_2d, face_3d, img_w, img_h)

#             # Generate the output image using the generator
#             p = y
#             gaze_target = np.array([[0.0, 0.0]])  # Adjust as needed
#             gaze_target = tf.convert_to_tensor(gaze_target, dtype=tf.float32)
#             gaze_target = tf.expand_dims(gaze_target, axis=0)
#             landmarks_l = tf.cast(left_eye_landmarks, tf.float32) 
#             landmarks_r = tf.cast(right_eye_landmarks, tf.float32)
#             landmarks_l = tf.expand_dims(landmarks_l, axis=0)
#             landmarks_r = tf.expand_dims(landmarks_r, axis=0)
#             prediction_l = generator(input_image_l, p, gaze_target, landmarks_l, training=False)
#             prediction_r = generator(input_image_r, p, gaze_target, landmarks_r, training=False)
            
#             # Postprocess the generated image
#             output_image_l = self.postprocess_image(prediction_l)
#             output_image_r = self.postprocess_image(prediction_r)

#             facial_landmark_ex.replace_eye_regions(output_frame, face_landmarks[0], output_image_l, output_image_r)
#             self.draw_annotations(output_frame, x, y, z)
            
#             return output_frame
#         return frame

#     def worker(self):
#         generator = self.loadCheckPoint()
#         while not self.stop_flag:
#             if not self.frame_queue.empty():
#                 frame = self.frame_queue.get()
#                 output_frame = self.process_frame(frame, generator)
#                 if not self.output_queue.full():
#                     self.output_queue.put(output_frame)
#                 tf.keras.backend.clear_session()
#                 gc.collect()  # Manually trigger garbage collection

#     def display_frames(self):
#         while not self.stop_flag:
#             if not self.output_queue.empty():
#                 output_frame = self.output_queue.get()
#                 cv2.imshow('Input Image (Left) | Generated Image (Right)', cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
                
#                 # Break the loop on 'q' key press
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     self.stop_flag = True
#                     break

#         cv2.destroyAllWindows()

#     def stop(self):
#         self.stop_flag = True
#         while not self.frame_queue.empty():
#             self.frame_queue.get()
#         while not self.output_queue.empty():
#             self.output_queue.get()

#     def run(self):
#         capture_thread = Thread(target=self.capture_frames)

#         # Using multiple processes for processing frames
#         num_workers = 12  # Adjust based on your CPU cores
#         process_threads = [Process(target=self.worker) for _ in range(num_workers)]

#         display_thread = Thread(target=self.display_frames)

#         capture_thread.start()
#         for pt in process_threads:
#             pt.start()
#         display_thread.start()

#         capture_thread.join()
#         for pt in process_threads:
#             pt.join()
#         display_thread.join()

# def signal_handler(sig, frame):
#     print('You pressed Ctrl+C! Exiting gracefully...')
#     gaze_corr_sys.stop()
#     cv2.destroyAllWindows()
#     tf.keras.backend.clear_session()
#     gc.collect()  # Manually trigger garbage collection
#     os._exit(0)

# if __name__ == '__main__':
#     gaze_corr_sys = GazeCorrSys()
#     signal.signal(signal.SIGINT, signal_handler)
#     gaze_corr_sys.run()


