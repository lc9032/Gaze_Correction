import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from Model.model import Generator, Discriminator
from Model.facial_landmark import FacialLandmark
from Model.transformation import Transformation

class ImageProcessor:
    def __init__(self):
        self.checkpoint_dir = './TrainingCheckPoints/training_checkpoints_N_0906_SP'
        self.generator, self.discriminator = self.loadCheckPoint()
        self.facial_landmark_ex = FacialLandmark()
        self.image_width = 64
        self.image_height = 48
        self.trans = Transformation()
        pass


    def preprocess_image(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_width, self.image_height))

        cv2.imwrite('left_eye_region_flipped.jpg', image)


        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1.0
        image = np.expand_dims(image, axis=0)
        return image

    # Function to postprocess the output image
    def postprocess_image(self, image):
        image = (image + 1) * 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def landmarks_adj(self, landmarks, eye_region):
        """
        Adjusts the landmark positions based on the resized image dimensions (64x48).
        :param landmarks: List of (x, y) tuples representing landmarks.
        :param eye_region: The original eye region image from which the landmarks were extracted.
        :return: A list of adjusted landmarks for the resized image.
        """
        # Get the original dimensions of the eye region
        original_height, original_width = eye_region.shape[:2]

        # Calculate scaling factors
        scale_x = self.image_width / original_width
        scale_y = self.image_height / original_height

        # Adjust landmarks based on the scaling factors
        adjusted_landmarks = []
        for (x, y) in landmarks:
            new_x = int(x * scale_x)
            new_y = int(y * scale_y)
            adjusted_landmarks.append((new_x, new_y))

        return adjusted_landmarks

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

        return generator, discriminator
    

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
        y = angles[1] * 360 * 2
        z = angles[2] * 360
        
        return x, y, z, rotation_vec, translation_vec, cam_matrix, distortion_matrix

    def flip_eye_image_and_update_landmarks(self, eye_region, eye_landmarks, img_w = 64):
        """
        Flip the right eye image horizontally and update its landmarks.
        """
        # Flip right eye image horizontally
        eye_region_flipped = cv2.flip(eye_region, 1)

        # Update right eye landmarks after flipping
        eye_landmarks_flipped = eye_landmarks.copy()

        #print(eye_landmarks_flipped)

        eye_landmarks_np = np.array(eye_landmarks)
        eye_landmarks_np[:, 0] = img_w - eye_landmarks_np[:, 0]
        eye_landmarks_flipped = [tuple(point) for point in eye_landmarks_np]
        reordered_landmarks = [
            eye_landmarks_flipped[3],  # Switch index 0 with 3
            eye_landmarks_flipped[2],  # Switch index 1 with 2
            eye_landmarks_flipped[1],  # Switch index 2 with 1
            eye_landmarks_flipped[0],  # Switch index 3 with 0
            eye_landmarks_flipped[5],  # Switch index 4 with 5
            eye_landmarks_flipped[4],  # Switch index 5 with 4
        ]

        #print(reordered_landmarks)

        return eye_region_flipped, reordered_landmarks

    def process_image(self, image_path, output_image_path):
        # Read the image
        frame = cv2.imread(image_path)

        # Ensure the image was successfully loaded
        if frame is None:
            print(f"Error: Could not load image at {image_path}")
            return
        
        frame = cv2.resize(frame, (1280, 720))
        # frame = cv2.resize(frame, (1920, 1080))

        # Process the frame (same logic as the original frame processing)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_landmarks = self.facial_landmark_ex.face_landmark(frame)

        gaze_corr_flag = False

        if face_landmarks:    
            left_eye_region, right_eye_region, left_eye_landmarks, right_eye_landmarks = self.facial_landmark_ex.extract_eye_regions(frame, face_landmarks[0])

            left_eye_landmarks = self.landmarks_adj(left_eye_landmarks, left_eye_region)
            right_eye_landmarks = self.landmarks_adj(right_eye_landmarks, right_eye_region)

            # left_eye_region_flipped, left_eye_landmarks_flipped = self.flip_eye_image_and_update_landmarks(left_eye_region, left_eye_landmarks)


            input_image_r = self.preprocess_image(right_eye_region)
            input_image_l = self.preprocess_image(left_eye_region)


            # print(left_eye_landmarks_flipped)
            # print(right_eye_landmarks)

            input_images = np.concatenate([input_image_l, input_image_r], axis=0)
            img_h, img_w, _ = output_frame.shape
            face_2d, face_3d, _, _ = self.extract_landmarks(face_landmarks[0], img_w, img_h)
            x, y, z, _, _, _, _ = self.calculate_head_pose(face_2d, face_3d, img_w, img_h)

            if x < 20 and x > -20 and y < 30 and y > -30:
                gaze_corr_flag = True
                pose_values = [-y * 3, y * 3]
                pose = tf.constant(pose_values, shape=(2, 1), dtype=tf.float32)
                gaze_target = np.array([[0.0, 0.0]])
                gaze_target = tf.convert_to_tensor(gaze_target, dtype=tf.float32)
                gaze_target = tf.expand_dims(gaze_target, axis=0)
                gaze_targets = tf.tile(gaze_target, [2, 1, 1])
                landmarks_l = tf.cast(left_eye_landmarks, tf.float32)
                landmarks_r = tf.cast(right_eye_landmarks, tf.float32)
                landmarks_batch = tf.concat([tf.expand_dims(landmarks_l, axis=0), tf.expand_dims(landmarks_r, axis=0)], axis=0)
                corr_flow, corr_brightness_map = self.generator(input_images, pose, gaze_targets, landmarks_batch, training=False)
                warped_images = self.trans.apply_transformation(corr_flow, input_images)
                predictions = self.trans.apply_lcm(warped_images, corr_brightness_map)

                output_image_l = self.postprocess_image(predictions[0])
                output_image_r = self.postprocess_image(predictions[1])
                self.facial_landmark_ex.replace_eye_regions(output_frame, face_landmarks[0], output_image_l, output_image_r)
                # self.facial_landmark_ex.replace_eye_regions(output_frame, face_landmarks[0], cv2.flip(output_image_l, 1), output_image_r)

                pose_text = f"HeadPose: ({y:.2f}"


        # # Convert back to BGR for saving
        output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        # Save the processed image to the specified output path
        cv2.imwrite(output_image_path, output_frame_bgr)

        # print(f"Processed image saved at {output_image_path}")

        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # # Display the original and processed images side by side using matplotlib
        # plt.figure(figsize=(10, 5))

        # # Display the original image
        # plt.subplot(1, 2, 1)
        # plt.imshow(frame_rgb)
        # plt.title('Original Image', fontsize=20)
        # plt.axis('off')

        # # Display the processed image
        # plt.subplot(1, 2, 2)
        # plt.imshow(output_frame)
        # plt.title('Processed Image', fontsize=20)
        # plt.axis('off')

        # # Show the images side by side
        # plt.tight_layout()
        # plt.savefig('./OUT_COM36.png')

        # print(f"Processed image saved at {output_image_path}")

        return output_frame_bgr

# Example usage
image_processor = ImageProcessor()
image_path = 'Screenshot from 2024-09-12 14-38-57.png'#'/media/lc/ADATA SX8200PNP/FH_WEDEL/thesis/DATA_SETS/C_DataSet/columbia_gaze_data_set/Columbia Gaze Data Set/0036/0036_2m_0P_10V_15H.jpg'  # Path to the input image
output_image_path = 'output_image.jpg'  # Path to save the processed image
image_processor.process_image(image_path, output_image_path)
