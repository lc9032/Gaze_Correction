import os
import logging



import socket
import struct
import pickle



from silence_tensorflow import silence_tensorflow
silence_tensorflow("ERROR")

import tensorflow as tf # type: ignore

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 




import numpy as np # type: ignore
import cv2 # type: ignore
import ffmpeg # type: ignore
# from threading import Thread, Lock
# import multiprocessing as mp
# from queue import Queue

import pyvirtualcam # type: ignore

from Model.model import Generator, Discriminator
from Model.facial_landmark import FacialLandmark

from Model.transformation import Transformation
import time

CAMERA_VIDEO_SWITCH = 0


# input_video_path = './testVideos/SampleFile_720.mp4'
input_video_path = './testVideos/TEST8_480.mp4'
output_video_path = './output_video.mp4'




class GazeCorrSys_server():
    def __init__(self):
        self.cap = None
        self.checkpoint_dir = './TrainingCheckPoints/training_checkpoints_N_0912_2'
        self.generator, self.discriminator = self.loadCheckPoint()
        self.facial_landmark_ex = FacialLandmark()
        self.image_width = 64
        self.image_height = 48
        self.trans = Transformation()

        self.generator.trainable = False
        self.discriminator.trainable = False

    @tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 48, 64, 3], dtype=tf.float32),  # img_input
    tf.TensorSpec(shape=[None, 1], dtype=tf.float32),          # pose_input
    tf.TensorSpec(shape=[None, 1, 2], dtype=tf.float32),       # gaze_target
    tf.TensorSpec(shape=[None, 6, 2], dtype=tf.float32)        # landmarks
    ])
    def generator_inference(self, img_input, pose_input, gaze_target, landmarks):
        # Your inference logic here (from your Generator class)
        return self.generator(img_input, pose_input, gaze_target, landmarks, training=False)

    # Function to preprocess the input image
    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_width, self.image_height))
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1.0
        image = np.expand_dims(image, axis=0)
        return image

    # Function to postprocess the output image
    def postprocess_image(self, image):
        image = (image + 1) * 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

    def draw_annotations(self, image, x, y, z):
        cv2.putText(image, "original video", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # nose_3d_projection, _ = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix)
        # p1 = (int(nose_2d[0]), int(nose_2d[1]))
        # p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
        
        # cv2.line(image, p1, p2, (255, 0, 0), 3)
        # cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(image, "headpose_x: " + str(np.round(x, 2)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "headpose_y: " + str(np.round(y, 2)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(image, "z: " + str(np.round(z, 2)), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        return image

    def draw_gaze_predition(self, image, gaze_p):
        # Extract x and y coordinates from gaze_p
        gaze_x, gaze_y = gaze_p[0][0][0].numpy(), gaze_p[0][0][1].numpy()

        # Convert gaze coordinates to pixel coordinates
        frame_height, frame_width = image.shape[:2]
        pixel_x = int((gaze_x*20) + frame_width / 2)
        pixel_y = int(frame_height - ((gaze_y*20) + frame_height / 2))

        # Draw a small circle at the gaze position
        cv2.circle(image, (pixel_x, pixel_y), 10, (255, 255, 0), -1)

        # cv2.putText(image, "gx: " + str(np.round(int(gaze_x), 2)), (500, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(image, "gy: " + str(np.round(int(gaze_y), 2)), (500, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return image
    
    def draw_output_annotations(self, image, gaze_corr_flag):
        # Extract x and y coordinates from gaze_p
        if (gaze_corr_flag):
            cv2.putText(image, "gaze correction ON", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        else:
            cv2.putText(image, "gaze correction OFF", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

        return image

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


    def process_frame(self, frame):
        # Processing the frame (similar to run method)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_landmarks = self.facial_landmark_ex.face_landmark(frame)

        gaze_corr_flag = False

        if face_landmarks:    
            left_eye_region, right_eye_region, left_eye_landmarks, right_eye_landmarks = self.facial_landmark_ex.extract_eye_regions(frame, face_landmarks[0])

            # left_eye_region_flipped, left_eye_landmarks_flipped = self.flip_eye_image_and_update_landmarks(left_eye_region, left_eye_landmarks)

            input_image_l = self.preprocess_image(left_eye_region)#(left_eye_region)
            input_image_r = self.preprocess_image(right_eye_region)

            # left_eye_landmarks_flipped = self.landmarks_adj(left_eye_landmarks_flipped, left_eye_region)
            left_eye_landmarks = self.landmarks_adj(left_eye_landmarks, left_eye_region)
            right_eye_landmarks = self.landmarks_adj(right_eye_landmarks, right_eye_region)


            input_images = np.concatenate([input_image_l, input_image_r], axis=0)
            img_h, img_w, _ = output_frame.shape
            face_2d, face_3d, _, _ = self.extract_landmarks(face_landmarks[0], img_w, img_h)
            x, y, z, _, _, _, _ = self.calculate_head_pose(face_2d, face_3d, img_w, img_h)

            if x < 20 and x > -20 and y < 30 and y > -30:
                gaze_corr_flag = True
                pose_values = [y*3, y*3]  
                pose = tf.constant(pose_values, shape=(2, 1), dtype=tf.float32)
                gaze_target = np.array([[0.0, 0.0]])  
                gaze_target = tf.convert_to_tensor(gaze_target, dtype=tf.float32)
                gaze_target = tf.expand_dims(gaze_target, axis=0)
                gaze_targets = tf.tile(gaze_target, [2, 1, 1])
                landmarks_l = tf.cast(left_eye_landmarks, tf.float32)#tf.cast(left_eye_landmarks, tf.float32) 
                landmarks_r = tf.cast(right_eye_landmarks, tf.float32)
                landmarks_batch = tf.concat([tf.expand_dims(landmarks_l, axis=0), tf.expand_dims(landmarks_r, axis=0)], axis=0)
                corr_flow, corr_brightness_map = self.generator(input_images, pose, gaze_targets, landmarks_batch, training=False)

                
                # corr_flow, corr_brightness_map = self.generator_inference(input_images, pose, gaze_targets, landmarks_batch)

                warped_images = self.trans.apply_transformation(corr_flow, input_images)
                predictions = self.trans.apply_lcm(warped_images, corr_brightness_map)

                output_image_l = self.postprocess_image(predictions[0])
                output_image_r = self.postprocess_image(predictions[1])
                self.facial_landmark_ex.replace_eye_regions(output_frame, face_landmarks[0], output_image_l, output_image_r)

                ##############################################################################################
                # _, gaze_d = self.discriminator(input_images, pose, landmarks_batch, training=False)
                # gaze_direction = gaze_d
                # gaze_direction = gaze_direction.numpy().reshape(-1, 2)

                # # Calculate the eye center for the left and right eyes separately
                # left_eye_points, right_eye_points = self.facial_landmark_ex.extract_eye_points(output_frame, face_landmarks[0])
                # eye_center_l = left_eye_points[0]#np.mean(left_eye_points.numpy(), axis=0).astype(int)
                # eye_center_r = right_eye_points[0]#np.mean(right_eye_points.numpy(), axis=0).astype(int)

                # # Calculate the end points for the gaze direction
                # end_point_l = (eye_center_l[0] + int(gaze_direction[0, 0] * 50),
                #             eye_center_l[1] - int(gaze_direction[0, 1] * 50))
                # end_point_r = (eye_center_r[0] + int(gaze_direction[1, 0] * 50),
                #             eye_center_r[1] - int(gaze_direction[1, 1] * 50))

                # # Draw the gaze direction on the frame
                # cv2.arrowedLine(output_frame, tuple(eye_center_l), end_point_l, (0, 255, 0), 2)
                # cv2.arrowedLine(output_frame, tuple(eye_center_r), end_point_r, (0, 255, 0), 2)

                # gaze_text_l = f"Gaze L: ({gaze_direction[0, 0]:.2f}, {gaze_direction[0, 1]:.2f})"
                # gaze_text_r = f"Gaze R: ({gaze_direction[1, 0]:.2f}, {gaze_direction[1, 1]:.2f})"

                # cv2.putText(output_frame, gaze_text_l, (eye_center_l[0] + 10, eye_center_l[1] - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                # cv2.putText(output_frame, gaze_text_r, (eye_center_r[0] + 10, eye_center_r[1] - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                ##############################################################################################
                #pose_text = f"HeadPose: ({y:.2f}"

                             
            # self.draw_annotations(output_frame, x, y, z)
            
        self.draw_output_annotations(output_frame, gaze_corr_flag)
        # combined_frame = np.hstack((output_frame, frame_rgb))
        # output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        return output_frame
    
    def run(self):
        # Start the video capture
        # self.cap = cv2.VideoCapture(1)
        self.cap = cv2.VideoCapture(1,cv2.CAP_V4L2)
        

        if not self.cap.isOpened():
            print("Error: Could not open video capture.")
            exit()

        width = 640#1280
        height = 360#720
        #self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ######################
        prev_frame_time = 0
        new_frame_time = 0
        ######################

        try:
            # with pyvirtualcam.Camera(width=width, height=height, fps=30, device='/dev/video10') as cam:
            with pyvirtualcam.Camera(width=width, height=height, fps=30) as cam:
                # frame_count = 0
                while True:
                    ret, frame = self.cap.read()

                    

                    if not ret:
                        break

                    # frame_count += 1
                    # if frame_count % 2 != 0:
                    #     continue


  
                    frame = cv2.flip(frame, 1)

                    start_time = time.time()


                    processed_frame = self.process_frame(frame)




                    end_time = time.time()
                    latency = end_time - start_time
                    # cv2.putText(processed_frame, f"Latency: {latency*1000:.2f} ms", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)



                    new_frame_time = time.time() 
                    fps = 1/(new_frame_time-prev_frame_time)
                    prev_frame_time = new_frame_time 

                    # Display FPS on the frame (optional)
                    # cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    cam.send(processed_frame)
                    cam.sleep_until_next_frame()

                    

                    # cv2.imshow('Input Image (Left) | Generated Image (Right)', cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                
                    # # Break the loop on 'q' key press
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     self.cap.release()
                    #     cv2.destroyAllWindows()
                    #     break

        finally:
            # Any cleanup code here
            print("Cleaned up resources")
            # Release the video capture
            if self.cap:
                self.cap.release()

            # Close all OpenCV windows
            cv2.destroyAllWindows()



    
    def run_mp4(self):
        # Open the input video file
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print("Error: Could not open input video file.")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Create VideoWriter object to save the output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        total_latency = 0
        frame_count = 0

        while True:
            ret, frame = cap.read()

            start_time = time.time()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            processed_frame = self.process_frame(frame)


            # Capture the end time and calculate latency
            end_time = time.time()
            latency = end_time - start_time
            total_latency += latency
            frame_count += 1
            cv2.putText(processed_frame, f"Latency: {latency*1000:.2f} ms", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            if frame_count % 30 == 0:
                average_latency = (total_latency / frame_count) * 1000  # Convert to milliseconds
                print(f"Average Latency: {average_latency:.2f} ms")
            

            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

            out.write(processed_frame_rgb)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # # Extract audio from input video
        # audio_output_path = 'audio.aac'
        # ffmpeg.input(input_video_path).output(audio_output_path).run(overwrite_output=True)

        # input_video = ffmpeg.input(output_video_path)
        # audio = ffmpeg.input(audio_output_path)
        # ffmpeg.output(input_video, audio, "./owa.mp4", vcodec='copy', acodec='aac').run(overwrite_output=True)

    
    def run_server(self, host='0.0.0.0', port=9999):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"Listening on {host}:{port}")

        while True:
            client_socket, addr = server_socket.accept()
            print('Connection from:', addr)
            data = b""
            payload_size = struct.calcsize("Q")

            while True:
                while len(data) < payload_size:
                    packet = client_socket.recv(4 * 1024)  # 4K
                    if not packet: break
                    data += packet
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]

                while len(data) < msg_size:
                    data += client_socket.recv(4 * 1024)
                frame_data = data[:msg_size]
                data = data[msg_size:]

                frame = pickle.loads(frame_data)
                processed_frame = self.process_frame(frame)
                a = pickle.dumps(processed_frame)
                message = struct.pack("Q", len(a)) + a
                client_socket.sendall(message)


if __name__ == '__main__':
    gaze_corr_sys = GazeCorrSys_server()
    gaze_corr_sys.run()