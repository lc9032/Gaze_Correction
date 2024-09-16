import os
import pickle
import tensorflow as tf # type: ignore
import glob
import numpy as np # type: ignore
import cv2 # type: ignore
import json

from Model.facial_landmark import FacialLandmark


import matplotlib.pyplot as plt

from collections import defaultdict


class ProcessingDataset:
    def __init__(self):
        self.image_width = 64
        self.image_height = 48
        
        self.base_dataset_folder = r"../DATA_SETS/C_DataSet/columbia_gaze_data_set/Columbia Gaze Data Set"

        # self.preprocessing_dataset_dir = './DataSets/preprocessing_dataset_COL_0814'
        # self.preprocessing_dataset_dir = './DataSets/preprocessing_dataset_U2_0817'
        self.preprocessing_dataset_dir = './DataSets/preprocessing_dataset_DIRL_0824'

        # self.save_pickle_path = './DataSets/training_inputs_COL_0817'
        self.save_pickle_path = './DataSets/training_inputs_DIRL_0908'
        # self.ignore_list = ['0008', '0010', '0011', '0016', '0024', '0025', '0043', '0053']
        # self.ignore_list = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0010',
        #                     '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020',
        #                     '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029', '0030',
        #                     '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0039', '0040',
        #                     '0041', '0042', '0043', '0044', '0045', '0046', '0047', '0048', '0049', '0050',
        #                     '0051', '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '0060']
        self.ignore_list = []

    def preprocess_image(self, input_path, output_path):
        facial_landmark = FacialLandmark()

        image = cv2.imread(input_path)
        if image is None:
            print(f"Failed to read image {input_path}")
            return

        face_landmarks = facial_landmark.face_landmark(image)

        if face_landmarks:    
            # eye_regions = self.extract_eye_regions_combined(image, face_landmarks[0])
            eye_regions_left, eye_regions_right, left_eye_landmarks, right_eye_landmarks= facial_landmark.extract_eye_regions(image, face_landmarks[0])

            # eye_regions = cv2.resize(eye_regions, (256, 256)) / 255.0

            # Scale the landmark points to match the resized dimensions
            left_eye_landmarks_resized = [(int(pt[0] * self.image_width / eye_regions_left.shape[1]), int(pt[1] * self.image_height / eye_regions_left.shape[0])) for pt in left_eye_landmarks]
            right_eye_landmarks_resized = [(int(pt[0] * self.image_width / eye_regions_right.shape[1]), int(pt[1] * self.image_height / eye_regions_right.shape[0])) for pt in right_eye_landmarks]

            eye_regions_left = cv2.resize(eye_regions_left, (self.image_width, self.image_height)) / 255.0
            eye_regions_right = cv2.resize(eye_regions_right, (self.image_width, self.image_height)) / 255.0


            # left_most = left_eye_landmarks[0]
            # right_most = left_eye_landmarks[3]
            # Ec = (
            #     int((left_most[0] + right_most[0]) / 2),  # Average of x-coordinates
            #     int((left_most[1] + right_most[1]) / 2)   # Average of y-coordinates
            # )

            # cv2.line(eye_regions_left, left_most, right_most, (0, 255, 0), thickness=5)
            # cv2.circle(eye_regions_left, Ec, 8, (0, 0, 255), -1)

            # for i, point in enumerate(left_eye_landmarks):
            #     if i == 0 or i == 3:
            #         cv2.circle(eye_regions_left, point, 8, (255, 0, 0), -1)
            #     else:
            #         cv2.circle(eye_regions_left, point, 8, (255, 255, 0), -1)

            # for i, point in enumerate(right_eye_landmarks):
            #     if i == 0 or i == 3:
            #         cv2.circle(eye_regions_right, point, 8, (255, 0, 0), -1)
            #     else:
            #         cv2.circle(eye_regions_right, point, 8, (255, 255, 0), -1)


            # cv2.imshow("Image(R)", eye_regions_right)
            # cv2.imshow("Image(L)", eye_regions_left)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

 
            # Save the processed image
            # output_file = os.path.join(output_path, os.path.basename(input_path))
            # cv2.imwrite(output_file, eye_regions * 255.0)

            output_path_left = os.path.join(output_path, 'left')
            output_path_right = os.path.join(output_path, 'right')
            os.makedirs(output_path_left, exist_ok=True)
            os.makedirs(output_path_right, exist_ok=True)

            output_file_left = os.path.join(output_path_left, os.path.basename(input_path))
            output_file_right = os.path.join(output_path_right, os.path.basename(input_path))
            cv2.imwrite(output_file_left, eye_regions_left * 255.0)
            cv2.imwrite(output_file_right, eye_regions_right * 255.0)


            # dir_name = os.path.dirname(input_path)
            file_name_without_extension = os.path.splitext(os.path.basename(input_path))[0]
            new_file_name = file_name_without_extension + '.txt'

            os.makedirs(output_path + '/info/left/', exist_ok=True)
            os.makedirs(output_path + '/info/right/', exist_ok=True)

            output_path_info = os.path.join(output_path, 'info/left', new_file_name)
            # Save the landmarks to a text file
            with open(output_path_info, "w") as f:
                # f.write("Left Eye Landmarks:\n")
                for point in left_eye_landmarks_resized:
                    f.write(f"{point[0]},{point[1]}\n")

            output_path_info = os.path.join(output_path, 'info/right', new_file_name)
            with open(output_path_info, "w") as f:
                # f.write("\nRight Eye Landmarks:\n")
                for point in right_eye_landmarks_resized:
                    f.write(f"{point[0]},{point[1]}\n")
    
    def process_images_in_folder(self, input_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # if filename.endswith("0044_2m_0P_0V_0H.jpg"):
                input_path = os.path.join(input_folder, filename)
                self.preprocess_image(input_path, output_folder)
    
    def preprocess_dataset(self):
        for i in range(1, 57):
        # for i in range(44, 45):
            folder_number = f"{i:04d}"
            input_folder = os.path.join(self.base_dataset_folder, folder_number)
            output_folder = os.path.join(self.preprocessing_dataset_dir, folder_number)

            self.process_images_in_folder(input_folder, output_folder)
    
    def process_image(self, img_path):
        # Load the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_width, self.image_height))  # Resize to the required dimensions
        
        # Extract additional information from the filename
        # basename = os.path.basename(img_path)
        # parts = basename.split('_')
        # person_id = parts[0]
        # distance = parts[1][:-1]  # Remove the 'm'
        # head_pose_y = int(parts[2][:-1])  # Remove the 'P'
        # gaze_v = int(parts[3][:-1])  # Remove the 'V'
        # gaze_h = int(parts[4][:-5])  # Remove the 'H' and '.jpg'

        basename = os.path.basename(img_path)
        parts = basename.split('_')
        person_id = parts[0]
        distance = parts[1][:-1]  # Remove the 'm'
        head_pose_x = int(parts[2][:-2])  # Remove the 'P'
        head_pose_y = int(parts[3][:-2])  # Remove the 'P'
        head_pose_z = int(parts[4][:-2])  # Remove the 'P'
        gaze_v = int(parts[5][:-1])  # Remove the 'V'
        gaze_h = int(parts[6][:-5])  # Remove the 'H' and '.jpg'
        
        return img, person_id, distance, head_pose_y, gaze_v, gaze_h

    def read_landmarks_from_txt(self, file_path):
        landmarks = []
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(int, line.strip().split(','))
                landmarks.append((x, y))
        return landmarks
    
    def read_landmarks_from_json(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            
            # Extract landmarks from the JSON data
            landmarks = [(point['x'], point['y']) for point in data['landmarks']]
            
            # Extract middle_gaze_point if needed
            middle_gaze_point = (data['middle_gaze_point']['x'], data['middle_gaze_point']['y'])
            
        return landmarks, middle_gaze_point
    
    def create_gaussian_weight_map(self, eye_center, image_shape, sigma=5):
        """
        Creates a Gaussian weight map centered around the eye.
        image_shape: (height, width)
        eye_center: (x, y) coordinates of the eye center
        sigma: Standard deviation of the Gaussian distribution
        """
        y, x = np.meshgrid(np.arange(image_shape[0]), np.arange(image_shape[1]), indexing='ij')
        distance = (x - eye_center[0])**2 + (y - eye_center[1])**2
        weight_map = np.exp(-distance / (2 * sigma**2))
        
        # Normalize the weight map so that it ranges from 0 to 1
        weight_map = weight_map / np.max(weight_map)
        
        return weight_map
    
    def create_weight_map_from_landmarks(self, landmarks, image_shape, sigma=8):
        """
        Creates a weight map based on facial landmarks.
        image_shape: (height, width)
        landmarks: List of (x, y) coordinates of the eye region landmarks
        sigma: Standard deviation for the Gaussian smoothing
        
        Returns:
        weight_map: A weight map with values between 0 and 1.
        """
        height, width = image_shape
        weight_map = np.zeros((height, width), dtype=np.float32)

        # Create a binary mask where the eye region is set to 1
        eye_mask = np.zeros_like(weight_map)
        eye_contour = np.array(landmarks, dtype=np.int32)
        cv2.fillConvexPoly(eye_mask, eye_contour, 1)

        # Apply Gaussian smoothing to create a smooth transition
        weight_map = cv2.GaussianBlur(eye_mask, (0, 0), sigma)

        # Normalize the weight map to the range [0, 1]
        weight_map = weight_map / np.max(weight_map)

        return weight_map

    def create_pickle_data(self, img_list, save_path, eye_type):
        data = {'img': [], 'p': [], 'h': [], 'v': [], 'img_t': [], 'p_t': [], 'h_t': [], 'v_t': [],
                'landmarks': [], 'landmarks_t': [], 'weightMap': [], 'weightMapEyeball':[], 'weightMapEyeball_t':[], 'mg':[], 'mg_t':[]}
        
        for img_path in img_list:
            img, person_id, distance, head_pose, gaze_v, gaze_h = self.process_image(img_path)

            if (person_id not in self.ignore_list) and (gaze_v > -6):

                dir_name, file_name = os.path.split(img_path)
                dir_parts = dir_name.split('/')
                dir_parts.insert(-1, 'info')
                modified_dir_name = '/'.join(dir_parts)
                file_name_without_extension = os.path.splitext(file_name)[0]
                # modified_file_name = file_name_without_extension + ".txt"
                modified_file_name = file_name_without_extension + ".json"
                modified_file_path = os.path.join(modified_dir_name, modified_file_name)
                # eye_landmarks = self.read_landmarks_from_txt(modified_file_path)
                # middle_gaze_point = (32,24)
                eye_landmarks, middle_gaze_point = self.read_landmarks_from_json(modified_file_path)

                for img_path_t in img_list:
                    img_t, person_id_t, distance_t, head_pose_t, gaze_v_t, gaze_h_t = self.process_image(img_path_t)

                    dir_name, file_name = os.path.split(img_path_t)
                    dir_parts = dir_name.split('/')
                    dir_parts.insert(-1, 'info')
                    modified_dir_name = '/'.join(dir_parts)
                    file_name_without_extension = os.path.splitext(file_name)[0]
                    # modified_file_name = file_name_without_extension + ".txt"
                    modified_file_name = file_name_without_extension + ".json"
                    modified_file_path = os.path.join(modified_dir_name, modified_file_name)
                    # eye_landmarks_t = self.read_landmarks_from_txt(modified_file_path)
                    # middle_gaze_point_t = (32,24)
                    eye_landmarks_t, middle_gaze_point_t = self.read_landmarks_from_json(modified_file_path)


                    if person_id == person_id_t and head_pose == head_pose_t and gaze_h_t == 0 and gaze_v_t == 0:
                        # and head_pose <= 5 and abs(gaze_h) <= 20 and abs(gaze_v) <= 5
                        data['img'].append(img)
                        data['p'].append(head_pose)
                        data['h'].append(gaze_h)
                        data['v'].append(gaze_v)
                        data['landmarks'].append(eye_landmarks)

                        data['img_t'].append(img_t)
                        data['p_t'].append(head_pose_t)
                        data['h_t'].append(gaze_h_t)
                        data['v_t'].append(gaze_v_t)
                        data['landmarks_t'].append(eye_landmarks_t)

                        weight_map_o = self.create_weight_map_from_landmarks(eye_landmarks,(self.image_height,self.image_width))
                        weight_map_t = self.create_weight_map_from_landmarks(eye_landmarks_t,(self.image_height,self.image_width))
                        weight_map = np.maximum(weight_map_o, weight_map_t)

                        weight_map_gaze_o = self.create_gaussian_weight_map(middle_gaze_point,(self.image_height,self.image_width))
                        weight_map_gaze_t = self.create_gaussian_weight_map(middle_gaze_point_t,(self.image_height,self.image_width))
                        # weight_map_gaze = np.maximum(weight_map_gaze_o, weight_map_gaze_t)

                        data['weightMap'].append(weight_map)

                        data['weightMapEyeball'].append(weight_map_gaze_o)
                        data['weightMapEyeball_t'].append(weight_map_gaze_t)

                        data['mg'].append(middle_gaze_point)
                        data['mg_t'].append(middle_gaze_point_t)

                        print(img_path, img_path_t)

                        ###########################
                        # # print(weight_map.shape)
                        # plt.imshow(weight_map_gaze.squeeze(), cmap='hot')
                        # plt.title(f'Gaussian Weight Map with sigma')
                        # plt.colorbar()
                        # # plt.show()

                        # plt.savefig(f'gaussian_weight_map_sigma.png')
                        # plt.close()  # Close the figure to free up memory

                        break
        
        for key in data.keys():
            data[key] = np.array(data[key])
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, f'{eye_type}_data.pkl'), 'wb') as f:
            pickle.dump(data, f)
        
        print(f'{save_path}/{eye_type}_data.pkl Completed!')

    def save_as_pickle(self):
        # Process left eye images
        # left_img_list = glob.glob(self.preprocessing_dataset_dir + '/*/left/*.jpg')
        # left_img_list.sort()
        # self.create_pickle_data(left_img_list, self.save_pickle_path, 'left')

        # Process right eye images
        right_img_list = glob.glob(self.preprocessing_dataset_dir + '/*/right/*.jpg')
        right_img_list.sort()
        self.create_pickle_data(right_img_list, self.save_pickle_path, 'right')
    
    def load_pickle_data(self, folder_path):
        data_list = []

        # List all files in the directory
        for filename in os.listdir(folder_path):
            # Check if the file is a .pkl file
            if filename.endswith('.pkl'):
                file_path = os.path.join(folder_path, filename)
                # Load the pickle file
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                    # Append the loaded data to the data_list
                    data_list.append(data)

        return data_list
    

    def preprocess_data(self, img, p, h, v, img_t, p_t, h_t, v_t, landmarks, landmarks_t, weightMap, weightMapEyeball, weightMapEyeball_t, mg, mg_t):
        
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


        weightMap = tf.cast(weightMap, tf.float32)
        weightMap = tf.expand_dims(weightMap, axis=-1)  # Ensure it's [height, width, 1]

        weightMapEyeball = tf.cast(weightMapEyeball, tf.float32)
        weightMapEyeball = tf.expand_dims(weightMapEyeball, axis=-1)  # Ensure it's [height, width, 1]

        weightMapEyeball_t = tf.cast(weightMapEyeball_t, tf.float32)
        weightMapEyeball_t = tf.expand_dims(weightMapEyeball_t, axis=-1)  # Ensure it's [height, width, 1]

        # weightMap = tf.image.resize(weightMap, [tf.shape(img)[1], tf.shape(img)[2]])  # Resize if necessary

        # Cast middle gaze points to float32
        mg = tf.cast(mg, tf.float32)
        mg_t = tf.cast(mg_t, tf.float32)

        return (img_t, p_t, gaze_target, landmarks_t, weightMapEyeball_t, mg_t), (img, p, gaze_real, landmarks, weightMapEyeball, mg), weightMap

    def create_dataset(self, data, batch_size=32):
        dataset = tf.data.Dataset.from_tensor_slices((data['img'], data['p'], data['h'], data['v'], data['img_t'], data['p_t'], data['h_t'], data['v_t'],
                                                      data['landmarks'], data['landmarks_t'], data['weightMap'], data['weightMapEyeball'], data['weightMapEyeball_t'], data['mg'], data['mg_t']))
        dataset = dataset.map(self.preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    # Function to load and preprocess the image
    def load_and_preprocess_image(self, image_path, target_size=(48, 64)):
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


if __name__ == '__main__':
    processing_dataset = ProcessingDataset()
    # processing_dataset.preprocess_dataset()
    processing_dataset.save_as_pickle()