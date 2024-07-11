import os
import pickle
import tensorflow as tf # type: ignore
import glob
import numpy as np # type: ignore
import cv2 # type: ignore

from facial_landmark import FacialLandmark

class ProcessingDataset:
    def __init__(self):
        self.image_width = 64
        self.image_height = 48
        
        self.base_dataset_folder = r"../DATA_SETS/C_DataSet/columbia_gaze_data_set/Columbia Gaze Data Set"

        self.preprocessing_dataset_dir = './preprocessing_dataset_COL_48'

        self.save_pickle_path = './training_inputs_COL_48'
        # self.ignore_list = ['0008', '0010', '0011', '0016', '0024', '0025', '0043', '0053']
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


            dir_name = os.path.dirname(input_path)
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
                input_path = os.path.join(input_folder, filename)
                self.preprocess_image(input_path, output_folder)
    
    def preprocess_dataset(self):
        for i in range(1, 57):
            folder_number = f"{i:04d}"
            input_folder = os.path.join(self.base_dataset_folder, folder_number)
            output_folder = os.path.join(self.preprocessing_dataset_dir, folder_number)

            self.process_images_in_folder(input_folder, output_folder)
        pass
    
    def process_image(self, img_path):
        # Load the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_width, self.image_height))  # Resize to the required dimensions
        
        # Extract additional information from the filename
        basename = os.path.basename(img_path)
        parts = basename.split('_')
        person_id = parts[0]
        distance = parts[1][:-1]  # Remove the 'm'
        head_pose = int(parts[2][:-1])  # Remove the 'P'
        gaze_v = int(parts[3][:-1])  # Remove the 'V'
        gaze_h = int(parts[4][:-5])  # Remove the 'H' and '.jpg'
        
        return img, person_id, distance, head_pose, gaze_v, gaze_h

    def read_landmarks_from_txt(self, file_path):
        landmarks = []
        with open(file_path, 'r') as file:
            for line in file:
                x, y = map(int, line.strip().split(','))
                landmarks.append((x, y))
        return landmarks

    def create_pickle_data(self, img_list, save_path, eye_type):
        data = {'img': [], 'p': [], 'h': [], 'v': [], 'img_t': [], 'p_t': [], 'h_t': [], 'v_t': [], 'landmarks': [], 'landmarks_t': []}
        
        for img_path in img_list:
            img, person_id, distance, head_pose, gaze_v, gaze_h = self.process_image(img_path)

            if person_id not in self.ignore_list:

                dir_name, file_name = os.path.split(img_path)
                dir_parts = dir_name.split('/')
                dir_parts.insert(-1, 'info')
                modified_dir_name = '/'.join(dir_parts)
                file_name_without_extension = os.path.splitext(file_name)[0]
                modified_file_name = file_name_without_extension + ".txt"
                modified_file_path = os.path.join(modified_dir_name, modified_file_name)
                eye_landmarks = self.read_landmarks_from_txt(modified_file_path)

                for img_path_t in img_list:
                    img_t, person_id_t, distance_t, head_pose_t, gaze_v_t, gaze_h_t = self.process_image(img_path_t)

                    dir_name, file_name = os.path.split(img_path)
                    dir_parts = dir_name.split('/')
                    dir_parts.insert(-1, 'info')
                    modified_dir_name = '/'.join(dir_parts)
                    file_name_without_extension = os.path.splitext(file_name)[0]
                    modified_file_name = file_name_without_extension + ".txt"
                    modified_file_path = os.path.join(modified_dir_name, modified_file_name)
                    eye_landmarks_t = self.read_landmarks_from_txt(modified_file_path)

                    if person_id == person_id_t and head_pose == head_pose_t and gaze_h_t == 0 and gaze_v_t == 0:
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

                        print(img_path, img_path_t)

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
        left_img_list = glob.glob(self.preprocessing_dataset_dir + '/*/left/*.jpg')
        left_img_list.sort()
        self.create_pickle_data(left_img_list, self.save_pickle_path, 'left')

        # Process right eye images
        right_img_list = glob.glob(self.preprocessing_dataset_dir + '/*/right/*.jpg')
        right_img_list.sort()
        self.create_pickle_data(right_img_list, self.save_pickle_path, 'right')
    
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


if __name__ == '__main__':
    processing_dataset = ProcessingDataset()
    # processing_dataset.preprocess_dataset()
    processing_dataset.save_as_pickle()