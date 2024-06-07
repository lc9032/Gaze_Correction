import os
import glob
import numpy as np
import cv2
import pickle

dataset_dir = './preprocessing_dataset'
image_width = 64
image_height = 32
save_pickle_path = './training_inputs_x'

def process_image(img_path):
    # Load the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_width, image_height))  # Resize to the required dimensions
    
    # Extract additional information from the filename
    basename = os.path.basename(img_path)
    parts = basename.split('_')
    person_id = parts[0]
    distance = parts[1][:-1]  # Remove the 'm'
    head_pose = int(parts[2][:-1])  # Remove the 'P'
    gaze_v = int(parts[3][:-1])  # Remove the 'V'
    gaze_h = int(parts[4][:-5])  # Remove the 'H' and '.jpg'
    
    return img, person_id, distance, head_pose, gaze_v, gaze_h

def create_pickle_data(img_list, save_path, eye_type):
    data = {'img': [], 'p': [], 'h': [], 'v': [], 'img_t': [], 'p_t': [], 'h_t': [], 'v_t': []}
    
    for img_path in img_list:
        img, person_id, distance, head_pose, gaze_v, gaze_h = process_image(img_path)

        for img_path_t in img_list:
            img_t, person_id_t, distance_t, head_pose_t, gaze_v_t, gaze_h_t = process_image(img_path_t)

            if person_id == person_id_t and head_pose == head_pose_t and gaze_h_t == 0 and gaze_v_t == 0 and int(person_id) > 20 and int(person_id) < 30:
                data['img'].append(img)
                # data['img'].append(img_path)
                data['p'].append(head_pose)
                data['h'].append(gaze_h)
                data['v'].append(gaze_v)

                data['img_t'].append(img_t)
                # data['img_t'].append(img_path_t)
                data['p_t'].append(head_pose_t)
                data['h_t'].append(gaze_h_t)
                data['v_t'].append(gaze_v_t)

                print(img_path, img_path_t)

                break

        # print(img_path)
    
    for key in data.keys():
        data[key] = np.array(data[key])
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, f'{eye_type}_data.pkl'), 'wb') as f:
        pickle.dump(data, f)
    
    print(f'{save_path}/{eye_type}_data.pkl Completed!')

# # Process left eye images
# left_img_list = glob.glob(dataset_dir + '/*/left/*.jpg')
# # left_img_list = glob.glob(dataset_dir + '/0032/left/*.jpg')
# left_img_list.sort()
# create_pickle_data(left_img_list, save_pickle_path, 'left')

# Process right eye images
# right_img_list = glob.glob(dataset_dir + '/*/right/*.jpg')
# right_img_list = glob.glob(dataset_dir + '/0006/right/*.jpg')

right_img_list = glob.glob(dataset_dir + '/*/right/*.jpg')
right_img_list.sort()
create_pickle_data(right_img_list, save_pickle_path, 'right')
