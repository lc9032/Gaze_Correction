import xml.etree.ElementTree as ET
import cv2
import os
import json
import re

users = 31
poses = ['-20', '-10', '0', '10', '20']
# poses = ['0']


base_dataset_folder = r"../DATA_SETS/T_DataSet"

pointsGrids = 15

image_width = 64
image_height = 48
horizontal_margin = 1.5
vertical_margin = 1.125

# Function to extract points from XML
def extract_points_from_xml(root, tag_name):
    elem_points = []
    
    for element in root.findall(f".//{tag_name}"):
        points = []
        if element is not None:
            for point in element.findall("Point2D/Vector2"):
                x = float(point.find('x').text)
                y = float(point.find('y').text)
                points.append((int(x), int(y)))

            elem_points.append(points)

    return elem_points

def extract_headpose_from_xml(root):
    elem_points = []
    
    for element in root.findall(".//HeadposeDef"):
        points = []
        if element is not None:
            pose_x = element.find("Rotation/x").text
            pose_y = element.find("Rotation/y").text
            pose_z = element.find("Rotation/z").text
            gaze_x = element.find("LookAtPoint/x").text
            gaze_y = element.find("LookAtPoint/y").text

            points.append((int(float(pose_x)), int(float(pose_y)), int(float(pose_z)), int(float(gaze_x)), int(float(gaze_y))))

        elem_points.append(points)

    return elem_points

def extract_eye_regions(image, left_most, right_most):
    Ec = (
        int((left_most[0] + right_most[0]) / 2),  # Average of x-coordinates
        int((left_most[1] + right_most[1]) / 2)   # Average of y-coordinates
    )
    L = abs(left_most[0] - right_most[0])

    # cv2.line(image, left_most, right_most, (0, 255, 0), thickness=5)
    # cv2.circle(image, Ec, 8, (0, 0, 255), -1)

    x = max(Ec[0] - (horizontal_margin/2) * L, 0)
    y = max(Ec[1] - (vertical_margin/2) * L, 0)
    w = min(horizontal_margin * L, image.shape[1] - x)
    h = min(vertical_margin * L, image.shape[0] - y)

    right_eye_region = image[int(y):int(y+h), int(x):int(x+w)]
    
    return right_eye_region ,x, y, w, h

def eye_landmarks(points, eye_region, x, y, w, h):
    info_points = []
    key_indices = [0, 3]

    info_indices = [1, 2, 4, 5]

    # Add points for key_indices
    for i in key_indices:
        point = points[i]
        resize_point = (int((point[0] - x) * eye_region.shape[1] / w), int((point[1] - y) * eye_region.shape[0] / h))
        # cv2.circle(eye_region, resize_point, 8, (255, 0, 0), -1)
        info_points.append(resize_point)

    # Calculate center points for center_pairs
    for i in info_indices:
        point = points[i]
        
        resize_point = (int((point[0] - x) * eye_region.shape[1] / w),int((point[1] - y) * eye_region.shape[0] / h))
        # cv2.circle(eye_region, resize_point, 8, (255, 255, 0), -1)
        info_points.append(resize_point)

    return info_points

def calculate_average_point(points):
    """Calculates the average (center) point from a list of points."""
    x_sum = sum([point[0] for point in points])
    y_sum = sum([point[1] for point in points])
    count = len(points)
    return (int(x_sum / count), int(y_sum / count))

def plot_points_on_image(user_number, pose, dataset_folder_imgs, dataset_folder_lbls, output_path):

    for i, filename_img in enumerate(os.listdir(dataset_folder_imgs)):
        # if (i != 0) or (user_number != 7) or (pose != '0'):
        #     break

        filename_lbl = filename_img.replace(".png", ".txt")

        r_points = []
        l_points = []
        with open(dataset_folder_lbls + '/' + filename_lbl, 'r') as file:
            for line in file:
                # Strip any leading/trailing whitespace and split the line by commas
                parts = line.strip().split(',')
                
                # Convert each part to an integer
                row = int(parts[0])
                column = int(parts[1])
                x = int(parts[2])
                y = int(parts[3])
                
                # Append the tuple (row, column, x, y) to the points list
                if row == 0:
                    r_points.append((x, y))
                else:
                    l_points.append((x, y))

        image = cv2.imread(dataset_folder_imgs + '/' + filename_img)

        r_left_most = r_points[0]
        r_right_most = r_points[3]
        right_gaze_point = r_points[6]


        index_mapping = {0: 3, 1: 2, 2: 1, 3: 0, 4: 5, 5: 4, 6: 6}
        l_points_m = [0] * len(l_points)
        for old_index, new_index in index_mapping.items():
            l_points_m[new_index] = l_points[old_index]

        l_left_most = l_points_m[0]
        l_right_most = l_points_m[3]
        left_gaze_point = l_points_m[6]

        match = re.search(r'_(\-?\d+)H_(\-?\d+)V', filename_img)
        pose_x = 0
        pose_y = pose
        pose_z = 0
        gaze_h = int(match.group(1))
        gaze_v = int(match.group(2))

        # cv2.circle(image, r_left_most, 5, (255, 0, 255), -1)
        # cv2.circle(image, r_right_most, 5, (255, 0, 255), -1)
        right_eye_region, rx, ry, rw, rh = extract_eye_regions(image, r_left_most, r_right_most)
        right_eye_region = cv2.resize(right_eye_region, (image_width, image_height))

    

        # cv2.circle(image, l_left_most, 5, (255, 0, 255), -1)
        # cv2.circle(image, l_right_most, 5, (255, 0, 255), -1)
        left_eye_region, lx, ly,lw,lh = extract_eye_regions(image, l_left_most, l_right_most)
        left_eye_region = cv2.resize(left_eye_region, (image_width, image_height))

        r_info_points = eye_landmarks(r_points[:6], right_eye_region, rx, ry, rw, rh)
        l_info_points = eye_landmarks(l_points_m[:6], left_eye_region, lx, ly, lw, lh)

        # for points in l_info_points:
        #     cv2.circle(left_eye_region, points, 5, (255, 0, 255), -1)


        resize_right_gaze_point = (int((right_gaze_point[0] - rx) * right_eye_region.shape[1] / rw), int((right_gaze_point[1] - ry) * right_eye_region.shape[0] / rh))
        # # cv2.circle(right_eye_region, resize_right_gaze_point, 2, (255, 0, 255), -1)

        resize_left_gaze_point = (int((left_gaze_point[0] - lx) * left_eye_region.shape[1] / lw), int((left_gaze_point[1] - ly) * right_eye_region.shape[0] / lh))
        # # cv2.circle(left_eye_region, resize_left_gaze_point, 2, (255, 0, 255), -1)


        # Save the output image
        if not os.path.exists(output_path):
            os.makedirs(output_path+"left")
            os.makedirs(output_path+"right")
            os.makedirs(output_path+"info/right")
            os.makedirs(output_path+"info/left")

        cv2.imwrite(output_path + f"right/{user_number:04}_2m_{pose_x}PX_{pose_y}PY_{pose_z}PZ_{gaze_v}V_{gaze_h}H.jpg", right_eye_region)
        print(f"Output saved to {output_path}")

        cv2.imwrite(output_path + f"left/{user_number:04}_2m_{pose_x}PX_{pose_y}PY_{pose_z}PZ_{gaze_v}V_{gaze_h}H.jpg", left_eye_region)
        print(f"Output saved to {output_path}")


        # Save the landmarks to a JSON file
        output_path_info_right = os.path.join(output_path, f"info/right/{user_number:04}_2m_{pose_x}PX_{pose_y}PY_{pose_z}PZ_{gaze_v}V_{gaze_h}H.json")
        output_path_info_left = os.path.join(output_path, f"info/left/{user_number:04}_2m_{pose_x}PX_{pose_y}PY_{pose_z}PZ_{gaze_v}V_{gaze_h}H.json")

        right_eye_data = {
            'landmarks': [{'x': point[0], 'y': point[1]} for point in r_info_points],
            'middle_gaze_point': {'x': resize_right_gaze_point[0], 'y': resize_right_gaze_point[1]}
        }

        left_eye_data = {
            'landmarks': [{'x': point[0], 'y': point[1]} for point in l_info_points],
            'middle_gaze_point': {'x': resize_left_gaze_point[0], 'y': resize_left_gaze_point[1]}
        }

        with open(output_path_info_right, "w") as f:
            json.dump(right_eye_data, f)

        with open(output_path_info_left, "w") as f:
            json.dump(left_eye_data, f)

        # cv2.imshow("Image(R)", right_eye_region)
        # cv2.imshow("Image(L)", left_eye_region)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# Example usage
# image_path = f"{base_dataset_folder}/{image_number:02}.png"


for user_number in range(0, users):
    user_folder = f"{user_number:04}"

    for pose in poses:

        dataset_folder_imgs = f"{base_dataset_folder}/imgs/{pose}/{user_folder}"
        dataset_folder_lbls = f"{base_dataset_folder}/lbls/{pose}/{user_folder}"

        output_path = f"./DataSets/preprocessing_dataset_DIRL_0824/{user_number:04}/"
        
        plot_points_on_image(user_number, pose, dataset_folder_imgs, dataset_folder_lbls, output_path)