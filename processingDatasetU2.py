import xml.etree.ElementTree as ET
import cv2
import os
import json

users = 20
hps = 125


base_dataset_folder = r"../DATA_SETS/U2"

pointsGrids = 15

image_width = 64
image_height = 48
horizontal_margin = 2.2#2.0#1.6
vertical_margin = 1.65#1.5#1.2

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
    key_indices = [0, 8]
    center_pairs = [(2, 3), (4, 5), (11, 12), (13, 14)]

    # Add points for key_indices
    for i in key_indices:
        point = points[i]
        resize_point = (int((point[0] - x) * eye_region.shape[1] / w), int((point[1] - y) * eye_region.shape[0] / h))
        # cv2.circle(eye_region, resize_point, 8, (255, 0, 0), -1)
        info_points.append(resize_point)

    # Calculate center points for center_pairs
    for p1, p2 in center_pairs:
        point1 = points[p1]
        point2 = points[p2]
        
        # Calculate the center point
        center_point = ((point1[0] + point2[0]) / 2,(point1[1] + point2[1]) / 2)
        resize_center_point = (int((center_point[0] - x) * eye_region.shape[1] / w),int((center_point[1] - y) * eye_region.shape[0] / h))
        # cv2.circle(eye_region, resize_center_point, 8, (255, 255, 0), -1)
        info_points.append(resize_center_point)

    return info_points

def calculate_average_point(points):
    """Calculates the average (center) point from a list of points."""
    x_sum = sum([point[0] for point in points])
    y_sum = sum([point[1] for point in points])
    count = len(points)
    return (int(x_sum / count), int(y_sum / count))

def plot_points_on_image(user_number, dataset_folder, xml_path, pose_xml_path, output_path):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract points
    caruncle_points = extract_points_from_xml(root, 'Caruncle')
    interior_margin_points = extract_points_from_xml(root, 'InteriorMargin')
    iris_points = extract_points_from_xml(root, 'Iris')
    pupil_points = extract_points_from_xml(root, 'Pupil')

    poseTree = ET.parse(pose_xml_path)
    poseRoot = poseTree.getroot()
    poseGaze = extract_headpose_from_xml(poseRoot)

    for image_number in range(1,pointsGrids+1):
    # for image_number in range(1,2):
        image_path = f"{dataset_folder}/{image_number:02}.png"
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image {image_path}")
            break
    
        pose_x = poseGaze[image_number-1][0][0]
        pose_z = poseGaze[image_number-1][0][2]
        if pose_x > 1 or pose_x < -1:
        # if pose_x != 0:
            break
        if pose_z > 1 or pose_z < -1:
            break

        pose_y = poseGaze[image_number-1][0][1]
        gaze_y = poseGaze[image_number-1][0][3]
        gaze_x = -poseGaze[image_number-1][0][4]

        r_left_most = interior_margin_points[image_number-1][0] #caruncle_points[image_number-1][6]#
        r_right_most = interior_margin_points[image_number-1][8]
        # cv2.circle(image, r_left_most, 5, (255, 0, 255), -1)
        # cv2.circle(image, r_right_most, 5, (255, 0, 255), -1)
        right_eye_region, rx, ry, rw, rh = extract_eye_regions(image, r_left_most, r_right_most)
        right_eye_region = cv2.resize(right_eye_region, (image_width, image_height))

        
        l_left_most = interior_margin_points[image_number-1+pointsGrids][0] #caruncle_points[image_number-1+pointsGrids][6]#
        l_right_most = interior_margin_points[image_number-1+pointsGrids][8]
        # cv2.circle(image, l_left_most, 5, (255, 0, 255), -1)
        # cv2.circle(image, l_right_most, 5, (255, 0, 255), -1)
        left_eye_region, lx, ly,lw,lh = extract_eye_regions(image, l_left_most, l_right_most)
        left_eye_region = cv2.resize(left_eye_region, (image_width, image_height))

        r_info_points = eye_landmarks(interior_margin_points[image_number-1], right_eye_region, rx, ry, rw, rh)
        l_info_points = eye_landmarks(interior_margin_points[image_number-1+pointsGrids], left_eye_region, lx, ly, lw, lh)

        right_gaze_point = calculate_average_point(pupil_points[image_number - 1])
        resize_right_gaze_point = (int((right_gaze_point[0] - rx) * right_eye_region.shape[1] / rw), int((right_gaze_point[1] - ry) * right_eye_region.shape[0] / rh))
        # cv2.circle(right_eye_region, resize_right_gaze_point, 2, (255, 0, 255), -1)

        left_gaze_point = calculate_average_point(pupil_points[image_number - 1 + pointsGrids])
        resize_left_gaze_point = (int((left_gaze_point[0] - lx) * left_eye_region.shape[1] / lw), int((left_gaze_point[1] - ly) * right_eye_region.shape[0] / lh))
        # cv2.circle(left_eye_region, resize_left_gaze_point, 2, (255, 0, 255), -1)


        # Save the output image
        if not os.path.exists(output_path):
            os.makedirs(output_path+"left")
            os.makedirs(output_path+"right")
            os.makedirs(output_path+"info/right")
            os.makedirs(output_path+"info/left")

        cv2.imwrite(output_path + f"right/{user_number:04}_2m_{pose_x}PX_{pose_y}PY_{pose_z}PZ_{gaze_x}V_{gaze_y}H.jpg", right_eye_region)
        print(f"Output saved to {output_path}")

        cv2.imwrite(output_path + f"left/{user_number:04}_2m_{pose_x}PX_{pose_y}PY_{pose_z}PZ_{gaze_x}V_{gaze_y}H.jpg", left_eye_region)
        print(f"Output saved to {output_path}")

        # # Save the landmarks to a text file
        # output_path_info = os.path.join(output_path, f"info/right/{user_number:04}_2m_{pose_x}PX_{pose_y}PY_{pose_z}PZ_{gaze_x}V_{gaze_y}H.txt")
        # with open(output_path_info, "w") as f:
        #     # f.write("Left Eye Landmarks:\n")
        #     for point in r_info_points:
        #         f.write(f"{point[0]},{point[1]}\n")

        # output_path_info = os.path.join(output_path, f"info/left/{user_number:04}_2m_{pose_x}PX_{pose_y}PY_{pose_z}PZ_{gaze_x}V_{gaze_y}H.txt")
        # with open(output_path_info, "w") as f:
        #     # f.write("\nRight Eye Landmarks:\n")
        #     for point in l_info_points:
        #         f.write(f"{point[0]},{point[1]}\n")

        # Save the landmarks to a JSON file
        output_path_info_right = os.path.join(output_path, f"info/right/{user_number:04}_2m_{pose_x}PX_{pose_y}PY_{pose_z}PZ_{gaze_x}V_{gaze_y}H.json")
        output_path_info_left = os.path.join(output_path, f"info/left/{user_number:04}_2m_{pose_x}PX_{pose_y}PY_{pose_z}PZ_{gaze_x}V_{gaze_y}H.json")

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


for user_number in range(1, users+1):
    user_folder = f"User_{user_number:02}"

    for hp_number in range(1, hps+1):
        hp_folder = f"HP_{hp_number:03}"
        dataset_folder = f"{base_dataset_folder}/{user_folder}/Grid_{pointsGrids}/{hp_folder}"

        xml_path = dataset_folder + "/many_poi_data.xml"
        pose_xml_path = dataset_folder + "/many_headpose.xml"
        output_path = f"./DataSets/preprocessing_dataset_U2_0817/{user_number:04}/"

        plot_points_on_image(user_number ,dataset_folder, xml_path, pose_xml_path, output_path)