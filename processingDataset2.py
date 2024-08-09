import xml.etree.ElementTree as ET
import cv2

base_dataset_folder = r"../DATA_SETS/U2/User_01/Grid_15/HP_001"

image_number = 8

horizontal_margin = 1.6
vertical_margin = 1.2

# Function to extract points from XML
# def extract_points_from_xml(root, tag_name):
#     points = []
#     for point in root.findall(f".//{tag_name}/Point2D/Vector2"):
#         x = float(point.find('x').text)
#         y = float(point.find('y').text)
#         points.append((int(x), int(y)))
#     return points
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


def plot_points_on_image(image_path, xml_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image {image_path}")
        return
    
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract points
    caruncle_points = extract_points_from_xml(root, 'Caruncle')
    interior_margin_points = extract_points_from_xml(root, 'InteriorMargin')
    iris_points = extract_points_from_xml(root, 'Iris')
    pupil_points = extract_points_from_xml(root, 'Pupil')

    # Plot points on the image
    ##left
    # for i, point in enumerate(caruncle_points[image_number-1]):
    #     cv2.circle(image, point, 3, (0, 255, 0), -1)

    # for i, point in enumerate(interior_margin_points[image_number-1]):
    #     cv2.circle(image, point, 3, (255, 0, 0), -1)
    #     if i == 0:
    #         cv2.circle(image, point, 5, (255, 0, 0), -1)
    #     if i == 8:
    #         cv2.circle(image, point, 5, (255, 0, 0), -1)

    # for point in iris_points[image_number-1]:
    #     cv2.circle(image, point, 3, (0, 0, 255), -1)
    # for point in pupil_points[image_number-1]:
    #     cv2.circle(image, point, 3, (255, 255, 0), -1)

    r_left_most = interior_margin_points[image_number-1][0]
    r_right_most = interior_margin_points[image_number-1][8]
    Ec = (
        int((r_left_most[0] + r_right_most[0]) / 2),  # Average of x-coordinates
        int((r_left_most[1] + r_right_most[1]) / 2)   # Average of y-coordinates
    )
    L = abs(r_left_most[0] - r_right_most[0])

    x = max(Ec[0] - (horizontal_margin/2) * L, 0)
    y = max(Ec[1] - (vertical_margin/2) * L, 0)
    w = min(horizontal_margin * L, image.shape[1] - x)
    h = min(vertical_margin * L, image.shape[0] - y)

    cv2.circle(image, r_left_most, 5, (255, 0, 255), -1)
    cv2.circle(image, r_right_most, 5, (255, 0, 255), -1)


    x2 = x + w
    y2 = y + h
    cv2.rectangle(image, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.circle(image, Ec, 5, (255, 0, 255), -1)

    ##right
    # for i, point in enumerate(caruncle_points[image_number-1+15]):
    #     cv2.circle(image, point, 3, (0, 255, 0), -1)

    # for i, point in enumerate(interior_margin_points[image_number-1+15]):
    #     cv2.circle(image, point, 3, (255, 0, 0), -1)
    #     if i == 0:
    #         cv2.circle(image, point, 5, (255, 0, 0), -1)
    #     if i == 8:
    #         cv2.circle(image, point, 5, (255, 0, 0), -1)

    # for point in iris_points[image_number-1+15]:
    #     cv2.circle(image, point, 3, (0, 0, 255), -1)
    # for point in pupil_points[image_number-1+15]:
    #     cv2.circle(image, point, 3, (255, 255, 0), -1)

    l_left_most = interior_margin_points[image_number-1+15][0]
    l_right_most = interior_margin_points[image_number-1+15][8]
    cv2.circle(image, l_left_most, 5, (255, 0, 255), -1)
    cv2.circle(image, l_right_most, 5, (255, 0, 255), -1)



    # # Save the output image
    # cv2.imwrite(output_path, image)
    # print(f"Output saved to {output_path}")

    # for point in manual_points:
    #     cv2.circle(image, point, 3, (0, 255, 0), -1)

    cv2.imshow("Image with Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = f"{base_dataset_folder}/{image_number:02}.png"
xml_path = base_dataset_folder + "/many_poi_data.xml"
output_path = "/mnt/data/output.png"

plot_points_on_image(image_path, xml_path, output_path)