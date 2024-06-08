
import os
import cv2
# import multiprocessing as mpg
import numpy as np
import mediapipe as mp

class facial_landmark:
    def __init__(self):
        pass

    def extract_eye_regions(self, frame, landmarks, top_margin=80, bottom_mergin=60,right_mergin=40,left_mergin=40):
        left_eye_indices = [362, 385, 387, 263, 373, 380, 374, 390, 249, 466, 388]
        right_eye_indices = [33, 160, 158, 133, 153, 144, 163, 7, 173, 246, 161]

        left_eye_points = [(int(landmarks.landmark[idx].x * frame.shape[1]), 
                            int(landmarks.landmark[idx].y * frame.shape[0])) for idx in left_eye_indices]
        right_eye_points = [(int(landmarks.landmark[idx].x * frame.shape[1]), 
                            int(landmarks.landmark[idx].y * frame.shape[0])) for idx in right_eye_indices]

        # Extract the bounding boxes around the eyes
        left_eye_bbox = cv2.boundingRect(np.array(left_eye_points))
        right_eye_bbox = cv2.boundingRect(np.array(right_eye_points))
        
        # Add margin to the bounding boxes
        left_eye_bbox = (max(left_eye_bbox[0] - left_mergin, 0),
                         max(left_eye_bbox[1] - top_margin, 0),
                         min(left_eye_bbox[2] + 2 * right_mergin, frame.shape[1] - left_eye_bbox[0] + right_mergin),
                         min(left_eye_bbox[3] + 2 * bottom_mergin, frame.shape[0] - left_eye_bbox[1] + bottom_mergin))
        
        right_eye_bbox = (max(right_eye_bbox[0] - left_mergin, 0),
                          max(right_eye_bbox[1] - top_margin, 0),
                          min(right_eye_bbox[2] + 2 * right_mergin, frame.shape[1] - right_eye_bbox[0] + right_mergin),
                          min(right_eye_bbox[3] + 2 * bottom_mergin, frame.shape[0] - right_eye_bbox[1] + bottom_mergin))
        
        # Crop the eye regions with the added margin
        left_eye_region = frame[left_eye_bbox[1]:left_eye_bbox[1] + left_eye_bbox[3], 
                                left_eye_bbox[0]:left_eye_bbox[0] + left_eye_bbox[2]]
        right_eye_region = frame[right_eye_bbox[1]:right_eye_bbox[1] + right_eye_bbox[3], 
                                 right_eye_bbox[0]:right_eye_bbox[0] + right_eye_bbox[2]]


        # Adjust the landmark points to match the new coordinate systems of the cropped regions
        left_eye_landmarks = [(pt[0] - left_eye_bbox[0], pt[1] - left_eye_bbox[1]) for pt in left_eye_points]
        right_eye_landmarks = [(pt[0] - right_eye_bbox[0], pt[1] - right_eye_bbox[1]) for pt in right_eye_points]

    
        
        return left_eye_region, right_eye_region, left_eye_landmarks, right_eye_landmarks
    
    def extract_eye_regions_combined(self, frame, landmarks, top_margin=1, bottom_mergin=1,right_mergin=1,left_mergin=1):
        left_eye_indices = [362, 385, 387, 263, 373, 380, 374, 390, 249, 466, 388]
        right_eye_indices = [33, 160, 158, 133, 153, 144, 163, 7, 173, 246, 161]

        left_eye_points = [(int(landmarks.landmark[idx].x * frame.shape[1]), 
                            int(landmarks.landmark[idx].y * frame.shape[0])) for idx in left_eye_indices]
        right_eye_points = [(int(landmarks.landmark[idx].x * frame.shape[1]), 
                            int(landmarks.landmark[idx].y * frame.shape[0])) for idx in right_eye_indices]
        
        # Extract the bounding boxes around the eyes
        left_eye_bbox = cv2.boundingRect(np.array(left_eye_points))
        right_eye_bbox = cv2.boundingRect(np.array(right_eye_points))
        
        # Add margin to the bounding boxes
        left_eye_bbox = (max(left_eye_bbox[0] - left_mergin, 0),
                         max(left_eye_bbox[1] - top_margin, 0),
                         left_eye_bbox[2] + right_mergin,
                         left_eye_bbox[3] + bottom_mergin)
        
        right_eye_bbox = (max(right_eye_bbox[0] - left_mergin, 0),
                          max(right_eye_bbox[1] - top_margin, 0),
                          right_eye_bbox[2] + right_mergin,
                          right_eye_bbox[3] + bottom_mergin)
        
        # Crop the eye regions with the added margin
        left_eye_region = frame[left_eye_bbox[1]:left_eye_bbox[1] + left_eye_bbox[3], 
                                left_eye_bbox[0]:left_eye_bbox[0] + left_eye_bbox[2]]
        right_eye_region = frame[right_eye_bbox[1]:right_eye_bbox[1] + right_eye_bbox[3], 
                                 right_eye_bbox[0]:right_eye_bbox[0] + right_eye_bbox[2]]
        
        # Resize the eye regions to be the same height (optional, but often useful for concatenation)
        if left_eye_region.shape[0] != right_eye_region.shape[0]:
            new_height = min(left_eye_region.shape[0], right_eye_region.shape[0])
            left_eye_region = cv2.resize(left_eye_region, (left_eye_region.shape[1], new_height))
            right_eye_region = cv2.resize(right_eye_region, (right_eye_region.shape[1], new_height))
        
        # Concatenate the eye regions horizontally
        combined_eye_region = np.hstack((right_eye_region, left_eye_region))

        return combined_eye_region
    
    def replace_eye_regions(self, frame, landmarks, new_eye_regions):
        left_eye_indices = [362, 385, 387, 263, 373, 380, 374, 390, 249, 466, 388]
        right_eye_indices = [33, 160, 158, 133, 153, 144, 163, 7, 173, 246, 161]

        left_eye_points = [(int(landmarks.landmark[idx].x * frame.shape[1]), 
                            int(landmarks.landmark[idx].y * frame.shape[0])) for idx in left_eye_indices]
        right_eye_points = [(int(landmarks.landmark[idx].x * frame.shape[1]), 
                            int(landmarks.landmark[idx].y * frame.shape[0])) for idx in right_eye_indices]
        
        # Extract the bounding boxes around the eyes
        left_eye_bbox = cv2.boundingRect(np.array(left_eye_points))
        right_eye_bbox = cv2.boundingRect(np.array(right_eye_points))


        # Extract the bounding boxes' dimensions
        left_x, left_y, left_w, left_h = left_eye_bbox
        right_x, right_y, right_w, right_h = right_eye_bbox

        # Calculate the split point (halfway through the combined eye regions)
        split_point = new_eye_regions.shape[1] // 2

        # Split the combined new eye regions into left and right eye regions
        new_left_eye = new_eye_regions[:, split_point:]
        new_right_eye = new_eye_regions[:, :split_point]

        # Resize the new eye regions to match the original bounding box sizes
        new_left_eye_resized = cv2.resize(new_left_eye, (left_w, left_h))
        new_right_eye_resized = cv2.resize(new_right_eye, (right_w, right_h))

        # Replace the corresponding regions in the original frame with the new eye regions
        frame[left_y:left_y + left_h, left_x:left_x + left_w] = new_left_eye_resized
        frame[right_y:right_y + right_h, right_x:right_x + right_w] = new_right_eye_resized

        return frame

    def face_landmark(self, image):

        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

            results = face_mesh.process(image)

        return results.multi_face_landmarks
    
    def process_image(self, input_path, output_path):
        image = cv2.imread(input_path)
        if image is None:
            print(f"Failed to read image {input_path}")
            return

        face_landmarks = self.face_landmark(image)

        if face_landmarks:    
            # eye_regions = self.extract_eye_regions_combined(image, face_landmarks[0])
            eye_regions_left, eye_regions_right, left_eye_landmarks, right_eye_landmarks= self.extract_eye_regions(image, face_landmarks[0])

            # eye_regions = cv2.resize(eye_regions, (256, 256)) / 255.0

            # Scale the landmark points to match the resized dimensions
            left_eye_landmarks_resized = [(int(pt[0] * 64 / eye_regions_left.shape[1]), int(pt[1] * 32 / eye_regions_left.shape[0])) for pt in left_eye_landmarks]
            right_eye_landmarks_resized = [(int(pt[0] * 64 / eye_regions_right.shape[1]), int(pt[1] * 32 / eye_regions_right.shape[0])) for pt in right_eye_landmarks]

            eye_regions_left = cv2.resize(eye_regions_left, (64, 32)) / 255.0
            eye_regions_right = cv2.resize(eye_regions_right, (64, 32)) / 255.0

            
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
                self.process_image(input_path, output_folder)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            
            face_landmarks = self.face_landmark(image)

            if face_landmarks:    
                # left_eye, right_eye = self.extract_eye_regions(frame, face_landmarks[0])
                eye_regions = self.extract_eye_regions_combined(frame, face_landmarks[0])
                            
                # Normalize eye regions (example)
                # left_eye = cv2.resize(left_eye, (64, 64)) / 255.0
                # right_eye = cv2.resize(right_eye, (64, 64)) / 255.0
                eye_regions = cv2.resize(eye_regions, (256, 128)) / 255.0

                flipped = cv2.flip(frame, 1)

                # Show the eye regions
                # cv2.imshow('Left Eye', left_eye)
                # cv2.imshow('Right Eye', right_eye)
                cv2.imshow('Eyes', eye_regions)
                cv2.imshow('DBG', flipped)
                    
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    face_ex = facial_landmark()

    base_input_folder = r"../DATA_SETS/C_DataSet/columbia_gaze_data_set/Columbia Gaze Data Set"

    base_output_folder = "./preprocessing_dataset_NEW"

    for i in range(29, 30):
        folder_number = f"{i:04d}"
        input_folder = os.path.join(base_input_folder, folder_number)
        output_folder = os.path.join(base_output_folder, folder_number)

        face_ex.process_images_in_folder(input_folder, output_folder)
    # face_ex.run()