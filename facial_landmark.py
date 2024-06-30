
import os
import cv2 # type: ignore
import numpy as np # type: ignore
import mediapipe as mp # type: ignore

class FacialLandmark:
    def __init__(self):
        self.horizontal_margin = 1.6
        self.vertical_margin = 0.8
        self.horizontal_offset = -5.0
        self.vertical_offset = -5.0

        self.left_eye_indices = [362, 385, 387, 380, 373, 263]
        self.right_eye_indices = [33, 160, 158, 144, 153, 133]

        # self.left_eye_indices = [362, 388]
        # self.right_eye_indices = [33, 133]

        self.face_width_indices = [234, 454]  # Outer corners of the eyes
        self.face_height_indices = [10, 152]  # Top of the forehead and bottom of the chin

    def extract_eye_points(self, frame, landmarks):
        left_eye_points = [(int(landmarks.landmark[idx].x * frame.shape[1]), 
                            int(landmarks.landmark[idx].y * frame.shape[0])) for idx in self.left_eye_indices]
        right_eye_points = [(int(landmarks.landmark[idx].x * frame.shape[1]), 
                            int(landmarks.landmark[idx].y * frame.shape[0])) for idx in self.right_eye_indices]
        
        return left_eye_points, right_eye_points
    
    
    def extract_face_dimensions(self, frame, landmarks):
        # Get face width
        left_face = landmarks.landmark[self.face_width_indices[0]]
        right_face = landmarks.landmark[self.face_width_indices[1]]
        face_width = int(abs(right_face.x - left_face.x) * frame.shape[1])

        # Get face height
        top_face = landmarks.landmark[self.face_height_indices[0]]
        bottom_face = landmarks.landmark[self.face_height_indices[1]]
        face_height = int(abs(bottom_face.y - top_face.y) * frame.shape[0])

        self.top_margin = int((face_width + face_height) * 0.012 * 2)#0.16
        self.bottom_mergin = int((face_width + face_height) * 0.012 * 2)
        self.right_mergin = int((face_width + face_height) * 0.012 * 2)
        self.left_mergin = int((face_width + face_height) * 0.012 * 2)

        return face_width, face_height

    # def extract_eye_bboxs(self, frame, left_eye_points, right_eye_points):
    #     # Extract the bounding boxes around the eyes
    #     left_eye_bbox = cv2.boundingRect(np.array(left_eye_points))
    #     right_eye_bbox = cv2.boundingRect(np.array(right_eye_points))
        
    #     # Add margin to the bounding boxes
    #     left_eye_bbox = (max(left_eye_bbox[0] - self.left_mergin, 0),
    #                      max(left_eye_bbox[1] - self.top_margin, 0),
    #                      min(left_eye_bbox[2] + 2 * self.right_mergin, frame.shape[1] - left_eye_bbox[0] + self.right_mergin),
    #                      min(left_eye_bbox[3] + 2 * self.bottom_mergin, frame.shape[0] - left_eye_bbox[1] + self.bottom_mergin))
        
    #     right_eye_bbox = (max(right_eye_bbox[0] - self.left_mergin, 0),
    #                       max(right_eye_bbox[1] - self.top_margin, 0),
    #                       min(right_eye_bbox[2] + 2 * self.right_mergin, frame.shape[1] - right_eye_bbox[0] + self.right_mergin),
    #                       min(right_eye_bbox[3] + 2 * self.bottom_mergin, frame.shape[0] - right_eye_bbox[1] + self.bottom_mergin))

    #     return left_eye_bbox, right_eye_bbox
    
    def extract_eye_bboxs(self, frame, left_eye_points, right_eye_points):

        def calculate_bbox(points):
            # Calculate the center point (Ec)
            Ec_x = (points[0][0] + points[len(points)-1][0]) / 2
            Ec_y = (points[0][1] + points[len(points)-1][1]) / 2
            Ec = (int(Ec_x), int(Ec_y))

            # Find the leftmost and rightmost points
            # left_point = min(points, key=lambda p: p[0])
            # right_point = max(points, key=lambda p: p[0])
            
            # Calculate L (distance between leftmost and rightmost points)
            L = abs(points[len(points)-1][0] - points[0][0])
            
            # Calculate bounding box
            x = max(Ec[0] - (self.horizontal_margin/2) * L, 0)
            y = max(Ec[1] - (self.vertical_margin/2) * L, 0)
            w = min(self.horizontal_margin * L, frame.shape[1] - x)
            h = min(self.vertical_margin * L, frame.shape[0] - y)

            x = x + self.horizontal_offset
            y = y + self.vertical_offset
            
            return (int(x), int(y), int(w), int(h))

        left_eye_bbox = calculate_bbox(left_eye_points)
        right_eye_bbox = calculate_bbox(right_eye_points)

        return left_eye_bbox, right_eye_bbox
        
    def extract_eye_regions(self, frame, landmarks):
        left_eye_points, right_eye_points = self.extract_eye_points(frame, landmarks)
        _, _ = self.extract_face_dimensions(frame, landmarks)
        left_eye_bbox, right_eye_bbox = self.extract_eye_bboxs(frame, left_eye_points, right_eye_points)

        # Crop the eye regions with the added margin
        left_eye_region = frame[left_eye_bbox[1]:left_eye_bbox[1] + left_eye_bbox[3], 
                                left_eye_bbox[0]:left_eye_bbox[0] + left_eye_bbox[2]]
        right_eye_region = frame[right_eye_bbox[1]:right_eye_bbox[1] + right_eye_bbox[3], 
                                 right_eye_bbox[0]:right_eye_bbox[0] + right_eye_bbox[2]]

        # Adjust the landmark points to match the new coordinate systems of the cropped regions
        left_eye_landmarks = [(pt[0] - left_eye_bbox[0], pt[1] - left_eye_bbox[1]) for pt in left_eye_points]
        right_eye_landmarks = [(pt[0] - right_eye_bbox[0], pt[1] - right_eye_bbox[1]) for pt in right_eye_points]

        return left_eye_region, right_eye_region, left_eye_landmarks, right_eye_landmarks
    

    def replace_eye_regions(self, frame, landmarks, new_left_eye, new_right_eye):
        left_eye_points, right_eye_points = self.extract_eye_points(frame, landmarks)
        left_eye_bbox, right_eye_bbox = self.extract_eye_bboxs(frame, left_eye_points, right_eye_points)

        # Extract the bounding boxes' dimensions
        left_x, left_y, left_w, left_h = left_eye_bbox
        right_x, right_y, right_w, right_h = right_eye_bbox

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