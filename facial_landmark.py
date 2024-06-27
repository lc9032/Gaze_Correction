
import os
import cv2 # type: ignore
import numpy as np # type: ignore
import mediapipe as mp # type: ignore

class FacialLandmark:
    def __init__(self):
        self.top_margin = 10
        self.bottom_mergin = 10
        self.right_mergin = 5
        self.left_mergin = 5

        # self.top_margin = 60
        # self.bottom_mergin = 60
        # self.right_mergin = 30
        # self.left_mergin = 30

        self.left_eye_indices = [362, 385, 387, 263, 373, 380, 374, 390, 249, 466, 388]
        self.right_eye_indices = [33, 160, 158, 133, 153, 144, 163, 7, 173, 246, 161]

    def extract_eye_points(self, frame, landmarks):
        left_eye_points = [(int(landmarks.landmark[idx].x * frame.shape[1]), 
                            int(landmarks.landmark[idx].y * frame.shape[0])) for idx in self.left_eye_indices]
        right_eye_points = [(int(landmarks.landmark[idx].x * frame.shape[1]), 
                            int(landmarks.landmark[idx].y * frame.shape[0])) for idx in self.right_eye_indices]
        
        return left_eye_points, right_eye_points


    def extract_eye_bboxs(self, frame, left_eye_points, right_eye_points):
        # Extract the bounding boxes around the eyes
        left_eye_bbox = cv2.boundingRect(np.array(left_eye_points))
        right_eye_bbox = cv2.boundingRect(np.array(right_eye_points))
        
        # Add margin to the bounding boxes
        left_eye_bbox = (max(left_eye_bbox[0] - self.left_mergin, 0),
                         max(left_eye_bbox[1] - self.top_margin, 0),
                         min(left_eye_bbox[2] + 2 * self.right_mergin, frame.shape[1] - left_eye_bbox[0] + self.right_mergin),
                         min(left_eye_bbox[3] + 2 * self.bottom_mergin, frame.shape[0] - left_eye_bbox[1] + self.bottom_mergin))
        
        right_eye_bbox = (max(right_eye_bbox[0] - self.left_mergin, 0),
                          max(right_eye_bbox[1] - self.top_margin, 0),
                          min(right_eye_bbox[2] + 2 * self.right_mergin, frame.shape[1] - right_eye_bbox[0] + self.right_mergin),
                          min(right_eye_bbox[3] + 2 * self.bottom_mergin, frame.shape[0] - right_eye_bbox[1] + self.bottom_mergin))

        return left_eye_bbox, right_eye_bbox
    
    def extract_eye_regions(self, frame, landmarks):
        left_eye_points, right_eye_points = self.extract_eye_points(frame, landmarks)
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
    
#     def run(self):
#         cap = cv2.VideoCapture(0)
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Convert the BGR image to RGB
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             face_landmarks = self.face_landmark(image)

#             if face_landmarks:    
#                 left_eye_region, right_eye_region, _, _ = self.extract_eye_regions(frame, face_landmarks[0])

#                 # Resize the eye regions to be the same height (optional, but often useful for concatenation)
#                 if left_eye_region.shape[0] != right_eye_region.shape[0]:
#                     new_height = min(left_eye_region.shape[0], right_eye_region.shape[0])
#                     left_eye_region = cv2.resize(left_eye_region, (left_eye_region.shape[1], new_height))
#                     right_eye_region = cv2.resize(right_eye_region, (right_eye_region.shape[1], new_height))
#                 combined_eye_region = np.hstack((right_eye_region, left_eye_region))
#                 # eye_regions = self.extract_eye_regions_combined(frame, face_landmarks[0])
                            
#                 # Normalize eye regions (example)
#                 # left_eye = cv2.resize(left_eye, (64, 64)) / 255.0
#                 # right_eye = cv2.resize(right_eye, (64, 64)) / 255.0
#                 # eye_regions = cv2.resize(eye_regions, (256, 128)) / 255.0

#                 flipped = cv2.flip(frame, 1)

#                 # Show the eye regions
#                 # cv2.imshow('Left Eye', left_eye)
#                 # cv2.imshow('Right Eye', right_eye)
#                 cv2.imshow('Eyes', combined_eye_region)
#                 cv2.imshow('DBG', flipped)
                    
#                 if cv2.waitKey(5) & 0xFF == 27:
#                     break

#         cap.release()
#         cv2.destroyAllWindows()


# if __name__ == '__main__':
#     face_ex = FacialLandmark()

#     face_ex.run()