import numpy as np # type: ignore
import cv2 # type: ignore
import mediapipe as mp # type: ignore
import time

# Initialize MediaPipe Face Mesh and Drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=1)

def initialize_face_mesh(min_detection_confidence=0.5, min_tracking_confidence=0.5):
    return mp_face_mesh.FaceMesh(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

def process_frame(image, face_mesh):
    # Convert the image color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_landmarks(face_landmarks, img_w, img_h):
    face_2d = []
    face_3d = []
    nose_2d = None
    nose_3d = None

    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in {33, 263, 1, 61, 291, 199}:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            if idx == 1:
                nose_2d = (lm.x * img_w, lm.y * img_h)
                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

    return np.array(face_2d, dtype=np.float64), np.array(face_3d, dtype=np.float64), nose_2d, nose_3d

def calculate_head_pose(face_2d, face_3d, img_w, img_h):
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])
    distortion_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
    
    rmat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    
    return x, y, z, rotation_vec, translation_vec, cam_matrix, distortion_matrix

def determine_head_pose_direction(x, y):
    if y < -10:
        return "Looking Left"
    elif y > 10:
        return "Looking Right"
    elif x < -10:
        return "Looking Down"
    elif x > 10:
        return "Looking Up"
    else:
        return "Forward"

def draw_annotations(image, nose_2d, nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix, x, y, z, text, fps):
    nose_3d_projection, _ = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix)
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
    
    cv2.line(image, p1, p2, (255, 0, 0), 3)
    cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    return image

def main():
    cap = cv2.VideoCapture(0)
    face_mesh = initialize_face_mesh()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        start = time.time()
        image, results = process_frame(image, face_mesh)
        img_h, img_w, img_c = image.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_2d, face_3d, nose_2d, nose_3d = extract_landmarks(face_landmarks, img_w, img_h)
                x, y, z, rotation_vec, translation_vec, cam_matrix, distortion_matrix = calculate_head_pose(face_2d, face_3d, img_w, img_h)
                text = determine_head_pose_direction(x, y)
                end = time.time()
                fps = 1 / (end - start)
                image = draw_annotations(image, nose_2d, nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix, x, y, z, text, fps)

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )

        cv2.imshow('Head Pose Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
