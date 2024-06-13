import cv2
import dlib
from imutils import face_utils

# Земи го детекторот за лице од dlib
detector = dlib.get_frontal_face_detector()
# Земи го предикторот за точки на лице од shape_predictor_68_face_landmarks.dat
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def detect_face_landmarks_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for rect in faces:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Прошетај низ обележјата и нацртај ги со cv2.circle
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (95, 250, 110), -1)


        # Прикажи го обработениот кадар
        cv2.imshow('Facial Landmarks Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty('Facial Landmarks Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


