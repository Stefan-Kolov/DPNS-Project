import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Items/shape_predictor_68_face_landmarks.dat')

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
frame_counter = 0


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    res = (A + B) / (2.0 * C)
    return res


def detect_face_video(video_path):
    global frame_counter
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

            if shape.shape[0] != 68:
                print(f"Unexpected number of landmarks: {shape.shape[0]}")
                continue

            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= CONSEC_FRAMES:
                    cv2.putText(frame, "Sleeping!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                frame_counter = 0

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (95, 250, 110), -1)

        cv2.imshow('Facial Landmarks Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty('Facial Landmarks Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
