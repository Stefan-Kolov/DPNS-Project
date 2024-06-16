import cv2
import dlib
from imutils import face_utils


def detect_face_image(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("Items/shape_predictor_68_face_landmarks.dat")

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 1)

    for (i, rect) in enumerate(faces):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

    for (x, y) in shape:
        cv2.circle(image, (x, y), 3, (95, 250, 110), -1)

    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
