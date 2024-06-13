import cv2
import dlib
import imutils
from imutils import face_utils


def detect_faces(image_path):

    # Иницијализирај детектор за лице (HOG-базиран) и предиктор за обележја
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Учитај слика и претвори ја во grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Детектирај лица во grayscale сликата
    faces = detector(gray, 1)

    # Прошетај низ детектираните лица
    for (i, rect) in enumerate(faces):
        # Добиј координати на обележја на лицето
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

    # Прошетај низ обележјата и нацртај ги
    for (x, y) in shape:
        cv2.circle(image, (x, y), 3, (95, 250, 110), -1)

    # Прикажи ја излезната слика со нацртани обележја
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

