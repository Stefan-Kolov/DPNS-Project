import os.path

import image
import video

print(
    "Welcome to face detection tool! You can use this tool to detect specific facial features such as eyes, nose, "
    "mouth in a given image or video, and detecting if the person is sleeping or not in given video.")
user_input = input("Video or Image?\n")

if user_input == "Video":
    video_path = input("Enter the video file path:\n")
    if os.path.exists(video_path):
        print("The path is valid... Video processing...")
        video.detect_face_landmarks_video(video_path)
    else:
        print(f"Error: The file path '{video_path}' does not exist.")

if user_input == "Image":
    image_path = input("Enter the image file path:\n")
    if os.path.exists(image_path):
        print("The path is valid... Image processing...")
        image.detect_faces(image_path)
    else:
        print(f"Error: The file path '{image_path}' does not exist.")

else:
    print("Enter a valid input.")
