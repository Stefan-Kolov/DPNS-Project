import os.path
import sys
import image
import video

print(
    "Welcome to face detection tool! You can use this tool to detect specific facial features such as eyes, nose, "
    "mouth in a given image or video, and detecting if the person is sleeping or not in given video.")
user_input = input("Video or Image?\n")

if user_input == "Video":
    video_path = input("Enter the video file path (Enter / to use the project default video!):\n")
    if video_path == "/":
        print("Using default video... Video processing...")
        video.detect_face_video("Items/default.mp4")
        sys.exit()
    if os.path.exists(video_path):
        print("The path is valid... Video processing...")
        video.detect_face_video(video_path)
        sys.exit()
    else:
        print(f"Error: The file path '{video_path}' does not exist.")

if user_input == "Image":
    image_path = input("Enter the image file path (Enter / to use the project default image!):\n")
    if image_path == "/":
        print("Using default image... Image processing...")
        image.detect_face_image("Items/default.jpg")
        sys.exit()
    if os.path.exists(image_path):
        print("The path is valid... Image processing...")
        image.detect_face_image(image_path)
        sys.exit()
    else:
        print(f"Error: The file path '{image_path}' does not exist.")

else:
    print("Enter a valid input.")
