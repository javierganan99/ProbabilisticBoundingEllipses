from tools import *
import os

video_path = "/your_video_location/your_video"  # Add here the path to your video!!

output_path = "frames/"
if not os.path.exists(output_path):
    os.mkdir(output_path)

save_images_from_video(
    video_path, output_path, stride=30
)  # Saving the frames of the video (1 in 30)

input_path = "frames/"
output_path = ""
yolo_crop(input_path, output_path, show=True, size=(100, 100)) #Cropping persons

input_path = "RGBpersons/"
output_path = "persons/"
if not os.path.exists(output_path):
    os.mkdir(output_path)
images_to_gray(input_path, output_path)

input_path = "RGBnoPersons/"
output_path = "noPersons/"
if not os.path.exists(output_path):
    os.mkdir(output_path)
images_to_gray(input_path, output_path)

output_path = "dataset/"
if not os.path.exists(output_path):
    os.mkdir(output_path)
train_test_split(
    classes_names=["persons", "noPersons"], input_dir="", output_dir=output_path
)
