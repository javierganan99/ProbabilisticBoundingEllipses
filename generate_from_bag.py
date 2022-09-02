from dataset_tools import *
import os

bag_path =  "/your_bag_location/your_bag_name"  # Add here the path to your video!!

output_path = ""

save_images_from_bag(
    bag_path, output_path, stride=30
)  # Saving the frames and event images of the bag (1 in stride)

input_path_images = "frames/"
input_path_events = "eventImages/"
output_path = ""
yolo_crop_events(input_path_images, input_path_events, output_path, show=True, size=(100, 100)) #Cropping persons

output_path = "dataset/"
if not os.path.exists(output_path):
    os.mkdir(output_path)

train_test_split(
    classes_names=["persons", "noPersons"], input_dir="", output_dir=output_path
)
