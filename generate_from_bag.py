from tools import *

# input_path = "/home/grvc/input/croppedPersons/"
# output_path = "/home/grvc/output/"
# yolo_crop_persons(input_path=input_path, output_path=output_path)

# path_output = "/home/grvc/GitHub/wfilter-cnn/dataset/VOC_edges/croppedPersons"
# path_input = "/home/grvc/GitHub/wfilter-cnn/dataset/VOC_cropped/croppedPersons"
path_input = "/home/grvc/GitHub/wfilter-cnn/output/"
path_output = "/home/grvc/GitHub/wfilter-cnn/dataset_frames/"
#save_images_from_video(path_input, path_output, show=False)
#yolo_crop_NO_persons(path_input, path_output, show=False, size=(100, 100))
#images_to_gray(input_path = path_input, output_path = path_output)
train_test_split(classes_names=["persons", "noPersons"], input_dir =path_input, output_dir = path_output)
