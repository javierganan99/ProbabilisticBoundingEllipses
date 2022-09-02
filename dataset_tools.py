import cv2
from cv_bridge import CvBridge as cvB
import rosbag
import numpy as np
import os
import random
import shutil
import sys


def salt_pepper(img):
    """This function adds salt and
    pepper noise to images
    """

    row, col = img.shape
    number_of_pixels = random.randint(int(row * col / 200), int(row * col / 100))
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 255
    for i in range(row):
        for j in range(col):
            if img[i][j] == 255 and random.uniform(0, 1) < 0.25:
                img[i][j] = 0
    return img


class events_image_simulated:
    """This class is used to generate simulated
    event images from an image dataset
    """

    # vert = range(400, 50, -50)
    # hor = range(400, 50, -50)
    i = 200
    j = 200

    def __init__(self):
        self.salt_pepper = salt_pepper

    def generate_events(self, im):
        self.im = cv2.Canny(im, self.i, self.j)  # self.vert[self.i], self.hor[self.j])
        # if self.i >= len(self.vert) - 1:
        #     self.i = 0
        #     self.j += 1
        # else:
        #     self.i += 1
        # if self.j >= len(self.hor) - 1 and self.i >= len(self.vert) - 1:
        #     self.i = 0
        #     self.j = 0
        self.im = self.salt_pepper(self.im)
        return self.im


def save_images_from_bag(input_path, output_path, show=False):
    """This function extract images from "imput_path" .bag file
    and write them in "output_path" folder
    """

    bridge = cvB()  # To convert image format to cv
    bag = rosbag.Bag(input_path)  # Bag object
    # Storing the images
    Images_dataset = [images for images in bag.read_messages("/dvs/image_raw")]
    cont_images = 0  # To name the images
    for images in Images_dataset:
        cv_image = bridge.imgmsg_to_cv2(images.message)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        if cont_images % 20 == 0:
            cv2.imwrite(output_path + str(cont_images).zfill(5) + ".png", cv_image)
        cont_images += 1
        if show:
            cv2.imshow("Person", cv_image)
            cv2.waitKey(1)


def save_images_from_video(input_path, output_path, stride=30):
    """This function extract images from "imput_path" video file
    and write them in "output_path" folder
    """
    vidcap = cv2.VideoCapture(input_path)
    success, image = vidcap.read()
    cont_images = 0
    cont_saved = 0
    while success:
        if cont_images % stride == 0:
            cv2.imwrite(output_path + str(cont_saved).zfill(5) + ".png", image)
            cont_saved += 1
        success, image = vidcap.read()
        cont_images += 1


def images_to_gray(input_path, output_path, show=False, subsample=None):
    """This function extract images from "imput_path" folder,
    convert them to grayscale and write them in "output_path" folder
    """
    # Storing the images
    Images = [f for f in os.listdir(input_path) if f.endswith(".png")]
    if subsample is not None:
        Images = random.sample(Images, subsample)
    for image in Images:
        im = cv2.imread(str(input_path) + "/" + image)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = np.stack((gray,) * 3, axis=-1)
        cv2.imwrite(output_path + "/" + image, gray)
        if show:
            cv2.imshow("Person", gray)
            cv2.waitKey(1)


def images_to_edges(input_path, output_path, show=False, subsample=None):
    """This function extract images from "imput_path" folder,
    convert them to simulated event images (using edges) and
    write them in "output_path" folder
    """
    ev = events_image_simulated()
    # Storing the images
    Images = [f for f in os.listdir(input_path) if f.endswith(".png")]
    if subsample is not None:
        Images = random.sample(Images, subsample)
    for image in Images:
        im = cv2.imread(str(input_path) + "/" + image)
        edges = ev.generate_events(im)
        cv2.imwrite(output_path + "/" + image, edges)
        if show:
            cv2.imshow("Person", edges)
            cv2.waitKey(1)


def train_test_split(classes_names, input_dir, output_dir):
    """This function copy the images from "input_dir/classes_names",
    divide them into train, test, and val, and save them into
    "output_dir" folder in the corresponding classes' folders
    """
    output_folders = ["train", "val", "test"]

    # Create (if not crated before) the output folders
    for f in output_folders:
        for c in classes_names:
            path = output_dir + "/" + f + "/" + c
            if not os.path.exists(path):
                os.makedirs(path)

    # Creating partitions of the data after shuffeling
    input_folders = [input_dir + c for c in classes_names]
    files_names = [os.listdir(f) for f in input_folders]

    # Saving the names of the files for all the classes
    files_split = []
    for files in files_names:
        np.random.shuffle(files)
        files_split.append(
            np.split(np.array(files), [int(len(files) * 0.8), int(len(files) * 0.9)])
        )

    # Saving the paths of the images to copy/paste
    files_to_write = []
    for i, input_f in enumerate(input_folders):
        classes = []
        for fs in files_split[i]:
            classes.append([input_f + "/" + name for name in fs.tolist()])
        files_to_write.append(classes)

    # Copy-pasting images
    for c, fw in zip(classes_names, files_to_write):
        for of, f in zip(output_folders, fw):
            for name_p in f:
                shutil.copy(name_p, output_dir + "/" + of + "/" + c)
    return 0


def yolo_crop(input_path, output_path, show=False, size=(100, 100), bb_aug=0):
    """This function extract images from "imput_path" folder,
    apply YOLO to detect persons, crop them, resize to indicated size,
    and save the cropped persons into "output_path" folder.
    """
    p = os.path.abspath(".")
    sys.path.insert(1, p)
    from yolov5.detect_multiple import run as yolo
    from yolov5.utils.augmentations import letterbox

    img_size = 640  # Size to convert the image to the proper yolo size
    stride = 32  # Parameter for yolo
    cont_images_persons = 0  # To count the persons detected by yolo
    cont_images_NO_persons = 0  # To count the non-persons detected by yolo
    Images = [f for f in os.listdir(input_path) if f.endswith(".png")]
    rows, cols = cv2.imread(input_path + Images[0]).shape[:2]

    path = output_path + "RGBpersons/"
    if not os.path.exists(path):
        os.mkdir(path)
    path = output_path + "RGBnoPersons"

    if not os.path.exists(path):
        os.mkdir(path)
    for image in Images:
        cv_image = cv2.imread(input_path + image)
        if len(cv_image.shape) < 3:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        im0 = cv_image.copy()  # Image in the original format
        if show:
            im2 = im0.copy()
        cv_image = letterbox(cv_image, img_size, stride=stride)[0].transpose(
            2, 0, 1
        )  # Converting image to yolo format
        # Using yolo for detecting the person
        pred = yolo(im=cv_image, im0=im0)
        af = 0.5 * (
            1 + random.randint(0, bb_aug) * 0.1
        )  # Augmentation factor for half width & height
        for p in pred:
            if p[0] == 0:
                bb = p[1]
                cropped = im0[
                    np.clip(int(bb[1] - bb[3] * af), 0, rows) : np.clip(
                        int(bb[1] + bb[3] * af), 0, cols
                    ),
                    np.clip(int(bb[0] - bb[2] * af), 0, rows) : np.clip(
                        int(bb[0] + bb[2] * af), 0, cols
                    ),
                ]
                x = cv2.resize(
                    cropped,
                    size,
                    interpolation=cv2.INTER_NEAREST,
                )
                cv2.imwrite(
                    output_path
                    + "RGBpersons/"
                    + str(cont_images_persons).zfill(5)
                    + ".png",
                    x,
                )
                cont_images_persons += 1
                if show:
                    im2 = cv2.rectangle(
                        im2,
                        (int(bb[0] - bb[2] / 2), int(bb[1] - bb[3] / 2)),
                        (int(bb[0] + bb[2] / 2), int(bb[1] + bb[3] / 2)),
                        (0, 255, 0),
                        2,
                    )
            else:
                bb = p[1]
                cropped = im0[
                    np.clip(int(bb[1] - bb[3] * af), 0, rows) : np.clip(
                        int(bb[1] + bb[3] * af), 0, cols
                    ),
                    np.clip(int(bb[0] - bb[2] * af), 0, rows) : np.clip(
                        int(bb[0] + bb[2] * af), 0, cols
                    ),
                ]
                x = cv2.resize(
                    cropped,
                    size,
                    interpolation=cv2.INTER_NEAREST,
                )
                cv2.imwrite(
                    output_path
                    + "RGBnoPersons/"
                    + str(cont_images_NO_persons).zfill(5)
                    + ".png",
                    x,
                )
                cont_images_NO_persons += 1
                if show:
                    im2 = cv2.rectangle(
                        im2,
                        (int(bb[0] - bb[2] / 2), int(bb[1] - bb[3] / 2)),
                        (int(bb[0] + bb[2] / 2), int(bb[1] + bb[3] / 2)),
                        (0, 0, 255),
                        2,
                    )

        if show:
            cv2.imshow("Person", im2)
            cv2.waitKey(1)


def crop_from_label(path_images, path_anotations, path_output, size=(100, 100)):
    """This function extract images from "path_images" folder,
    check the "path_anotations" folder to crop the classes, resize
    to indicated size, and save the cropped "persons" and "non-persons"
    into "path_output" folder
    """
    import xml.etree.ElementTree as ET
    import cv2
    import os

    cont_images_persons = 0
    cont_images_no_persons = 0
    # We store the name of the images and the anotations
    images = [
        f for f in os.listdir(path_images) if f.endswith(".png") or f.endswith(".jpg")
    ]
    images.sort()
    anotations = [f for f in os.listdir(path_anotations) if f.endswith(".xml")]
    anotations.sort()
    # Creating the folders
    if not os.path.exists(path_output + "/" + "croppedPersons"):
        os.makedirs(path_output + "/" + "croppedPersons")
    if not os.path.exists(path_output + "/" + "croppedNoPersons"):
        os.makedirs(path_output + "/" + "croppedNoPersons")

    for im, an in zip(images, anotations):
        # Path to read the image
        path_image = path_images + "/" + im
        # Using cv2.imread() method
        img = cv2.imread(path_image)
        # parse xml file
        path_anot = path_anotations + "/" + an
        tree = ET.parse(path_anot)
        root = tree.getroot()  # get root object

        bbox_coordinates = []
        for member in root.findall("object"):
            try:
                class_name = member[0].text  # class name
                # bbox coordinates
                xmin = int(member[4][0].text)
                ymin = int(member[4][1].text)
                xmax = int(member[4][2].text)
                ymax = int(member[4][3].text)
                # store data in list
                bbox_coordinates.append([class_name, xmin, ymin, xmax, ymax])
                # Saving the images
                for bb in bbox_coordinates:
                    try:
                        if bb[0] == "person":
                            cropped = img[bb[2] : bb[4], bb[1] : bb[3]]
                            x = cv2.resize(
                                cropped,
                                size,
                                interpolation=cv2.INTER_NEAREST,
                            )
                            cv2.imwrite(
                                path_output
                                + "/"
                                + "croppedPersons/"
                                + str(cont_images_persons).zfill(5)
                                + ".png",
                                x,
                            )
                            cont_images_persons += 1
                        else:
                            cropped = img[bb[2] : bb[4], bb[1] : bb[3]]
                            x = cv2.resize(
                                cropped,
                                size,
                                interpolation=cv2.INTER_NEAREST,
                            )
                            cv2.imwrite(
                                path_output
                                + "/"
                                + "croppedNoPersons/"
                                + str(cont_images_no_persons).zfill(5)
                                + ".png",
                                x,
                            )
                            cont_images_no_persons += 1
                    except:
                        pass
            except:
                continue
