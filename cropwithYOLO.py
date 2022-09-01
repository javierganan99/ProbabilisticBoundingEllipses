import cv2
from cv_bridge import CvBridge as cvB
import rosbag
import rospy
from yolov5.detect_multiple import run as yolo
from yolov5.utils.augmentations import letterbox
import numpy as np

bag_name = "outdoors_running"
path = "bags/"

path = path + bag_name + ".bag" # Path to read the bag
bag = rosbag.Bag(path)          # Bag object

output_path = "dataset/"
persons_folder = "PERSONS/"
NO_persons_folder = "NOPERSONS/"


def main():
    cont_images = 0
    img_size=640 # Size to convert the image to the proper yolo size
    stride = 32 # Parameter for yolo
    bridge = cvB()  # To convert image format to cv
    show = True
    timestamp_ant = rospy.Time.from_sec(bag.get_start_time())
    cont_images_persons_saved = 48
    cont_images_NO_persons_saved = 1568
    rows,cols = 180,240
    event_image = np.zeros((rows, cols), dtype="float32")
    for images in bag.read_messages("/dvs/image_raw"):
        timestamp = images.timestamp

        if cont_images % 2 != 0:
            cont_images += 1
            timestamp_ant = timestamp
            continue
        else:
            event_image = np.zeros((rows, cols), dtype="float32")
        
        cv_image = bridge.imgmsg_to_cv2(images.message)
        cv_image = cv2.cvtColor(cv_image,cv2.COLOR_GRAY2RGB)
        im0 = cv_image.copy() # Image in the original format
        print(im0.shape)
        cv_image = letterbox(cv_image, img_size, stride=stride)[0].transpose(2,0,1) # Converting image to yolo format
        # Storing the corresponding events (simulating real time)
        for events in bag.read_messages(topics="/dvs/events", start_time=timestamp_ant, end_time=timestamp):
            for e in events.message.events:
                event_image[e.y, e.x] = 255.0
        if event_image.sum() / 255.0 > 0.001 * rows * cols:
            cv2.imshow("Events", event_image)
        else:
            cont_images += 1
            timestamp_ant = timestamp
            continue
        pred = yolo(im = cv_image, im0=im0)
        for p in pred:
            bb = p[1]
            cropped = event_image[
                int(bb[1] - bb[3] / 2) : int(bb[1] + bb[3] / 2), int(bb[0] - bb[2] / 2) : int(bb[0] + bb[2] / 2)
            ]
            if p[0] == 0:
                # if show:
                #     if cropped.sum() / 255.0 > 0.005 * bb[2] * bb[3] and cropped.sum() / 255.0 < 0.8 * bb[2] * bb[3]:
                #         cropped = cv2.resize(cropped, (100,100), interpolation = cv2.INTER_NEAREST)
                #         cv2.imwrite(output_path + persons_folder + str(cont_images_persons_saved).zfill(7) + ".png", cropped)
                #         im0 = cv2.rectangle(
                #             im0,
                #             (int(bb[0] - bb[2] / 2), int(bb[1] - bb[3] / 2)),
                #             (int(bb[0] + bb[2] / 2), int(bb[1] + bb[3] / 2)),
                #             (0, 255, 0),
                #             2,
                #         )
                #         cont_images_persons_saved += 1
                pass
                
            else:
                if show:
                    if cropped.sum() / 255.0 > 0.03 * bb[2] * bb[3] and cropped.sum() / 255.0 < 0.8 * bb[2] * bb[3]:
                        cropped = cv2.resize(cropped, (100,100), interpolation = cv2.INTER_NEAREST)
                        cv2.imwrite(output_path + NO_persons_folder + str(cont_images_NO_persons_saved).zfill(7) + ".png", cropped)
                        im0 = cv2.rectangle(
                            im0,
                            (int(bb[0] - bb[2] / 2), int(bb[1] - bb[3] / 2)),
                            (int(bb[0] + bb[2] / 2), int(bb[1] + bb[3] / 2)),
                            (0, 0, 255),
                            2,
                        )
                        cont_images_NO_persons_saved += 1
        if not pred:
            cropped = cv2.resize(event_image, (100,100), interpolation = cv2.INTER_NEAREST)
            cv2.imwrite(output_path + NO_persons_folder + str(cont_images_NO_persons_saved).zfill(7) + ".png", cropped)
            cont_images_NO_persons_saved += 1


        cont_images += 1
        if show:
            cv2.imshow("Person", im0)
        cv2.waitKey(1)
        timestamp_ant = timestamp

        


if __name__ == "__main__":
    main()
    print("Main finished")
