import cv2
from cv_bridge import CvBridge as cvB
import rosbag
from helper.auxiliary import *
from helper.draw import *
from yolov5.detect import run as yolo
from yolov5.utils.augmentations import letterbox

bag_name = "30_Day_2"
path = "bags/" # Path to read the bag
path = path + bag_name + ".bag"  # Path to read the bag
bag = rosbag.Bag(path)  # Bag object

topic_images = "/dvs/image_raw"
topic_events = "/dvs/events"

GENERATE_COLORMAP = False

def main():
    bridge = cvB() # To convert image format to cv
    img_size=640 # Size to convert the image to the proper yolo size
    stride = 32 # Parameter for yolo
    flag = False # Variable to initialize the algorithm
    #Event_dataset, Images_dataset = readBag(bag_name, path) # To read the bag
    cont_images = 0 # To count the detections made with yolo
    bb = BoundingBox()
    # Loop to process all the images
    for images in bag.read_messages(topic_images):
        timestamp = images.timestamp
        cv_image = bridge.imgmsg_to_cv2(images.message)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        im0 = cv_image.copy()  # Image in the original format
        cv_image = letterbox(cv_image, img_size, stride=stride)[0].transpose(2,0,1) # Converting image to yolo format

        # Using yolo at the begining for detecting the person
        pred = yolo(im = cv_image, im0=im0)

        if flag:
            # Storing the corresponding events (simulating real time)
            # Storing the corresponding events (simulating real time)
            Events_arriving = [
                e
                for events in bag.read_messages(
                    topics=topic_events, start_time=timestamp_ant, end_time=timestamp
                )
                for e in events.message.events
            ]
            for e in Events_arriving:
                bb.addEvent(e) # Actualizamos la media y la varianza

            #Colormap
            if GENERATE_COLORMAP:
                imProb = np.zeros(im0.shape[0:2], dtype=np.uint8)
                pmax = bb.getPixelProb((bb.mean[0],bb.mean[1]))
                for y in range(im0.shape[0]):
                    for x in range(im0.shape[1]):
                        p = bb.getPixelProb((x,y)) / pmax
                        imProb[y][x] = int(255*p)
                imColorMap = cv2.applyColorMap(imProb, cv2.COLORMAP_JET)
                cv2.imwrite("output/mapa/" + str(cont_images).zfill(5) + ".png", imColorMap)
                # imColorMap = cv2.addWeighted(im0, 0.5, imColorMap, 0.5, 10)
                # cv2.imwrite("output/mapa_superpuesto/" + str(cont_images).zfill(5) + ".png", imColorMap)
                cv2.imshow("colormap", imColorMap)

            cv2.imwrite("output/normal/" + str(cont_images).zfill(5) + ".png", im0)
            # # We show the tracking
            drawEllipse(im0, bb)            
            cv2.imwrite("output/normal_elipses/" + str(cont_images).zfill(5) + ".png", im0)
            if pred:
                im0 = cv2.rectangle(im0, (int(pred[0] - pred[2]/2), int(pred[1] - pred[3]/2)), (int(pred[0] + pred[2]/2), int(pred[1] + pred[3]/2)), (0, 255, 0), 2)
        if pred: # We show the YoloPrediction
            im0 = cv2.rectangle(im0, (int(pred[0] - pred[2]/2), int(pred[1] - pred[3]/2)), (int(pred[0] + pred[2]/2), int(pred[1] + pred[3]/2)), (0, 255, 0), 2)
            cont_images += 1
        cv2.imshow("Person",im0)
        cv2.waitKey(1)
        flag = True
        timestamp_ant = timestamp

if __name__ == "__main__":
    main()
    print("Main finished")
