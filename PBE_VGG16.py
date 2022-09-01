import cv2
from cv_bridge import CvBridge as cvB
import rosbag
from helper.auxiliary import *
from helper.draw import *

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16

# IMPORTANT for GPU implementation!!!
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import rospy
import argparse
bag_name = "30_Day_2"
path = "bags/"

path = path + bag_name + ".bag"  # Path to read the bag
bag = rosbag.Bag(path)  # Bag object

topic_images = "/dvs/image_raw"
topic_events = "/dvs/events"


def main(args):
    tracking = False
    text = "Searching"
    bridge = cvB()  # To convert image format to cv
    initiated = False  # Variable to initialize the algorithm
    if args.PLOT_DENSITY:
        pd = plotDensity()
    if args.PLOT_PRED:
        pp = plotPred()
    model = VGG16(
    input_shape=(100, 100, 3),  # Shape of our images
    include_top=False,  # Leave out the last fully connected layer
    weights="imagenet",
    )
    t = 0  # Time from the initialization
    cont_events = 0
    time_offset = 0
    cont_saved = 0
    dens = 0
    timestamp_ant = rospy.Time.from_sec(bag.get_start_time())
    # Loop to process all the images
    for images in bag.read_messages(topic_images):
        pred_model = np.array([[0]])
        timestamp = images.timestamp
        cv_image = bridge.imgmsg_to_cv2(images.message)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        im0 = cv_image.copy()  # Image in the original format

        if initiated:
            # Storing the corresponding events (simulating real time)
            Events_arriving = [
                e
                for events in bag.read_messages(
                    topics=topic_events, start_time=timestamp_ant, end_time=timestamp
                )
                for e in events.message.events
            ]
            for e in Events_arriving:
                cont_events += 1
                if args.EVENT_IMAGE:
                    event_image[e.y, e.x] = 255.0
                if args.DRAW_EVENTS:
                    im0[e.y, e.x] = (255, 255, 255)
                t = e.ts.secs + e.ts.nsecs * 10**-9 - offset
                check = bb.addEvent(e)  # Actualizamos la media y la varianza
                if check:
                    dens = ceils.updateEllipsoid(
                        t, bb.axes
                    )
                if args.PLOT_DENSITY:
                    pd.update(t, time_offset, dens)

            # Cropping the image for CNN
            if tracking == False and dens >= DENSITY_LIM and np.all(bb.axes != 0) and np.sum(event_image) != 0:
                if args.EVENT_IMAGE:
                    warped = crop(event_image, bb)
                    x = cv2.resize(
                        warped,
                        (100, 100),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
                else:
                    warped = crop(im0, bb)
                    x = cv2.resize(
                        warped,
                        (100, 100),
                        interpolation=cv2.INTER_NEAREST,
                    )
                x = np.expand_dims(x, axis=0)
                x = x / 255.0  # Necesario!!
                pred_model = model.predict(x)
                if args.PLOT_PRED:
                    pp.update(t, time_offset, pred_model)
        else:
            rows, cols = im0.shape[:2]
            if args.EVENT_IMAGE:
                event_image = np.zeros((rows, cols), dtype="float32")
            bb = BoundingBox(cols, rows)
            ceils = exponentialCluster(im0.shape[1], im0.shape[0])
            # Storing the corresponding events (simulating real time)
            Events_arriving = [
                e
                for events in bag.read_messages(
                    topics=topic_events, start_time=timestamp_ant, end_time=timestamp
                )
                for e in events.message.events
            ]
            offset = timestamp_ant.to_sec()
            t = 0
            initiated = True

        # We show the tracking
        drawEllipse(im0, bb)

        if args.PLOT_DENSITY:
            pd.plot()
        if args.PLOT_PRED:
            pp.plot()

        # Write the state
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im0, text, (int(cols/1.5),int(rows/8)), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("Person", im0)

        if args.SAVE_IMAGE:
            saveImage(im0, pred_model, cont_saved)
            cont_saved += 1

        if args.EVENT_IMAGE:
            cv2.imshow("Event_image", event_image)
            event_image *= 0.0
        cv2.waitKey(1)
        timestamp_ant = timestamp
        dens = ceils.updateTimeEllipsoid(t,bb.axes)
        # Condition to indicate the phase
        if tracking == False and pred_model[0,0] >= 0.5: # Start Tracking
            tracking = True
            text = "Tracking"
        elif tracking == False and t > TIMEOUT and pred_model[0,0] < 0.5: # Reinitialize
            time_offset += t
            initiated = False
        elif tracking == True and dens < DENSITY_LIM_STOP:
            text = "Searching"
            tracking = False
            time_offset += t
            initiated = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--PLOT_DENSITY", action="store_true", required=False)
    parser.add_argument("--SAVE_IMAGE", action="store_true", required=False)
    parser.add_argument(
        "--EVENT_IMAGE", action="store_true", required=False, default=True
    )
    parser.add_argument("--DRAW_EVENTS", action="store_true", required=False)
    parser.add_argument("--PLOT_PRED", action="store_true", required=False)
    args = parser.parse_args()
    main(args)
    print("Main finished")
