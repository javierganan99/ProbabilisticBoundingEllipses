import cv2
from cv_bridge import CvBridge as cvB
import rosbag
from helper.auxiliary import *
from helper.yolov5_detection import *
from helper.draw import *
import tensorflow as tf
import pickle

# IMPORTANT for GPU implementation!!!
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
import rospy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

bag_name = "3"
path = "bags/FarDistance/"

path = path + bag_name + ".bag"  # Path to read the bag
bag = rosbag.Bag(path)  # Bag object

prob99 = st.chi2.isf(1 - 0.99, 2)  # 99% of pprobability

topic_images = "/dvs/image_raw"
topic_events = "/dvs/events"

positive = "PERSON"
negative = "NOPERSON"


def main():
    store_pred = [] # Store YOLO's predicted center coordinates
    time_YOLO = [] # Store YOLO's timestamps 
    store_bb = [] # Store our method predicted center coordinates
    time_bb = [] # Store BB's timestamps
    distances = [] # Store the distances (in pixels) between YOLO's and our's predictions
    bridge = cvB()  # To convert image format to cv
    initiated = False  # Variable to initialize the algorithm
    thickness = 2  # Thickness of the drawn ellipses
    t = 0  # Time from the initialization
    cont_events = 0
    time_offset = 0
    yolo = YoloPred()
    timestamp_ant = rospy.Time.from_sec(bag.get_start_time())
    dens = 0  # Density of the bounding box
    # Loop to process all the images
    for images in bag.read_messages(topic_images):
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
                t = e.ts.secs + e.ts.nsecs * 10**-9 - offset

                check = bb.addEvent(e)  # Actualizamos la media y la varianza
                if check:
                    dens = ceils.updateEllipsoid(t, bb.axes)

                else:
                    rows, cols = im0.shape[:2]
        else:
            rows, cols = im0.shape[:2]
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
        pred = yolo.predict(im0)
        person = False
        for p in pred:
            boundbox = p[1]
            if p[0] == 0:
                person = True
                im0 = cv2.rectangle(
                    im0,
                    (
                        int(boundbox[0] - boundbox[2] / 2),
                        int(boundbox[1] - boundbox[3] / 2),
                    ),
                    (
                        int(boundbox[0] + boundbox[2] / 2),
                        int(boundbox[1] + boundbox[3] / 2),
                    ),
                    (0, 255, 0),
                    2,
                )
                store_pred.append(boundbox[0:2])
        # We show the tracking
        axes99 = tuple(
            [int(i * math.sqrt(prob99) / math.sqrt(bb.prob)) for i in bb.axes]
        )
        im0 = cv2.ellipse(
            im0,
            (int(bb.mean[0]), int(bb.mean[1])),
            axes99,
            bb.alpha * 180 / math.pi,
            0,
            360,
            (0, 0, 255),
            thickness,
        )
        store_bb.append([bb.mean[0], bb.mean[1]])
        time_bb.append(t + time_offset)
        if person:
            distances.append(
                math.sqrt(
                    (bb.mean[0] - store_pred[-1][0]) ** 2
                    + (bb.mean[1] - store_pred[-1][1]) ** 2
                )
            )
            time_YOLO.append(t + time_offset)
        cv2.imshow("Persons", im0)
        cv2.waitKey(1)
        timestamp_ant = timestamp
        dens = ceils.updateTimeEllipsoid(t,bb.axes)
        # Condition to reinitializate
        if t > TIMEOUT and dens < DENSITY_LIM_STOP:
            time_offset += t
            initiated = False
    store_bb = np.array(store_bb)
    store_pred = np.array(store_pred)
    plt.plot(time_YOLO, distances, c="green")
    # To save the plot
    plt.xlabel("Time [s]", fontsize = 17)
    plt.ylabel("Error [px]", fontsize = 17)
    plt.grid()
    plt.show()
    # To save the data
    a = {"time_YOLO": time_YOLO, "distances": distances}
    with open("data.pickle", "wb") as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
    print("Main finished")
