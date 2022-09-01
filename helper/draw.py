import math
import scipy.stats as st
import cv2
import matplotlib.pyplot as plt

from constant import DENSITY_LIM


class plotDensity:  # Class to dimanically plot the spatio-temporal density
    def __init__(self):
        self.store_time = []
        self.fig2, self.ax2 = plt.subplots(1)
        self.line2 = self.ax2.plot([], [], "b", label="std")[0]
        self.line22 = self.ax2.plot([], [], "r", label="Density_lim")[0]
        self.store_density = []

    def update(self, t, time_offset, dens):
        self.store_time.append(t + time_offset)
        self.store_density.append(dens)

    def plot(self):
        self.line2.set_data(self.store_time, self.store_density)
        self.line22.set_data(self.store_time, [DENSITY_LIM] * len(self.store_time))
        self.ax2.legend()
        self.ax2.set_xlabel("Time [s]")
        self.ax2.set_ylabel("Value")
        self.ax2.autoscale_view(True)
        self.ax2.relim()
        self.fig2.canvas.draw()
        plt.pause(0.00001)


class plotPred:  # Class to dinamically plot the prediction of the CNN
    def __init__(self):
        self.time_images = []
        self.fig3, self.ax3 = plt.subplots(1)
        self.line3 = self.ax3.plot([], [], "g", label="Prediction")[0]
        self.line32 = self.ax3.plot([], [], "r", label="Prob_lim")[0]
        self.store_pred = []

    def update(self, t, time_offset, pred_model):
        self.store_pred.append(pred_model[0, 0])
        self.time_images.append(t + time_offset)

    def plot(self):
        self.line3.set_data(self.time_images, self.store_pred)
        self.line32.set_data(self.time_images, [0.5] * len(self.time_images))
        self.ax3.legend()
        self.ax3.set_xlabel("Time [s]")
        self.ax3.set_ylabel("Prediction")
        self.ax3.autoscale_view(True)
        self.ax3.relim()
        self.fig3.canvas.draw()
        plt.pause(0.00001)


prob999 = st.chi2.isf(1 - 0.999, 2)  # 99.9% of pprobability
prob99 = st.chi2.isf(1 - 0.99, 2)  # 99% of pprobability
prob95 = st.chi2.isf(1 - 0.95, 2)  # 95% of pprobability
prob85 = st.chi2.isf(1 - 0.85, 2)  # 85% of pprobability
prob70 = st.chi2.isf(1 - 0.70, 2)  # 75% of pprobability
prob50 = st.chi2.isf(1 - 0.50, 2)  # 75% of pprobability


def drawEllipse(im0, bb):  # Function to draw confidence ellipsoid during tracking

    thickness = 1
    axes999 = tuple([int(i * math.sqrt(prob999) / math.sqrt(bb.prob)) for i in bb.axes])
    axes99 = tuple([int(i * math.sqrt(prob99) / math.sqrt(bb.prob)) for i in bb.axes])
    axes95 = tuple([int(i * math.sqrt(prob95) / math.sqrt(bb.prob)) for i in bb.axes])
    axes85 = tuple([int(i * math.sqrt(prob85) / math.sqrt(bb.prob)) for i in bb.axes])
    axes70 = tuple([int(i * math.sqrt(prob70) / math.sqrt(bb.prob)) for i in bb.axes])
    axes50 = tuple([int(i * math.sqrt(prob50) / math.sqrt(bb.prob)) for i in bb.axes])
    im0 = cv2.ellipse(
        im0,
        (int(bb.mean[0]), int(bb.mean[1])),
        axes999,
        bb.alpha * 180 / math.pi,
        0,
        360,
        (0, 0, 255),
        thickness,
    )
    im0 = cv2.ellipse(
        im0,
        (int(bb.mean[0]), int(bb.mean[1])),
        axes99,
        bb.alpha * 180 / math.pi,
        0,
        360,
        (0, 155, 255),
        thickness,
    )
    im0 = cv2.ellipse(
        im0,
        (int(bb.mean[0]), int(bb.mean[1])),
        axes95,
        bb.alpha * 180 / math.pi,
        0,
        360,
        (0, 243, 255),
        thickness,
    )
    im0 = cv2.ellipse(
        im0,
        (int(bb.mean[0]), int(bb.mean[1])),
        axes85,
        bb.alpha * 180 / math.pi,
        0,
        360,
        (0, 255, 185),
        thickness,
    )
    im0 = cv2.ellipse(
        im0,
        (int(bb.mean[0]), int(bb.mean[1])),
        axes70,
        bb.alpha * 180 / math.pi,
        0,
        360,
        (255, 212, 0),
        thickness,
    )
    im0 = cv2.ellipse(
        im0,
        (int(bb.mean[0]), int(bb.mean[1])),
        axes50,
        bb.alpha * 180 / math.pi,
        0,
        360,
        (255, 0, 158),
        thickness,
    )


def saveImage(
    im0, pred_model, cont_saved
):  # Function to save the images depending on the CNN prediction
    # It store the results in a folder named output that must previously exist
    positive = "PERSON"
    negative = "NOPERSON"

    if pred_model[0, 0] > 0.5:
        prediction = positive
        cv2.imwrite(
            "output/YES/" + str(cont_saved).zfill(5) + prediction + ".png",
            im0,
        )
    else:
        prediction = negative
        cv2.imwrite(
            "output/NO/" + str(cont_saved).zfill(5) + prediction + ".png",
            im0,
        )
