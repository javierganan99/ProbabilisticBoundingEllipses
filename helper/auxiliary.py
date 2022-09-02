import numpy as np
import scipy.stats as st
from parameters import *
import math
import cv2
import rosbag

class BoundingBox:
    
    def __init__(self, w = None, h = None):
        self.image_width = w
        self.image_height = h
        self.mean = np.array([5.0, 5.0])
        self.covariance = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.events_person = []
        self.sq_sum = np.array([0, 0])
        self.cross_mean = 0
        self.prob = st.chi2.isf(1 - 0.99, 2)  # 99% of pprobability
        self.axes = np.array([0,0]) # Seminaxes of the ellipsoid
        self.alpha = 0 # Orientation of the ellipsoid

    def getPixelProb(self, xy):  # Probability of a certain pixel of the image
        return st.multivariate_normal.pdf(xy, mean=self.mean, cov=self.covariance)

    def __get_cov_ellipsoid(self, cov):
        assert cov.shape == (2, 2)
        # Find and sort eigenvalues corresponding to the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.sum(cov, axis=0).argsort()
        eigvals_temp = eigvals[idx]
        idx = eigvals_temp.argsort()
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        return eigvals, eigvecs

    def addEvent(self, e):

        if len(self.events_person) >= BUFFER_SIZE:
            # Check if inside ellipse
            cos_angle = np.cos(self.alpha)
            sin_angle = np.sin(self.alpha)
            xc = e.x - self.mean[0]
            yc = e.y - self.mean[1]
            xct = xc * cos_angle + yc * sin_angle
            yct = xc * sin_angle - yc * cos_angle
            rad_cc = (xct**2 / (self.axes[0]) ** 2) + (yct**2 / (self.axes[1]) ** 2)
            if rad_cc > 1:  # If point out of the ellipse
                return False
            # Update mean
            self.mean[0] = (
                self.mean[0] - self.events_person[0].x / BUFFER_SIZE + e.x / BUFFER_SIZE
            )
            self.mean[1] = (
                self.mean[1] - self.events_person[0].y / BUFFER_SIZE + e.y / BUFFER_SIZE
            )
            self.sq_sum[0] = self.sq_sum[0] + e.x**2 - self.events_person[0].x ** 2
            self.sq_sum[1] = self.sq_sum[1] + e.y**2 - self.events_person[0].y ** 2
            self.cross_mean = (
                self.cross_mean
                + e.x * e.y
                - self.events_person[0].x * self.events_person[0].y
            )
            # Update standard deviation
            self.covariance[0, 0] = (
                - self.mean[0] ** 2 + self.sq_sum[0] / BUFFER_SIZE
            )
            self.covariance[1, 1] = (
                - self.mean[1] ** 2 + self.sq_sum[1] / BUFFER_SIZE
            )
            self.covariance[0, 1] = self.covariance[1, 0] = (
                self.cross_mean / BUFFER_SIZE - self.mean[0] * self.mean[1]
            )
            # Update buffer
            self.events_person.append(e)
            self.events_person = self.events_person[1:]
            # Compute ellipsoid
            eigvals, eigvects = self.__get_cov_ellipsoid(self.covariance)
            AxisMayor = math.sqrt(self.prob * eigvals[1])
            if AxisMayor < MIN_TAM:
                AxisMayor = MIN_TAM
            AxisMinor = math.sqrt(self.prob * eigvals[0])
            if AxisMinor < MIN_TAM:
                AxisMinor = MIN_TAM
            self.alpha = math.atan2(eigvects[1][1], eigvects[1][0])
            self.axes = np.array([int(AxisMayor), int(AxisMinor)])
            return True
        else:
            # Compute initial mean
            self.mean[0] = (self.mean[0] * len(self.events_person) + e.x) / (
                len(self.events_person) + 1
            )
            self.mean[1] = (self.mean[1] * len(self.events_person) + e.y) / (
                len(self.events_person) + 1
            )
            # Compute initial sd
            self.sq_sum[0] += e.x**2
            self.sq_sum[1] += e.y**2
            self.cross_mean += e.x * e.y
            # Update buffer
            self.events_person.append(e)
            # Update standard deviation
            N = len(self.events_person)
            self.covariance[0, 0] = (
                self.mean[0] ** 2 + self.sq_sum[0] / N - self.mean[0] ** 2 * 2
            )
            self.covariance[1, 1] = (
                self.mean[1] ** 2 + self.sq_sum[1] / N - self.mean[1] ** 2 * 2
            )
            self.covariance[0, 1] = self.covariance[1, 0] = (
                self.cross_mean / N - self.mean[0] * self.mean[1]
            )
            if len(self.events_person) == BUFFER_SIZE - 1:
                # Compute first ellipsoid
                eigvals, eigvects = self.__get_cov_ellipsoid(self.covariance)
                AxisMayor = math.sqrt(self.prob * eigvals[1])
                AxisMinor = math.sqrt(self.prob * eigvals[0])
                self.axes = np.array([int(AxisMayor), int(AxisMinor)])
                self.alpha = math.atan2(eigvects[1][1], eigvects[1][0])
            return True


class exponentialCluster:
    """
    This class is in charge of calculate the spatio-temporal density.
    It can calculate the density of an ellipsoid bounding box with updateEllipsoid
    method, but also the spatiotemporal density of each cell of the image.
    """

    def __init__(self, width, height):
        self.w = 5  # Number of horizontal divisions for cell calulation
        self.h = 5  # Number of vertical divisions for cell calulation
        self.t_ant = 0  # Previous time instant
        self.width, self.height = width, height  # Image dimensions
        self.cells = np.zeros((self.h, self.w))
        self.cellsMean = np.zeros((self.h, self.w))
        self.areas = np.zeros((self.h, self.w))
        self.cellArea = (self.width / self.w) * (self.height / self.h)
        self.densityBB = 0
    
    def updateTimeEllipsoid(self, t, axes):
        self.densityBB *= math.exp((-t + self.t_ant) * TAU)
        self.t_ant = t
        return self.densityBB / (math.pi * axes[0] * axes[1])

    def updateCells(self, x, y, t):  # Update the density of each cell of the image
        self.cells *= math.exp((-t + self.t_ant) * TAU)
        self.t_ant = t
        for iy, ix in np.ndindex(self.cells.shape):
            if (
                ix / self.w <= x / self.width <= (ix + 1) / self.w
                and iy / self.h <= y / self.height <= (iy + 1) / self.h
            ):
                self.cells[iy, ix] += 1
                return

    def updateEllipsoid(
        self, t, axes
    ):  # Returns the spatio temporal density of the tracked ellipse
        self.densityBB *= math.exp((-t + self.t_ant) * TAU)
        self.densityBB += 1
        self.t_ant = t
        return self.densityBB / (math.pi * axes[0] * axes[1])

    def updatecellsMean(self): 
        self.cellsMean = self.cells - self.cells.mean()
        return self.cells.std(), self.cells.mean()

    def selectMax(self):
        ind = np.unravel_index(np.argmax(self.cells, axis=None), self.cells.shape)
        start_point = (
            int(ind[1] / self.w * self.width),
            int(ind[0] / self.h * self.height),
        )
        end_point = (
            int((ind[1] + 1) / self.w * self.width),
            int((ind[0] + 1) / self.h * self.height),
        )
        return start_point, end_point

    def selectMaxPercentage(self, perc=0.6):
        start_points = []
        end_points = []
        max_density = self.cells.max()
        indexes = np.where(self.cells >= perc * max_density)
        indexes = np.asarray(indexes)
        for i in range(len(indexes[0])):
            start_points.append(
                (
                    int(indexes[1, i] / self.w * self.width),
                    int(indexes[0, i] / self.h * self.height),
                )
            )
            end_points.append(
                (
                    int((indexes[1, i] + 1) / self.w * self.width),
                    int((indexes[0, i] + 1) / self.h * self.height),
                )
            )
        return start_points, end_points

    def updateArea(self, x, y, t): # Update the cells taking the are of each one into account
        self.cells *= math.exp((-t + self.t_ant) * TAU)
        for iy, ix in np.ndindex(self.cells.shape):
            if (
                ix / self.w <= x / self.width <= (ix + 1) / self.w
                and iy / self.h <= y / self.height <= (iy + 1) / self.h
            ):
                self.cells[iy, ix] += 1
                break
        self.areas = self.cells / self.cellArea
        self.t_ant = t

def crop(img, bb):
    warped = img.copy()
    alpha = bb.alpha*180.0/math.pi + 90
    if alpha > 90:
        alpha -= 180.0
    if alpha < -90:
        alpha += 180.0
    R = cv2.getRotationMatrix2D(tuple(bb.mean.tolist()), alpha, 1)
    rows,cols=img.shape[:2]
    warped = cv2.warpAffine(warped, R, (cols,rows))
    c = tuple(bb.mean.tolist())
    s = (2 * bb.axes).tolist()
    px = int(c[0] - 0.5*s[1])
    py = int(c[1] - 0.5*s[0])
    if px < 0:
        s[1] += px
        px = 0
    if py < 0:
        s[0] += py
        py = 0
    warped = warped[py:int(py+s[0]), px:int(px+s[1])]
    return warped

def readBag(bag_name, path):
    path = path + bag_name + ".bag" # Path to read the bag
    bag = rosbag.Bag(path)          # Bag object
    topic_events = "/dvs/events"
    topic_images = "/dvs/image_raw"
    # Storing the events
    Event_dataset = [e for events in bag.read_messages(topics=topic_events) for e in events.message.events]
    # Storing the images
    Images_dataset = [images for images in bag.read_messages(topic_images)]
    return Event_dataset, Images_dataset
