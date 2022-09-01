from yolov5.detect_multiple import run as yolo
from yolov5.utils.augmentations import letterbox

class YoloPred():

    def __init__(self, conf_th = 0.25):
        self.conf_th = conf_th
        self.img_size=640 # Size to convert the image to the proper yolo size
        self.stride = 32 # Parameter for yolo

    def changeFormat(self):
        im0 = self.image.copy() # Image in the original format
        self.image = letterbox(self.image, self.img_size, stride=self.stride)[0].transpose(2,0,1) 		# Converting image to yolo format
        return im0
    
    def predict(self, image):
        self.image = image
        im0 = self.changeFormat()
        pred = yolo(im = self.image, im0=im0)
        return pred
