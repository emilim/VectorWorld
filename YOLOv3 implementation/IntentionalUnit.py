import time
import cv2
import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
import os
os.environ['TFHUB_CACHE_DIR'] = '/home/user/workspace/tf_cache'

num_classes = 80
size = 416
weights = 'C:/Users/emili/Documents/GitHub/YOLOv3 Vector implementation/main/weights/yolov3.tf'
classes = 'C:/Users/emili/Documents/GitHub/YOLOv3 Vector implementation/main/data/coco.names'
class_names = [c.strip() for c in open(classes).readlines()]

yolo = YoloV3(classes=num_classes)
yolo.load_weights(weights)

class IntentionalUnit:
    def __init__(self, robot):
        self.fps = 0.0
        self.robot = robot

    def category(self, img_ori):
        #img = np.asarray(img_ori, np.float32)
        #img = img[np.newaxis, :] / 255.
        img_in = tf.expand_dims(img_ori, 0)
        img_in = transform_images(img_in, size)
        t1 = time.time()
        boxes, scores, classes, nums = yolo(img_in)
        self.fps = (self.fps + (1. / (time.time() - t1))) / 2

        img_ori = draw_outputs(img_ori, (boxes, scores, classes, nums), class_names)
        img_ori = cv2.putText(img_ori, "FPS: {:.2f}".format(self.fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        return img_ori

    def intention(self):
        self.robot.motors.set_wheel_motors(self.left_wheel, self.right_wheel)