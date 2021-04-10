########################################################
##  Model: Lane detection & collision prevention system
##  Author : Rajat Agrawal
##  Initial commit: 12-Aug-2020
##  Last modification on 12-Aug-2020
##  Import all necessory packages. ####### First follow the installation instructions for tensorflow object detection API (link in readme). #######
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import cv2
from distutils.version import StrictVersion
from utils import label_map_util
from utils import visualization_utils as vis_util

##############################################################################
## ref: a function for refining the frame using gaussian blur and
## then applying canny edge detector. It returns the image with all the edges in it.
def ref(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    return cv2.Canny(blur, 50, 150)

##########################################################################
## region_of_interest: a function to apply region of interest (ROI)
## to capture only road lanes edges and removes noise (unnecessary edges).
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        ## define the shape of ROI
        ## coordinates may differ because of different
        ## camera positions on different vehicle. It requires one time calibration. 
        [(0, height-85 ), (2000, height-50), (1050, 595), (850,595)]
                        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 225)
    return cv2.bitwise_and(image, mask)

#######################################################
## make_coordinates: a function to calculate the exact
## coordinates of lanes using average fit coordinates. 
def make_coordinates(image, line_param):
    try:
        slope, intercept = line_param
        y1 = image.shape[0]
        y2 = int(y1*(3/5))
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return np.array([x1, y1, x2, y2])
    except:
        pass

####################################################
## average_slope_intercept: a function to calculate
## the road lanes coordinates.
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    try:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            param = np.polyfit((x1, x2), (y1, y2), 1)
            slope = param[0]
            intercept = param[1]
            if slope <-0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))
    except:
        pass

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

################################################################
## display_lines: a function to display the detected road lanes.
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        try:
            for x1, y1, x2, y2 in lines:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 7)
        except:
            pass
    return line_image

## path modification (main.py is in object_detection folder).
sys.path.append("..")
from object_detection.utils import ops as utils_ops

## avoiding older versions of tensorflow.
if StrictVersion(tf.__version__) < StrictVersion('1.14.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.15.0')

## using RCNN model to detect objects.
MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'

# Path to frozen detection graph. Model for object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

## path to labels for each boxes (detected objects).
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

## loading model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

## loading the lable map.
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

## Capture live video frames through camera.
#cap = cv2.VideoCapture(0)
## Loading recorded video.
cap = cv2.VideoCapture("test2.mp4")

## running the session.
with detection_graph.as_default():
    with tf.Session() as sess:
        while True:
            ret, frame = cap.read()
            
            ## calling functions for lane detection on each frame.
            canny_image = ref(frame)
            cropped_image = region_of_interest(canny_image)
            lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
            averaged_lines = average_slope_intercept(frame, lines)
            line_image = display_lines(frame, averaged_lines)
            
            #Code for object detection
            ## expanding dimensions of image.
            frame_expanded = np.expand_dims(frame, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            ## boxes represents the detected objects.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            ## scores are the confidence of the object detection.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
            
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            
            ## alert message if there is chances of collision with the detected objects on the road. 
            for i, b in enumerate(boxes[0]):
                ## class 3,6,7,8 is for 'car', 'bus', 'train', and 'truck' respectively.
                if classes[0][i] in [3,6,7,8]:
                    if scores[0][i] > 0.5:
                        mid_x = (boxes[0][i][3] + boxes[0][i][1]) / 2
                        mid_y = (boxes[0][i][2] + boxes[0][i][0]) / 2
                        apx_distance = round( (1-(boxes[0][i][3] - boxes[0][i][1]))**4, 1)
                        cv2.putText(frame, '{}'.format(apx_distance), (int(mid_x*1800), int(mid_y*1000)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                        ## condition to predict collision chances.
                        if apx_distance <= 0.5:
                            if mid_x > 0.3 and mid_x < 0.7:
                                cv2.putText(frame, 'ALERT!!!', (int(mid_x*1800)-50, int(mid_y*1000)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
                                cv2.putText(frame, '*Slow down speed*', (int(mid_x*1800)-50, int(mid_y*1000)+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
                ## class 1,2,4 is for 'person', 'bicycle', and 'motorcycle' respectively.
                elif classes[0][i] in [1,2,4]:
                    if scores[0][i] > 0.5:
                        mid_x = (boxes[0][i][3] + boxes[0][i][1]) / 2
                        mid_y = (boxes[0][i][2] + boxes[0][i][0]) / 2
                        apx_distance = round( (1-(boxes[0][i][3] - boxes[0][i][1]))**4, 1)
                        cv2.putText(frame, '{}'.format(apx_distance), (int(mid_x*1800), int(mid_y*1000)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                        ## condition to predict collision chances.
                        if apx_distance <= 0.6:
                            if mid_x > 0.3 and mid_x < 0.7:
                                cv2.putText(frame, 'ALERT!!!', (int(mid_x*1800)-50, int(mid_y*1000)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                                cv2.putText(frame, '*Slow down speed*', (int(mid_x*1800)-50, int(mid_y*1000)+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            ## combining the frames (frame with detected lanes + frame with detected objects).
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
            ## result visualization.
            cv2.imshow('LDCP System', cv2.resize(combo_image, (800, 600)))
            ## to slose the display window. (process terminates)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        ## releasing the camera.
        cap.release()
