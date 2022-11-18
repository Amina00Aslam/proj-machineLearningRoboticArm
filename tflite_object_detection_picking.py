# This program uses a TensorFlow Lite object detection model to perform object detection on an image
# or a folder full of images. It draws boxes and scores

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util
import time
import serial
from serial.tools import list_ports

global opening_cnt

count_objects=0
cal_angle=90
vid = cv2.VideoCapture(0)
count_label=""
obj_count=0
obj_count_i=0
label_found=0
opening_cnt=0

den_status=False
count_flag=False



serial_en=False
x_anngle=0
x_angle_pre=0
ports = list(serial.tools.list_ports.comports())
for port in ports:
    port_c1=str(port)
    print(port_c1)
    if(port_c1.find("USB Serial")):
        port_c2=port_c1.split('-')[0]
        port_c3=port_c2[0:len(port_c2)-1]

        port_g = serial.Serial(port_c3,timeout=1, baudrate=9600)
        serial_en=True
        print(port_g)

if(serial_en==True):
        port_g.write(b"#90#\n")
        time.sleep(5)
# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default= 'Sample_TFLite_mode_dust_new')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.40)
parser.add_argument('--image', help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir',
                    default=None)
parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                    default=None)
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
#MODEL_NAME = "Sample_TFLite_mode_dust_new"
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# Parse input image name and directory.
IM_NAME = args.image
IM_DIR = args.imagedir

# If both an image AND a folder are specified, throw an error
if (IM_NAME and IM_DIR):
    print('Error! Please only use the --image argument or the --imagedir argument, not both. Issue "python TFLite_detection_image.py -h" for help.')
    sys.exit()

# If neither an image or a folder are specified, default to using 'test1.jpg' for image name
if (not IM_NAME and not IM_DIR):
    IM_NAME = '8.jpg'

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'


# Get path to current working directory
CWD_PATH = os.getcwd()

# Define path to images and grab all image filenames
if IM_DIR:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_DIR)
    images = glob.glob(PATH_TO_IMAGES + '/*')

elif IM_NAME:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_NAME)
    images = glob.glob(PATH_TO_IMAGES)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Loop over every image and perform detection
while True:

    # Load image and resize to expected shape [1xHxWx3]
    #image = cv2.imread(image_path)
    ret, frame = vid.read()
    image=frame
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
##    print(type(boxes))
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0) and labels[int(classes[i])]!='dining table'):
##            count_objects+=1
            print(labels[int(classes[i])])
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            #object_name='car'

            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            x_c1=(abs(xmax-xmin)/2)
            #print(x_c1)
            x_c2=xmin+x_c1
            #print(x_c2)
            y_c1=(abs(ymax-ymin)/2)
            y_c2=ymin+y_c1
            x_cf=(x_c1+x_c2)/2
            print("x_cf",x_cf)
            x_angle=int(abs((180/350))*x_cf)
            print(x_angle)
            x_angle_s="#"+str(x_angle)+"#"
            print(x_angle_s)
            if(serial_en==True):
                if(abs(x_angle-x_angle_pre)>5):
##                    port_g.write(b"#1#\n")

                    port_g.write(bytes(x_angle_s, 'utf-8'))
                    time.sleep(1)
                    x_angle_pre=x_angle
                    opening_cnt-=1
                else :
                    opening_cnt+=1
                if(opening_cnt>3):

                    port_g.write(b"#500#\n")
                    time.sleep(1)
                    opening_cnt=0
##                port_g.write(b"##\n")
##                time.sleep(5)
            #print("("+str(xmin),str(xmax)+")"            #print("("+str(ymin),str(ymax)+")")
            #print("("+str(x_c2)+","+str(y_c2)+")"+"  "+str(i))
    # All the results have been drawn on the image, now display the image
##    if(count_objects>2):
##
##        obj_count_i+=1
##        print(obj_count_i)
##        print(count_objects)
##        print("****************************************8")
##    elif(count_objects<=2):
##        obj_count_i-=1
##
##    if(obj_count_i>5):
##        count_label="Status: "+"Dense"
##        den_status=True
##        obj_count_i=0
##    elif(obj_count_i<0):
##        count_label="Status: "+"Normal"
##        obj_count_i=0
##        den_status=False

    cv2.putText(image, count_label, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Draw label text
    cv2.imshow('Object detector', image)
    count_objects=0
    cv2.imwrite('output\8.jpg',image)
    #time.sleep(1)
    # Press any key to continue to next image, or press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
