import random

import cv2
import numpy as np
from ultralytics import YOLO

# opening the file in read mode
my_file = open("utils/coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Vals to resize video frames | small frame optimise the run
frame_wid = 640
frame_hyt = 480

# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture("videos/videoa.MP4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()


#Keeps track of the number of objects it classified per video.
totalClasses = 0
#Keeps track of the number of times it classified something not in the video
DoNotExistClasses = 0
frameCount = 0
while True:
    frameCount +=1
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    if frameCount > 5:
        frameCount = 0

        # Predict on image
        detect_params = model.predict(source=[frame], conf=0.45, save=False)

        # Convert tensor array to numpy
        DP = detect_params[0].cpu().numpy()
    
        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                boxes = detect_params[0].boxes
                box = boxes[i]  # returns one box
                classIDs = box.cpu().cls.numpy()[0]
                totalClasses +=1
                if((classIDs != 0.0) and (classIDs != 67.0) and (classIDs !=24.0) and (classIDs !=56.0) and (classIDs !=57.0) and (classIDs !=62.0) and (classIDs !=63.0)):
                    DoNotExistClasses +=1
                    #print(f"Wrong Classification ID: {classIDs}")
                conf = box.cpu().conf.numpy()[0]
                bb = box.cpu().xyxy.numpy()[0]


# When everything done, release the capture
#print(f"Wrong: {DoNotExistClasses}")
print(f"Accuracy: {(totalClasses-DoNotExistClasses)/totalClasses}")
cap.release()