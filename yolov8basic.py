from ultralytics import YOLO
import numpy

# load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")  

# predict on an image
detection_output = model.predict(source="videos/videoa.mp4", conf=0.4, save=True) 


# Display tensor array
#print(detection_output)

# Display numpy array
#print(detection_output[0].cpu().numpy())

#print(len(detection_output))
#print(detection_output[0][0][0].cpu().numpy())