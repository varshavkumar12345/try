!pip install ultralytics

from ultralytics import YOLO
import cv2
model=YOLO('/content/best-expression-size767-epoch100.pt')
results=model(source="/content/Screen Recording 2024-11-07 213651.mp4",show=True,conf=0.2,save=True)
#results=model(source="/content/WIN_20241016_19_55_37_Pro.mp4",show=True,conf=0.2,save=True)

