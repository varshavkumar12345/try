!nvidia-smi

!pip install ultralytics

from ultralytics import YOLO
import os
from IPython.display import display,Image
from IPython import display
display.clear_output()

!yolo mode=checks

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="rDCNDh4L8fc84eXqnDVi")
project = rf.workspace("superexpression").project("facial-expression-recognition-4ev5x")
version = project.version(3)
dataset = version.download("yolov8")

!yolo task=detect mode=train model=yolov8n.pt data={dataset.location}/data.yaml epochs=100 imgsz=640

Image(filename=f'/content/runs/detect/train3/confusion_matrix.png',width=600)

!yolo task=detect mode=val model=/content/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml

!yolo task=detect mode=predict model=/content/runs/detect/train/weights/best.pt conf=0.5 source={dataset.location}/test/images save=True

import glob
from IPython.display import Image,display
for img_path in glob.glob('/content/runs/detect/train3/*.jpg'):
  display(Image(filename=img_path,width=600))

