import cv2 as cv
import numpy as np
from ultralytics import YOLO

model_color_car = YOLO("models/YOLO/color.pt")  # pretrained YOLOv8n model
model_brand_car = YOLO("models/YOLO/brand.pt")  # pretrained YOLOv8n model


def brand_classifier(image):

    results = model_brand_car.predict(image)

    frame = results[0].plot(boxes=True, labels=True)
    
    return frame


def color_classifier(image):

    results = model_color_car.predict(image)

    frame = results[0].plot(boxes=True, labels=True)
    
    return frame

