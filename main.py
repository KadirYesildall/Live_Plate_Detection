from ultralytics import YOLO
import cv2

import functions
from sort.sort import *
from functions import get_car, read_license_plate, write_csv, write_cars_to_csv

obj_tracker = Sort()

results = {}
result_cars = {}
# Loading the pretrained model and model that we trained with our dataset
coco_model = YOLO('./ModelTrain/yolov8n.pt')
plate_detector = YOLO('./ModelTrain/runs/detect/train18/weights/best.pt')

# Loading the video i am going to use
cap = cv2.VideoCapture('Traffic.mp4')

# vehicles we need on coco dataset for vehicle detection are car, motorbike, bus, truck
vehicles = [2, 3, 5, 7]

ret = True
frame_number = -1
while ret:
    frame_number += 1
    ret, frame = cap.read()
    if ret:
        results[frame_number] = {}
        result_cars[frame_number] = {}
        plate_detections = coco_model(frame)[0]
        detections_bbox = []
        for detection in plate_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_bbox.append([x1, y1, x2, y2, score])

        # We will track vehicles using sort algorithm
        track_info = obj_tracker.update(np.asarray(detections_bbox))

        for car_info in track_info:
            xcar1, ycar1, xcar2, ycar2, car_id = car_info
            result_cars[frame_number][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]}}

        # We will detect license plate using the license plate detector that we trained using yolov8
        plate_detections = plate_detector(frame)[0]
        for license_plate in plate_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate


            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_info)


            if car_id != -1:
                # We will crop license plates to read the context of the license plate with EasyOcr more efficiently
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Using the cropped license plate we are getting the license plate text and its confidence score
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_number][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                     'license_plate': {'bbox': [x1, y1, x2, y2]},
                                                     'text': license_plate_text,
                                                     'bbox_score': score,
                                                     'text_score': license_plate_text_score}





# We will write the results to a csv file to monitor our detection system
write_csv(results, './plate_results.csv')
write_cars_to_csv(result_cars, './cars_results.csv')
