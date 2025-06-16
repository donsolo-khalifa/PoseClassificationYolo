import csv
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLOPose model
model = YOLO('yolo11x-pose.pt')  # replace with correct pose weights

output_csv = 'coords.csv'
# Prepare CSV headers
# 1 class label + 17 keypoints * (x,y,confidence)
num_kpts = 17
fields = ['class'] + [f'{ax}{i}' for i in range(1, num_kpts+1) for ax in ('x','y','c')]
if not os.path.exists(output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

class_name = 'Lewa'  # change per recording

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, stream=True, conf=0.5,verbose=False)
    for res in results:
        annotated_frame = res.plot()

        kpts = res.keypoints.data[0]  # shape (17,3)
        if kpts is None:
            continue
        row = [class_name] + kpts.flatten().tolist()

        # row = [class_name] + [[int(x),int(y),round(c, 2) ]for x,y,c in kpts.cpu().numpy()]

        # per_point = [[int(x), int(y), round(c, 2)] for x, y, c in kpts.cpu().numpy()]
        # row = [class_name] + sum(per_point, [])

        print(row)


        # for x, y, c in kpts.cpu().numpy():
        #     row = [class_name] +[int(x),int(y),round(c, 2)]
        #     cv2.circle(frame, (int(x), int(y)), 4, (0,255,0), -1)
        #
        #     print(row)

        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # draw keypoints
        # for x, y, c in kpts:
        #     cv2.circle(frame, (int(x), int(y)), 4, (0,255,0), -1)
    cv2.imshow('YOLOPose Extraction', annotated_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()