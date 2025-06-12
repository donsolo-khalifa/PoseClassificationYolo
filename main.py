import cv2
import numpy as np
import pandas as pd
import pickle
from ultralytics import YOLO

# Load YOLOPose and classifier
pose_model = YOLO('yolo11n-pose.pt')
with open('body_language_model.pkl', 'rb') as f:
    clf = pickle.load(f)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = pose_model(frame, stream=True, conf=0.5,verbose=False)
    for res in results:
        annotated_frame = res.plot()

        kpts = res.keypoints.data
        if kpts is None:
            continue

        try:
            # data = pd.DataFrame([kpts.flatten().tolist()])
            # pred = clf.predict(data)[0]
            # prob = clf.predict_proba(data).max()
            # # annotate
            # # for x,y,c in kpts:
            # #     cv2.circle(frame, (int(x), int(y)), 4, (0,255,0), -1)
            # cv2.putText(frame, f"{pred} ({prob:.2f})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # k = kpts.flatten()  # shape (51,)
            # X = k.reshape(1, -1)  # shape (1, 51)
            # pred = clf.predict(X)[0]
            # prob = clf.predict_proba(X).max()
            # cv2.putText(frame, f"{pred} ({prob:.2f})", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            feature_names = clf.named_steps['standardscaler'].feature_names_in_

            # then in your loop:
            data = pd.DataFrame([kpts.flatten().tolist()], columns=feature_names)
            pred = clf.predict(data)[0]
            prob = clf.predict_proba(data).max()
            cv2.putText(frame, f"{pred} ({prob:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except:
            pass

    cv2.imshow('YOLOPose Inference', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()