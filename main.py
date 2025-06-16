import sys

import cv2
import pandas as pd
import pickle
from ultralytics import YOLO
import numpy as np


POSE_MODEL_PATH = 'yolo11x-pose.pt'

# CLASSIFIER_PATH = 'body_language_model.pkl'


try:
    pose_model = YOLO(POSE_MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Load the classifier
#model selection
#pass model ('gb' to 'lr', 'rc', 'rf', or 'gb') as a command line argument like this:
# python main.py rf
# or change it in the else statement
if len(sys.argv) > 1:
    MODEL_TO_LOAD = sys.argv[1].lower()
else:
    # change 'gb' to 'lr', 'rc', 'rf', or 'gb'
    MODEL_TO_LOAD = 'rf' # Default model if no argument is provided

MODEL_FILENAME = f'body_language_{MODEL_TO_LOAD}.pkl'

# LOAD THE MODEL
try:
    with open(MODEL_FILENAME, 'rb') as f:
        model = pickle.load(f)
    print(f"Successfully loaded model: {MODEL_FILENAME}")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_FILENAME}' not found. Please ensure it has been trained and saved.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    sys.exit(1)

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam
cap.set(3, 1280)
cap.set(4, 720)
DISPLAY_WIDTH, DISPLAY_HEIGHT = 800, 600
cv2.namedWindow("YOLOPose Inference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOPose Inference", DISPLAY_WIDTH, DISPLAY_HEIGHT)
# h_frame, w_frame = cap.shape[:2]

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    annotated_img = frame.copy()
    results = pose_model.track(frame, conf=0.5, verbose=False)

    res = results[0]

    annotated_frame = res.plot(boxes=False, labels=False)

    # Iterate through each detected person using an index
    for i in range(len(res.boxes)):
        person_kpts = res.keypoints.data[i]  # Keypoints for person i
        person_box = res.boxes.data[i]  # Bounding box for person i
        # trackId = res.boxes.id[i].cpu().tolist()

        # print(trackId)

        # Ensure keypoints were detected
        if person_kpts.shape[0] == 0:
            continue

        try:
            # Prepare keypoints for the classifier
            # Flatten the keypoints tensor and convert to a list

            kpts_list = person_kpts.flatten().tolist()

            # pts = person_kpts.cpu().numpy()
            # kpts_list = [
            #     v
            #     for x, y, c in pts
            #     for v in (int(x), int(y)
            #               , round(c, 2))
            # ]

            # print(kpts_list)
            feature_names = model.named_steps['standardscaler'].feature_names_in_

            # Create a DataFrame with the correct feature names
            data = pd.DataFrame([kpts_list], columns=feature_names)

            # # Make a prediction
            # pred_class = model.predict(data)[0]
            # # pred_prob = model.predict_proba(data).max()
            #
            # # Get prediction probability
            # # RidgeClassifier does not have predict_proba
            # # RidgeClassifier does not have predict_proba
            # if hasattr(model, 'predict_proba'):
            #     body_language_prob = model.predict_proba(data)[0]
            #     prob_display = model.predict_proba(data).max()
            # else:
            #     # body_language_prob = None
            #     prob_display = "N/A"
            #
            # # Animation
            # # Get bounding box coordinates (top-left and bottom-right)
            # x1, y1, x2, y2 = person_box[:4].int().tolist()
            # cx = (x1+x2) //2
            # # print(person_box[:4].int().tolist())
            #
            # # Create the label text
            # label_text = f"{body_language_prob} ({prob_display:.2f})"
            #
            # # Calculate position for the text box
            # # We'll place it right above the person's bounding box
            # text_box_y = y1 - 5
            #
            # # Get text size to draw a proper background rectangle
            # (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 5)
            #
            # # Draw the background rectangle for the text
            # cv2.rectangle(annotated_frame, (cx, text_box_y - h - 5), (cx + w, text_box_y + 5), (255, 255, 0), -1)
            #
            # # Draw the classification text
            # cv2.putText(annotated_frame, pred_class.split()[0], (cx, text_box_y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw the person's bounding box
            # cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            data = pd.DataFrame([kpts_list], columns=feature_names)
            pred_class = model.predict(data)[0]

            # RidgeClassifier has no predict_proba
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(data).max()
                prob_str = f"{prob:.2f}"
            else:
                prob_str = "N/A"

            label_text = f"{pred_class.split()[0]} ({prob_str})"


            # Draw status box & text
            # Compute box coords
            x1, y1, x2, y2 = person_box[:4].int().tolist()
            cx = (x1+x2) //2

            box_w, box_h = 200, 60
            text_box_y = y1 - 5

            # Get text size to draw a proper background rectangle
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 5)

            # Draw the background rectangle for the text
            cv2.rectangle(annotated_frame, (cx, text_box_y - h - 5), (cx + w, text_box_y + 5), (255, 255, 0), -1)
            cv2.putText(annotated_frame, label_text, (cx, text_box_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)


        except Exception as e:
            # print(f"Error during classification or drawing: {e}")
            pass

    # Display the final frame
    cv2.imshow('YOLOPose Inference', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()