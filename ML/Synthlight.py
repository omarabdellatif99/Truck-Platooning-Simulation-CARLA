import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import Counter
import joblib
from sklearn.metrics import classification_report

def preprocess_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    X = []
    y = []

    for entry in data:
        for lane in entry["lanes"]:
            features = [lane["xmin"], lane["ymin"], lane["xmax"], lane["ymax"]]
            X.append(features)

            target = 0 if lane["type"] == "solidlane" else 1
            y.append(target)

    return np.array(X), np.array(y)

def detect_lanes(gray_frame):
    edges = cv2.Canny(gray_frame, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_lanes = []

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            detected_lanes.append((x, y, x + w, y + h))  # (xmin, ymin, xmax, ymax)

    return detected_lanes

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=10000, C=0.01, penalty='l2', solver='lbfgs')

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean() * 100:.2f}%")

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print(f"Model Accuracy: {np.mean(y_pred == y_test) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'lane_classification_model.pkl')
    print("Model saved as 'lane_classification_model.pkl'")

    return model, scaler

def process_video(video_path, model, scaler):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    print("Video opened successfully.")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lanes = detect_lanes(gray)
        for lane in lanes:
            X_test = np.array([[lane[0], lane[1], lane[2], lane[3]]])
            X_test_scaled = scaler.transform(X_test)  # Scale the features
            prediction = model.predict(X_test_scaled)
            color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
            cv2.rectangle(frame, (lane[0], lane[1]), (lane[2], lane[3]), color, 2)
        cv2.imshow("Frame with Lane Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
