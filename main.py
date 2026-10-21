"""Recognize faces and record attendance in attendance.csv."""
import csv
import os
import pickle
from datetime import datetime

import cv2

MODEL_PATH = "models/face_model.yml"
LABELS_PATH = "models/labels.pkl"
CSV_FILE = "attendance.csv"
# LBPH confidence is a distance: lower values usually mean a closer match.
# Tune this threshold using your own camera/data before real use.
MAX_LBPH_DISTANCE = 70.0

def load_face_cascade():
    candidates = [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    ]
    for path in candidates:
        cascade = cv2.CascadeClassifier(path)
        if not cascade.empty():
            return cascade
    raise FileNotFoundError("Could not load Haar Cascade XML file.")

def ensure_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(["Name", "Date", "Time", "Confidence"])

def main():
    if not hasattr(cv2, "face"):
        raise RuntimeError("cv2.face is unavailable; use an OpenCV contrib build.")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        raise FileNotFoundError("Model files missing. Run train_model.py first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH, "rb") as file:
        label_map = pickle.load(file)

    face_cascade = load_face_cascade()
    camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera. Check the USB camera connection.")

    ensure_csv()
    logged_today = set()
    today = datetime.now().strftime("%Y-%m-%d")
    print("Attendance logging started. Press q to stop.")
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Camera frame error.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in detected:
                face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                label_id, distance = recognizer.predict(face)
                name = label_map.get(label_id, "Unknown")
                recognized = name != "Unknown" and distance <= MAX_LBPH_DISTANCE
                display_name = name if recognized else "Unknown"
                confidence_score = max(0.0, round(100.0 - float(distance), 2))
                color = (0, 180, 0) if recognized else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{display_name} ({confidence_score}%)",
                            (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)
                if recognized and name not in logged_today:
                    now = datetime.now()
                    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
                        csv.writer(file).writerow([
                            name, now.strftime("%Y-%m-%d"),
                            now.strftime("%H:%M:%S"), confidence_score
                        ])
                    logged_today.add(name)
                    print(f"Attendance logged: {name} at {now.strftime('%H:%M:%S')}")
            cv2.imshow("PYNQ Z2 Face Recognition Attendance - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print("Attendance logging stopped.")

if __name__ == "__main__":
    main()
