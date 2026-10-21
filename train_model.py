"""Train an OpenCV LBPH face recognizer from dataset/<name>/*.jpg."""
import os
import pickle
import cv2
import numpy as np

DATASET_PATH = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_model.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pkl")

def main():
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "cv2.face is unavailable. Install opencv-contrib-python, "
            "or use the OpenCV build supplied for your PYNQ environment."
        )
    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_PATH}")

    faces, labels, label_map = [], [], {}
    for label_id, person_name in enumerate(sorted(os.listdir(DATASET_PATH))):
        person_folder = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_folder):
            continue
        image_count_before = len(faces)
        for image_name in sorted(os.listdir(person_folder)):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(person_folder, image_name)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Skipping unreadable image: {path}")
                continue
            faces.append(image)
            labels.append(label_id)
        if len(faces) > image_count_before:
            label_map[label_id] = person_name
            print(f"Loaded {len(faces) - image_count_before} images for {person_name}")

    if not faces:
        raise RuntimeError("No readable face images found. Run capture_dataset.py first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.asarray(labels, dtype=np.int32))
    os.makedirs(MODEL_DIR, exist_ok=True)
    recognizer.write(MODEL_PATH)
    with open(LABELS_PATH, "wb") as file:
        pickle.dump(label_map, file)
    print(f"Training complete. Model saved to {MODEL_PATH}")
    print(f"Label map saved to {LABELS_PATH}")

if __name__ == "__main__":
    main()
