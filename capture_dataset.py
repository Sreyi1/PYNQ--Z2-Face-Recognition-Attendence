"""Capture face images into dataset/<student_name>/ using a USB camera."""
import os
import cv2

CASCADE_PATHS = [
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
]
DATASET_PATH = "dataset"
IMAGES_PER_PERSON = 20

def load_face_cascade():
    for path in CASCADE_PATHS:
        cascade = cv2.CascadeClassifier(path)
        if not cascade.empty():
            return cascade
    raise FileNotFoundError(
        "Could not load haarcascade_frontalface_default.xml. "
        "Install OpenCV or update CASCADE_PATHS."
    )

def main():
    face_cascade = load_face_cascade()
    camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera. Check the USB camera connection.")

    os.makedirs(DATASET_PATH, exist_ok=True)
    try:
        print("Face dataset capture started. Type 'done' to finish.")
        while True:
            name = input("Student name (or 'done'): ").strip()
            if name.lower() == "done":
                break
            if not name or name in {".", ".."} or "/" in name or "\\" in name:
                print("Enter a non-empty name without slashes.")
                continue

            folder = os.path.join(DATASET_PATH, name)
            os.makedirs(folder, exist_ok=True)
            count = len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
            print(f"Capturing up to {IMAGES_PER_PERSON} images for {name}. Press q to stop this person.")
            while count < IMAGES_PER_PERSON:
                ret, frame = camera.read()
                if not ret:
                    print("Failed to capture a frame.")
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    filename = os.path.join(folder, f"{name}_{count:03d}.jpg")
                    if cv2.imwrite(filename, face):
                        count += 1
                        print(f"Saved {filename} ({count}/{IMAGES_PER_PERSON})")
                    break
                cv2.imshow("Face capture - press q to stop person", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print("Dataset capture finished.")

if __name__ == "__main__":
    main()
