# PYNQ Z2 Face Recognition Attendance System

A Python/OpenCV prototype for face detection, LBPH face recognition, and CSV attendance logging, based on the uploaded project notes.

## Features
- Capture face images from a camera into per-person folders
- Detect faces with a Haar Cascade
- Train an LBPH recognizer
- Record recognized names, date, time, and a confidence-like score in CSV
- Save example output in `outputs/`

## Files
- `capture_dataset.py` — capture a small face dataset
- `train_model.py` — train the LBPH model
- `main.py` — run recognition and attendance logging
- `models/` — place generated model files here
- `dataset/` — place/capture face images here
- `outputs/` — example output screenshot and anonymized sample CSV

## Run order
1. Use the target PYNQ/Linux environment and connect a compatible USB camera.
2. Install compatible dependencies. On a PYNQ image, the preinstalled OpenCV build may be preferable to pip:
   `python3 -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'face'))"`
3. Capture images: `python3 capture_dataset.py`
4. Train: `python3 train_model.py`
5. Run attendance: `python3 main.py`

`cv2.face` requires an OpenCV build with the contrib/face module. The exact installation method depends on the PYNQ image and Python version.

## Output
The `outputs/` folder contains an anonymized sample attendance CSV and a screenshot extracted from the supplied project PDF. The screenshot is documentation evidence, not a newly generated test run.

## Important notes
- This is a reconstructed, cleaned-up version of code transcribed from a PDF. Check it on your own PYNQ Z2 setup before relying on it.
- LBPH returns a distance where lower is generally a closer match. `MAX_LBPH_DISTANCE` in `main.py` is a starting point and must be tuned using test data.
- The confidence-like score `100 - distance` is only a rough display value, not a calibrated probability.
- Attendance is recorded once per name per program run. For continuous daily operation, persist/check the date and existing CSV records.
- Do not commit real face images or identifiable attendance data to a public repository without consent. Use anonymized sample data.
