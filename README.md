# Real-Time Face Recognition System

A beginner-friendly real-time face recognition project built with Python, OpenCV, MediaPipe, and InsightFace.

This project is the first stage of a larger multi-camera person-tracking system planned for a hackathon. The long-term goal is to recognize registered people across classrooms and corridors, record their movement, and display it on a 2D map.

## Project Goal

The first version demonstrates how to:

1. Access a laptop camera.
2. Detect faces in live video.
3. Generate face embeddings.
4. Compare a live face with a saved face.
5. Recognize a person as known or unknown.
6. Display the person's name.
7. Use different bounding-box colors for recognition results.

The basic pipeline is:

```text
Laptop Camera
      |
      v
Live Video Frame
      |
      v
Face Detection
      |
      v
Face Embedding
      |
      v
Compare With Saved Face
      |
      +----------------+
      |                |
      v                v
  Recognized        Unknown
      |                |
      v                v
    Manoj           Unknown
```

## Technologies

### Python

Python is used as the main programming language because it is beginner-friendly and has a strong computer-vision and AI ecosystem.

### OpenCV

OpenCV is used for camera access, video frames, windows, rectangles, text, and basic image processing.

```python
camera = cv2.VideoCapture(0)
```

### MediaPipe

MediaPipe was used during the initial face-detection stage with its Tasks API and the `face_detector.task` model. This helped demonstrate the difference between face detection and face recognition.

### InsightFace

InsightFace is used for the main face-analysis and recognition pipeline.

```python
FaceAnalysis(name="buffalo_l")
```

The model provides face detection and face embeddings through an SCRFD-based pipeline.

### NumPy

NumPy is used to calculate similarity between saved and live face embeddings.

## Installation

Check Python and pip:

```bash
python --version
pip --version
```

Install the project dependencies:

```bash
pip install opencv-python
pip install mediapipe
pip install -U insightface
pip install onnxruntime
```

## Project Structure

```text
Face -Rec/
|
|-- main.py
|-- insightface_test.py
|-- recognition.py
|-- known_faces/
|   `-- manoj.jpg
`-- models/
    `-- face_detector.task
```

### `main.py`

The initial OpenCV and MediaPipe face-detection implementation.

### `insightface_test.py`

An InsightFace camera test that detects faces using the `buffalo_l` model.

### `recognition.py`

The face-recognition implementation. It loads Manoj's saved image, creates an embedding, compares it with live camera embeddings, and displays a known or unknown label.

### `known_faces/`

Contains saved images for people who should be recognized.

### `models/`

Contains the MediaPipe face detector model used during the learning stage.

## Face Detection

Face detection answers:

> Is there a face in this image, and where is it?

It produces a bounding box, but it does not identify the person.

```text
Camera
  |
  v
Image
  |
  v
Face Detector
  |
  v
Face Location
```

## Face Recognition

Face recognition answers:

> Whose face is this?

The face is converted into a numerical representation called an embedding. The live embedding is compared with the saved embedding. If the similarity is high enough, the person is recognized; otherwise, the result is unknown.

```text
Saved Photo                  Live Camera
     |                            |
     v                            v
    Face                         Face
     |                            |
     v                            v
 Embedding                    Embedding
     |                            |
     +------------+-------------+
                  |
                  v
             Similarity
                  |
          Known or Unknown
```

OpenCV uses BGR color order:

```text
(0, 255, 0)       Green
(255, 255, 255)   White
(0, 0, 255)       Red
```

## Development Process

The project was built incrementally:

1. Set up Python and pip.
2. Open the laptop camera with OpenCV.
3. Read frames using `success, frame = camera.read()`.
4. Explore MediaPipe face detection.
5. Test InsightFace for stronger detection.
6. Generate and compare face embeddings.
7. Add known and unknown labels.

## Problems and Debugging

### OpenCV CascadeClassifier

An early Haar Cascade attempt produced an error similar to:

```text
AttributeError: module 'cv2' has no attribute 'CascadeClassifier'
```

The installed OpenCV version and packages were checked instead of assuming that the tutorial matched the local environment:

```bash
python -c "import cv2; print(cv2.__version__)"
pip list | findstr opencv
```

This showed that the installed OpenCV version did not match the older tutorial approach, so the project moved to MediaPipe Tasks and InsightFace.

### MediaPipe Model URL

An older model download URL was no longer valid. The project was updated to use the current MediaPipe Tasks API and the local `face_detector.task` model.

### Long-Range Detection

The initial detector was less reliable when faces were far from the camera. This led to testing InsightFace and its stronger modern face-analysis pipeline.

## Current Limitations

This is a learning project and is not a production-ready security system. Current limitations include:

- Accuracy depends on lighting and camera quality.
- Small faces and side profiles may be difficult to recognize.
- Only one known person is currently configured.
- The similarity threshold needs proper testing.
- There is no database or attendance system yet.
- Multi-camera synchronization and movement tracking are not implemented.
- There is no web dashboard or 2D floor map yet.

## Future Development

### Project 2: Multiple People

Support multiple registered faces:

```text
known_faces/
|-- manoj.jpg
|-- rahul.jpg
|-- arjun.jpg
`-- priya.jpg
```

### Project 3: PostgreSQL

Store people and face embeddings in a database.

```text
Students
-------------------------
ID
Name
Face Embedding
```

### Project 4: Entry and Exit Tracking

Record events such as:

```text
Person: Manoj
Camera: 1
Event: ENTER
Time: 09:05:12
```

### Final Hackathon System

The planned architecture is:

```text
Camera 1 --+
Camera 2 --+--> Recognition --> Backend --> PostgreSQL
Camera 3 --+                                  |
                                                v
                                         React Dashboard
                                                |
                                                v
                                           2D Map
```

The dashboard could show a person's movement, for example:

```text
09:05 -> Classroom 1
09:18 -> Corridor
09:20 -> Classroom 2
```

## What I Learned

- Python variables, loops, conditions, imports, and multiple assignment.
- Camera access and frame processing with OpenCV.
- The difference between face detection and face recognition.
- How face embeddings represent faces as numerical vectors.
- How to investigate package versions and error messages.
- How a large AI project can be built through small, testable stages.

## Development Philosophy

This project follows a learning-by-building approach. AI assistance was used as a coding tutor, debugger, and explanation tool. The focus was on understanding what the code does, why each component is needed, how the components connect, and how to debug failures.

## Project Status

```text
Python Setup             Done
OpenCV Camera            Done
Face Detection           Done
InsightFace              Done
Face Embeddings          Done
Face Recognition         Done
Known/Unknown Detection  Done

Multiple People          Planned
PostgreSQL               Planned
Entry/Exit Logging       Planned
Multiple Cameras         Planned
React Dashboard          Planned
Movement Timeline        Planned
2D Floor Map             Planned
```

## Next Milestone

The next milestone is multi-person face recognition. The hard-coded `name = "Manoj"` approach will be replaced with a system that recognizes multiple registered people automatically, preparing the project for PostgreSQL and multi-camera tracking.