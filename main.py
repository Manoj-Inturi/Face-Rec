import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------
# 1. Load the face detector model
# --------------------------------

model_path = "models/face_detector.task"

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.FaceDetectorOptions(
    base_options=base_options
)

detector = vision.FaceDetector.create_from_options(options)

print("Face detector created successfully!")


# --------------------------------
# 2. Open the laptop camera
# --------------------------------

camera = cv2.VideoCapture(0)


# --------------------------------
# 3. Continuously read camera frames
# --------------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break


    # --------------------------------
    # 4. Convert OpenCV BGR → RGB
    # --------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------
    # 5. Convert frame to MediaPipe image
    # --------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------
    # 6. Detect faces
    # --------------------------------

    result = detector.detect(mp_image)


    # --------------------------------
    # 7. Draw a rectangle around faces
    # --------------------------------

    for detection in result.detections:

        box = detection.bounding_box

        x = box.origin_x
        y = box.origin_y
        width = box.width
        height = box.height

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (255, 225, 225),
            2
        )


    # --------------------------------
    # 8. Display the camera
    # --------------------------------

    cv2.imshow(
        "Face Detection",
        frame
    )


    # --------------------------------
    # 9. Press Q to quit
    # --------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------
# 10. Close camera and windows
# --------------------------------

camera.release()
cv2.destroyAllWindows()