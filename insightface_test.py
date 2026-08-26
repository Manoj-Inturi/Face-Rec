import cv2
from insightface.app import FaceAnalysis


# Create the InsightFace application
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

# Prepare the model
app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# Open the laptop camera
camera = cv2.VideoCapture(0)


while True:

    # Get a frame from camera
    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break


    # Detect faces
    faces = app.get(frame)


    # Draw box around every detected face
    for face in faces:

        x1, y1, x2, y2 = face.bbox.astype(int)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )


    # Show camera
    cv2.imshow(
        "InsightFace Detection",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
