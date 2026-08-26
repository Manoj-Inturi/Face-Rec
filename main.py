import cv2

# Open the camera
camera = cv2.VideoCapture(0)

# Load OpenCV's face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break

    # Convert the camera image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # Draw a rectangle around every detected face
    for x, y, width, height in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (255, 0, 0),
            2
        )

    cv2.imshow("Face Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()