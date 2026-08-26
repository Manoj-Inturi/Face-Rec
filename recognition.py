import cv2
import numpy as np
from insightface.app import FaceAnalysis


# --------------------------------
# 1. Load InsightFace
# --------------------------------

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# --------------------------------
# 2. Load Manoj's photo
# --------------------------------

known_image = cv2.imread("known_faces/manoj.jpg")

known_faces = app.get(known_image)


if len(known_faces) == 0:
    print("No face found in Manoj's photo.")
    exit()


# Get Manoj's face embedding
manoj_embedding = known_faces[0].embedding

print("Manoj's face loaded successfully!")


# --------------------------------
# 3. Open camera
# --------------------------------

camera = cv2.VideoCapture(0)


while True:

    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break


    # --------------------------------
    # 4. Detect faces in camera
    # --------------------------------

    faces = app.get(frame)


    # --------------------------------
    # 5. Compare each face
    # --------------------------------

    for face in faces:

        current_embedding = face.embedding


        # Calculate similarity
        similarity = np.dot(
            manoj_embedding,
            current_embedding
        ) / (
            np.linalg.norm(manoj_embedding)
            * np.linalg.norm(current_embedding)
        )


        # --------------------------------
        # 6. Decide if it is Manoj
        # --------------------------------

        if similarity > 0.5:
            name = "Manoj"
        else:
            name = "Unknown"


        # --------------------------------
        # 7. Draw face box
        # --------------------------------

        x1, y1, x2, y2 = face.bbox.astype(int)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )


        # --------------------------------
        # 8. Show name
        # --------------------------------

        cv2.putText(
            frame,
            name,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # --------------------------------
    # 9. Show camera
    # --------------------------------

    cv2.imshow(
        "Face Recognition",
        frame
    )


    # --------------------------------
    # 10. Press Q to quit
    # --------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
