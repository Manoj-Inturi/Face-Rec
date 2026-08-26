import cv2

# Open the laptop camera
camera = cv2.VideoCapture(0)

while True:
    # Capture one frame
    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break

    # Show the camera
    cv2.imshow("My Camera", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close everything
camera.release()
cv2.destroyAllWindows()