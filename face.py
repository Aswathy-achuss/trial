import cv2

# Step 1: Load Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Step 2: Start webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("📸 Face Detection Started... Press 'q' to Quit.")

while True:
    # Step 3: Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Step 4: Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 5: Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )

    # Step 6: Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Step 7: Display the result
    cv2.imshow("Face Detection using Haar Cascade", frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 8: Release resources
cap.release()
cv2.destroyAllWindows()
