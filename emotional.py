import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load Pre-trained Emotion Detection Model
emotion_model = load_model("emotion_model.h5", compile=False)  # Load your emotion detection model
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start Webcam for Real-Time Emotion Detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image. Exiting...")
        break

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (320, 240))  # Smaller resolution for faster processing

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (64, 64)) / 255.0
            face = face.reshape(1, 64, 64, 1)

            # Predict Emotion
            prediction = emotion_model.predict(face)
            emotion = emotions[np.argmax(prediction)]

            # Display Emotion
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Emotion Detection", frame)

    # Break the loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
