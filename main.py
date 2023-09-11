import cv2
import numpy as np
import tensorflow as tf

emotion_model = tf.keras.models.load_model('emotion_detection_model.h5')

cap = cv2.VideoCapture(0)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Natural']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1,
                                                                                           minNeighbors=5,
                                                                                           minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        resized_face = cv2.resize(face_roi, (48, 48))
        normalized_face = resized_face / 255.0
        emotion_prediction = emotion_model.predict(np.expand_dims(normalized_face, axis=0))
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Real-time Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()