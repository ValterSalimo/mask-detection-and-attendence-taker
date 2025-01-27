import cv2
import numpy as np
import face_recognition
import os
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import io
import csv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

model = load_model('maskdetection.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
known_faces_dir = 'known_faces'

known_face_encodings = []
known_face_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith('.jpg'):
        image = cv2.imread(os.path.join(known_faces_dir, filename))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(image_rgb)
        if encodings:
            encoding = encodings[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])

video_capture = cv2.VideoCapture(0)
verified_individuals = set()

attendance_dir = 'attendance'
if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

current_date = datetime.datetime.now().strftime("%Y-%m-%d")
attendance_file_path = os.path.join(attendance_dir, f"{current_date}.csv")

with open(attendance_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Student Name", "Mask", "Attendance", "Verification Time"])

    while True:
        current_time = datetime.datetime.now()
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_day = current_time.weekday()

        if (current_day >= 0 and current_day <= 4) and (current_hour == 10 and current_minute >= 0 and current_minute <= 20):
            ret, frame = video_capture.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                photo = frame.copy()
                rgb_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_photo)
                face_encodings = face_recognition.face_encodings(rgb_photo, face_locations)

                for (x, y, w, h), face_encoding in zip(faces, face_encodings):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]

                        if name in verified_individuals:
                            continue

                        verification_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        face = photo[y:y+h, x:x+w]
                        face_resized = cv2.resize(face, (224, 224))
                        face_normalized = face_resized / 255.0
                        face_reshaped = np.reshape(face_normalized, [1, 224, 224, 3])

                        try:
                            prediction = model.predict(face_reshaped)
                            pred_label = np.argmax(prediction)

                            if pred_label == 1:
                                mask_status = "Mask"
                                color = (0, 255, 0)
                            else:
                                mask_status = "No Mask"
                                color = (0, 0, 255)

                            writer.writerow([name, mask_status, 1, verification_time])
                            verified_individuals.add(name)

                            cv2.rectangle(photo, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(photo, mask_status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                            cv2.putText(photo, name, (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        except Exception as e:
                            print(f"Error in mask detection: {e}")

                cv2.imshow("Captured Photo with Mask Detection", photo)

            cv2.imshow("Live Camera Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"System is inactive. Current time: {current_time.strftime('%H:%M:%S')}")

video_capture.release()
cv2.destroyAllWindows()

print("Attendance Records:")
for name in verified_individuals:
    print(f"{name} marked present.")
