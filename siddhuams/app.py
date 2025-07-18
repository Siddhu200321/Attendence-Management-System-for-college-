import os
import cv2
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import csv
import requests
from geopy.distance import geodesic

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Get current location using IPinfo
def get_current_location():
    try:
        response = requests.get('http://ipinfo.io', timeout=5)
        data = response.json()
        loc = data.get('loc', '')
        if loc:
            lat, lon = map(float, loc.split(','))
            print(f"Current location: Latitude = {lat}, Longitude = {lon}")
            return lat, lon
    except Exception as e:
        print(f"Error fetching location: {e}")
    return None

# Check if user is within allowed radius
def is_location_valid(permitted_location, current_location, radius_meters=200):
    if current_location is None:
        return False
    try:
        distance = geodesic(permitted_location, current_location).meters
        print(f"Distance from permitted location: {distance:.2f} meters")
        return distance <= radius_meters
    except Exception as e:
        print(f"Error calculating distance: {e}")
        return False

# Log attendance if location is valid
def log_face_capture_to_csv(name, roll_no, branch_name, permitted_location):
    current_location = get_current_location()

    if not is_location_valid(permitted_location, current_location):
        print("Access denied: You are not within the permitted location.")
        return False

    csv_file = f'{branch_name}_attendance.csv'
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Name', 'Roll Number', 'Date', 'Time'])
        capture_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date, time = capture_time.split()
        writer.writerow([name, roll_no, date, time])
        print("✅ Attendance logged successfully.")
        return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture_face', methods=['GET', 'POST'])
def capture_face():
    if request.method == 'POST':
        name = request.form['name']
        roll_no = request.form['roll_no']
        branch = request.form['branch']
        
        if not name or not roll_no or not branch:
            flash('Please fill in all fields.', 'error')
            return redirect(url_for('capture_face'))
        
        branch_dir = f"branches/{branch}"
        person_dir = f"{branch_dir}/{name}"

        os.makedirs(person_dir, exist_ok=True)

        count = 0
        video_capture = cv2.VideoCapture(0)
        try:
            while count < 4:
                ret, frame = video_capture.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face_region = gray[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
                    if len(eyes) >= 2:
                        face_file = f"{person_dir}/{name}_{roll_no}_{count + 1}.jpg"
                        cv2.imwrite(face_file, face_region)
                        count += 1
                        if count < 4:
                            flash("Please slightly change your head position or expression.", 'info')
                    else:
                        flash("Eyes not detected. Please open your eyes.", 'warning')

                cv2.imshow("Capture Face - Press 'q' to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()

        if count == 4:
            train_face_recognizer()
            flash('All face images captured and model trained successfully.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Failed to capture sufficient images, please try again.', 'error')

    return render_template('capture_face.html')

# Train recognizer
def train_face_recognizer():
    faces = []
    labels = []
    label_dict = {}
    label_counter = 0

    face_images_dir = 'branches'
    for branch_name in os.listdir(face_images_dir):
        branch_path = os.path.join(face_images_dir, branch_name)
        if os.path.isdir(branch_path):
            for person_name in os.listdir(branch_path):
                person_path = os.path.join(branch_path, person_name)
                if os.path.isdir(person_path):
                    for img_name in os.listdir(person_path):
                        if img_name.endswith('.jpg'):
                            img_path = os.path.join(person_path, img_name)
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            faces.append(img)
                            labels.append(label_counter)
                    label_dict[label_counter] = f"{branch_name}_{person_name}"
                    label_counter += 1

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save('face_model.yml')

    with open('label_dict.pkl', 'wb') as file:
        pickle.dump(label_dict, file)

# Attendance route
@app.route('/take_attendance', methods=['GET', 'POST'])
def take_attendance():
    if request.method == 'POST':
        roll_no = request.form['roll_no']
        branch = request.form['branch']

        if not os.path.exists('face_model.yml') or not os.path.exists('label_dict.pkl'):
            return jsonify({"status": "error", "message": "No trained model found. Please capture faces first."})

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('face_model.yml')

        with open('label_dict.pkl', 'rb') as file:
            label_dict = pickle.load(file)

        video_capture = cv2.VideoCapture(0)
        recognized_name = None

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                id_, confidence = recognizer.predict(face)

                if confidence < 70:
                    label = label_dict.get(id_, "Unknown")
                    branch_name, person_name = label.split('_', 1)
                    recognized_name = person_name

                    permitted_location = (17.384, 78.4564)  # Set to your permitted coordinates
                    current_location = get_current_location()

                    video_capture.release()
                    cv2.destroyAllWindows()

                    if is_location_valid(permitted_location, current_location):
                        success = log_face_capture_to_csv(person_name, roll_no, branch_name, permitted_location)
                        if success:
                            return jsonify({"status": "success", "message": f"Attendance marked for {recognized_name}."})
                        else:
                            return jsonify({"status": "error", "message": "Attendance not logged due to location mismatch."})
                    else:
                        return jsonify({"status": "error", "message": "Location not valid. Attendance cannot be marked."})

            cv2.imshow("Face Recognition - Press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        return jsonify({"status": "error", "message": "Face not recognized. Try again."})

    return render_template('take_attendance.html')

# Other route helpers
@app.route('/show_link')
def show_link():
    return render_template('show_link.html', link="http://localhost:5000/take_attendance")

@app.route('/take_attendance')
def take_attendance_page():
    return render_template('take_attendance.html')

if __name__ == '__main__':
    app.run(debug=True)
