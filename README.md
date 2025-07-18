# Attendence-Management-System-for-college-
ATTENDENCE MANAGEMENT SYSTEM 

🧠 Face Recognition Based Attendance Management System with Geolocation Validation
This project is a smart attendance management system built using Python, Flask, OpenCV, and Geopy, which ensures:

Face recognition-based identity validation

Geolocation-based location verification

Real-time webcam-based face capture

CSV logging of attendance with date & time

🔍 Key Features
🎥 Face Capture & Training: Captures 4 facial images with open eyes and varying head positions for each user and trains a face recognition model (LBPH).

📍 Location Validation: Uses IP-based geolocation (ipinfo.io) and validates if the user is within a specified radius (default 200 meters) of a permitted location.

✅ Face Recognition & Attendance Marking: Recognizes faces live through webcam, checks location, and marks attendance in a CSV file if all conditions are met.

🗂 Branch-wise Face Storage: Organizes captured face data by branch and student.

📊 CSV Logging: Stores logs with Name, Roll Number, Date, and Time in a branch-wise attendance CSV file.

🌐 Flask Web Interface: Includes web routes to register face, capture attendance, and show the link to attendance page.

🧾 Tech Stack
Python 3.8+

Flask (Web framework)

OpenCV (Face detection and recognition)

Geopy (Distance calculation for geofencing)

Requests (To fetch IP-based location)

HTML/CSS for simple templates

📂 Project Structure
yaml
Copy
Edit
├── branches/                  # Stores face images per branch and student
├── templates/
│   ├── index.html
│   ├── capture_face.html
│   ├── take_attendance.html
│   └── show_link.html
├── face_model.yml             # Trained face recognition model
├── label_dict.pkl             # Pickled labels for face IDs
├── app.py                     # Main application script
└── <branch>_attendance.csv    # CSV files logging attendance
📌 How it Works
Navigate to /capture_face and enter name, roll number, and branch.

System captures 4 valid face images with both eyes open.

Train LBPH face recognizer with captured data.

Navigate to /take_attendance, show your face to the webcam.

If recognized and within the permitted location, attendance is logged.

🚀 Getting Started
Clone the repo:

bash
Copy
Edit
git clone https://github.com/yourusername/face-attendance-system.git
cd face-attendance-system
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
python app.py
Access it at http://localhost:5000

🔐 Note: Update the permitted_location in app.py with your institution’s latitude and longitude.


📜 License
MIT License
