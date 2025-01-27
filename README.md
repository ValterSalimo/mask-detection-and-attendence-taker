# mask-detection-and-attendence-taker
Mask Detection and Attendance System
Description
This project is a real-time mask detection and attendance system that uses computer vision and machine learning to identify whether individuals are wearing masks. The system also records attendance by recognizing faces and logging the data with timestamps. It is designed to be used in environments where mask compliance is mandatory, such as schools, offices, or public spaces.

The system leverages a pre-trained deep learning model for mask detection and uses Haar Cascade for face detection. It captures video feed from a webcam, detects faces, and classifies whether the person is wearing a mask or not. The attendance data is saved in a CSV file for each day, making it easy to track and verify attendance.

Features
Real-time Mask Detection: Detects whether a person is wearing a mask in real-time using a webcam feed.

Face Recognition: Recognizes individuals and logs their attendance.

Attendance Logging: Saves attendance data with timestamps in a CSV file.

User-friendly Interface: Displays the live feed with bounding boxes and labels indicating mask status.

Customizable: The system can be easily extended or modified to suit different environments.

Technologies Used
Python: The primary programming language used for the project.

OpenCV: For real-time video processing and face detection.

TensorFlow/Keras: For building and training the mask detection model.

Haar Cascade: For face detection.

CSV: For logging attendance data.

Installation
Clone the Repository:

bash
Copy
git clone https://github.com/your-username/mask-detection-attendance-system.git
cd mask-detection-attendance-system
Install Dependencies:

bash
Copy
pip install -r requirements.txt
Download Pre-trained Models:

Ensure you have the maskdetection.h5 model file in the project directory.

The Haar Cascade file (haarcascade_frontalface_default.xml) is included in the OpenCV package.

Run the System:

bash
Copy
python attendence.py
Usage
Mask Detection: The system will automatically detect faces and classify whether a mask is being worn.

Attendance Logging: The system logs the attendance of recognized individuals in a CSV file located in the attendance directory.

Exit: Press q to stop the system.

File Structure
Copy
mask-detection-attendance-system/
├── attendance/                # Directory containing attendance logs
├── known_faces/               # Directory containing known face images for recognition
├── maskdetection.h5           # Pre-trained mask detection model
├── attendence.py              # Main script for mask detection and attendance logging
├── face_detetion.py           # Script for face detection (optional)
├── maskdetection.ipynb        # Jupyter notebook for model training and testing
├── requirements.txt           # List of dependencies
└── README.md                  # Project documentation
Contributing
Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.
