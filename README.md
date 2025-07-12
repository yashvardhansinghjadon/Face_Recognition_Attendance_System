# Face Recognition Attendance System

A Python-based face recognition attendance system that automatically marks attendance using facial recognition technology.
**Live Demo: https://face-recognition-attendance-system-5-z4tt.onrender.com/**

## Features
- **Face Recognition**: Uses OpenCV and face_recognition library for accurate face detection and recognition
- **Automatic Attendance**: Automatically marks attendance when a recognized face is detected
- **Web Interface**: Flask-based web application with user-friendly interface
- **Real-time Processing**: Live video feed processing for instant recognition
- **Attendance Records**: Maintains attendance records in CSV format
- **User Registration**: Add new users to the system with their face data

## Technologies Used
- **Python 3.x**
- **OpenCV** - Computer vision and image processing
- **face_recognition** - Face recognition library
- **Flask** - Web framework
- **HTML/CSS** - Frontend interface

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/yashvardhansinghjadon/Face_Recognition_Attendance_System.git
   cd Face_Recognition_Attendance_System
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requiremnt.txt
   ```

## Usage
1. **Train the system with face data**
   ```bash
   python train.py
   ```

2. **Run the application**
   ```bash
   python app.py
   ```

3. **Access the web interface**
   - Open your browser and go to `http://localhost:5000`
   - Use the interface to register new users or start attendance marking

## Project Structure
```
face_recognition/
├── app.py                 # Main Flask application
├── train.py              # Face training script
├── dataset/              # Face images for training
│   ├── person1/
│   ├── person2/
│   └── ...
├── templates/            # HTML templates
│   ├── front.html
│   ├── register.html
│   ├── capture.html
│   └── attendance.html
├── static/              # CSS and static files
│   └── style.css
├── attendance.csv       # Attendance records
├── users.csv           # User information
├── labels.pickle       # Trained face labels
├── trainer.yml         # Trained face data
└── requiremnt.txt      # Python dependencies
```

## How it Works
1. **Training Phase**: The system is trained with face images of individuals
2. **Recognition Phase**: Live video feed captures faces and compares them with trained data
3. **Attendance Marking**: When a match is found, attendance is automatically recorded
4. **Data Storage**: Attendance records are saved in CSV format with timestamps

## API Endpoints
- `/` - Main page
- `/register` - User registration page
- `/capture` - Face capture interface
- `/attendance` - View attendance records

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
**Yashvardhan Singh Jadon**
- GitHub: [@yashvardhansinghjadon](https://github.com/yashvardhansinghjadon)

## Acknowledgments
- OpenCV community for computer vision tools
- face_recognition library developers
- Flask web framework 
