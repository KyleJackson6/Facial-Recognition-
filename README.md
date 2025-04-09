
# FastAPI Feature Detection & Face Recognition API

This project provides a FastAPI-based web API for real-time face detection using a webcam and various computer vision feature detection techniques on uploaded images. It uses OpenCV and supports multiple algorithms like Harris, Shi-Tomasi, FAST, ORB, and Haar Cascade face detection.

## Features

- 📸 **Camera-based Face Detection**  
- 🖼️ **Image Upload for Feature Detection**
  - Harris Corner Detection
  - Shi-Tomasi Corner Detection
  - FAST Feature Detection
  - ORB Feature Detection
  - Haar Cascade Face Detection

## Endpoints

### `GET /camera_face_detection/`

Activates your webcam, captures a frame, detects faces using Haar Cascade, and returns an image with rectangles around detected faces.

**Response:**  
- `image/png` with detected faces outlined

### `POST /process_image/`

Uploads an image and applies selected feature detection algorithms. Returns the processed image.

**Form Data Parameters:**
- `file`: (required) The image file to upload
- `use_harris`: (optional) `true` to apply Harris Corner Detection
- `use_shi_tomasi`: (optional) `true` to apply Shi-Tomasi Corner Detection
- `use_fast`: (optional) `true` to apply FAST feature detection
- `use_orb`: (optional) `true` to apply ORB feature detection
- `detect_faces`: (optional) `true` to apply face detection

**Response:**  
- `image/png` with detected features annotated

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/KyleJackson6/Facial-Recognition-.git
   cd Facial-Recognition-
   ```

2. **Install Dependencies:**

   Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

   Install packages:

   ```bash
   pip install fastapi uvicorn opencv-python numpy
   ```

3. **Ensure Haar Cascade File is Available:**

   Place `haarcascade_frontalface_default.xml` inside a `models/` directory in the project root.

## Running the API

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Then go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to use the Swagger UI interface.

## Project Structure

```
.
├── main.py
├── models
│   └── haarcascade_frontalface_default.xml
├── face_detection_output.png  # Output from camera face detection
```

## Notes

- Make sure your webcam is connected and accessible for the `/camera_face_detection/` endpoint.
- All image processing is done using OpenCV.
- Feature colors:
  - Harris: Red dots
  - Shi-Tomasi: Green circles
  - FAST: Yellow dots
  - ORB: Pink dots
  - Face detection: Red rectangles

## License

This project is licensed under the MIT License.

## Author

**Kyle Jackson**
