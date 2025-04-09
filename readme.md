# Facial Recognition

This project implements a simple facial recognition system using Python and OpenCV. It detects faces and eyes in images utilizing pre-trained Haar Cascade classifiers.

## Features

- **Face Detection**: Identifies human faces in images.
- **Eye Detection**: Locates eyes within detected faces.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/KyleJackson6/Facial-Recognition-.git
   ```


2. **Navigate to the Project Directory**:

   ```bash
   cd Facial-Recognition-
   ```


3. **Install Dependencies**:

   Ensure you have Python installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```


## Usage

1. **Prepare Your Image**:

   Place the image you want to process in the project directory.

2. **Run the Script**:

   Execute the `main.py` script, specifying your image file:

   ```bash
   python main.py your_image.jpg
   ```


   Replace `your_image.jpg` with the name of your image file.

3. **View Results**:

   The script will display the image with rectangles drawn around detected faces and eyes.

## Files in the Repository

- `main.py`: The main script for detecting faces and eyes in an image.
- `haarcascade_frontalface_default.xml`: Pre-trained Haar Cascade classifier for face detection.
- `haarcascade_eye.xml`: Pre-trained Haar Cascade classifier for eye detection.
- `requirements.txt`: Lists the Python dependencies required for the project.

## Dependencies

- OpenCV
- NumPy

These are specified in the `requirements.txt` file.

## Notes

- Ensure that the Haar Cascade XML files are in the same directory as `main.py` when running the script.
- For optimal results, use clear images with well-lit faces.

## License

This project is licensed under the MIT License.

## Acknowledgments

Special thanks to the OpenCV team for providing the pre-trained Haar Cascade classifiers. 
