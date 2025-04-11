from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os
import base64

app = FastAPI()

# Setup template rendering
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load Haar cascade
current_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(current_dir, 'models', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/camera_face_detection/")
async def camera_face_detection():
    if os.getenv("DISABLE_CAMERA", "true").lower() == "true":
        return {"error": "Camera access is disabled on the server."}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"error": "Camera not accessible"}

    ret, frame = cap.read()
    if not ret:
        return {"error": "Failed to capture image"}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    output_path = os.path.join(current_dir, 'face_detection_output.png')
    cv2.imwrite(output_path, frame)

    with open(output_path, "rb") as file:
        img_bytes = file.read()

    cap.release()
    return Response(content=img_bytes, media_type="image/png")

@app.post("/process_image/", response_class=HTMLResponse)
async def process_image_ui(
    request: Request,
    file: UploadFile = File(...),
    use_harris: bool = Form(False),
    use_shi_tomasi: bool = Form(False),
    use_fast: bool = Form(False),
    use_orb: bool = Form(False),
    detect_faces: bool = Form(False)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return templates.TemplateResponse("index.html", {"request": request, "result": None})

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if use_harris:
        harris_dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        harris_dst = cv2.dilate(harris_dst, None)
        image[harris_dst > 0.01 * harris_dst.max()] = [0, 0, 255]

    if use_shi_tomasi:
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
        if corners is not None:
            corners = np.intp(corners)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

    if use_fast:
        fast = cv2.FastFeatureDetector_create(threshold=25)
        keypoints = fast.detect(gray, None)
        image = cv2.drawKeypoints(image, keypoints, None, color=(255, 255, 0))

    if use_orb:
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        image = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 255))

    if detect_faces:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

    _, img_encoded = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return templates.TemplateResponse("index.html", {"request": request, "result": img_base64})
