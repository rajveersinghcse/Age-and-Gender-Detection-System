import cv2 as cv
import time
import argparse

# Constants
FACE_PROTO = "constants/opencv_face_detector.pbtxt"
FACE_MODEL = "constants/opencv_face_detector_uint8.pb"
AGE_PROTO = "constants/age_deploy.prototxt"
AGE_MODEL = "constants/age_net.caffemodel"
GENDER_PROTO = "constants/gender_deploy.prototxt"
GENDER_MODEL = "constants/gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGELIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDERLIST = ['Male', 'Female']

# Argument Parsing
parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument("--device", default="cpu", help="Device to inference on")
args = parser.parse_args()

# Load Networks
face_net = cv.dnn.readNetFromTensorflow(FACE_MODEL, FACE_PROTO)
age_net = cv.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

# Function to get face bounding box
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

# Device Handling
if args.device == "cpu":
    cv.dnn.DNN_TARGET_CPU

elif args.device == "gpu":
    cv.dnn.DNN_BACKEND_CUDA
    cv.dnn.DNN_TARGET_CUDA

# Open a video file or an image file or a camera stream
cap = cv.VideoCapture(args.input if args.input else 0)
padding = 20

while cv.waitKey(1) < 0:
    # Read frame
    t = time.time()
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameFace, bboxes = getFaceBox(face_net, frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue

    for bbox in bboxes:
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

        # Resize face for age and gender prediction
        face = cv.resize(face, (227, 227))

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Gender Prediction
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDERLIST[gender_preds[0].argmax()]

        # Age Prediction
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGELIST[age_preds[0].argmax()]

        label = "{},{}".format(gender, age)
        cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow("Age Gender Demo", frameFace)
