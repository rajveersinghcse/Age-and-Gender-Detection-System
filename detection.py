import cv2
import numpy as np

# Constants
FACE_PROTO = "constants/opencv_face_detector.pbtxt"
FACE_MODEL = "constants/opencv_face_detector_uint8.pb"
AGE_PROTO = "constants/age_deploy.prototxt"
AGE_MODEL = "constants/age_net.caffemodel"
GENDER_PROTO = "constants/gender_deploy.prototxt"
GENDER_MODEL = "constants/gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
CONFIDENCE_THRESHOLD = 0.99

# Load pre-trained models
face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

def detect_age_gender(face):
    # Preprocess the face image for age and gender prediction
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Age prediction
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]

    # Gender prediction
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]

    return age, gender

def main():
    # Open a video capture stream
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Perform face detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE_THRESHOLD:
                x1 = int(detections[0, 0, i, 3] * frame.shape[1])
                y1 = int(detections[0, 0, i, 4] * frame.shape[0])
                x2 = int(detections[0, 0, i, 5] * frame.shape[1])
                y2 = int(detections[0, 0, i, 6] * frame.shape[0])

                face = frame[y1:y2, x1:x2]

                age, gender = detect_age_gender(face)

                label = f"{gender},{age}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Age and Gender Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
