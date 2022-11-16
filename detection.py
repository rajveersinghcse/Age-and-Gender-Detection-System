import cv2 as cv
import math
import time
import argparse

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0,(300,300),[104,117,123], True,False)

    net.setInput(blob)
    detections = net.forward()
    faceboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3] * frameWidth)
            y1 = int(detections[0,0,i,4] * frameHeight)
            x2 = int(detections[0,0,i,5] * frameWidth)
            y2 = int(detections[0,0,i,6] * frameHeight)
            faceboxes.append([x1,y1,x2,y2])
            cv.rectangle(frameOpencvDnn, (x1,y1),(x2,y2),(0,225,0), int(round(frameHeight/150)),8)
        return frameOpencvDnn, faceboxes

parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()

faceProto = "face_detector.pbtxt"
faceModel = "face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
genderList = ['Male', 'Female']

# Load Network
faceNet = cv.dnn.readNet(faceModel, faceProto)
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)


#open a video file or an image file or a camera stream
video = cv.VideoCapture(1)
padding = 20
while cv.waitKey(1) <0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameFace, faceboxes = getFaceBox(faceNet, frame)
    if not faceboxes:
        print(" No Face Detected!")
    
    for facebox in faceboxes:
        face = frame[max(0, facebox[1]-padding):min(facebox[3]+padding, frame.shape[0]-1), max(0,facebox[0]-padding):min(facebox[2]+padding, frame.shapep[1]-1)]
        blob = cv.dnn.blobFromImage(face, 1.0, (277,277), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender : {gender}')

        ageNet.setInput(blob)
        agePredes = ageNet.forward()
        age = ageList[agePredes[0].argmax()]
        print(f"Age : {age[1:-1]} years")

        cv.putText(frameFace,f'{gender},{age}',(facebox[0], facebox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow("Decting age and gender", frameFace)